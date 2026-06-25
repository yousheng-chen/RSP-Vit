import os
import pickle
import random
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from config import ResNet18_config


# torchvision 官方可直接调用的 ResNet 家族接口
SUPPORTED_TORCHVISION_RESNETS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


def _get_normalization_stats(img_channels: int, use_pretrained: bool) -> Tuple[List[float], List[float]]:
    """
    选择归一化参数。

    - 如果使用 ImageNet 预训练权重，并且输入是 RGB，则使用 ImageNet 标准均值/方差。
    - 否则使用更通用的 0.5 / 0.5 归一化。
    """
    if use_pretrained and img_channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5] * img_channels
        std = [0.5] * img_channels
    return mean, std


def get_dataloader(
    data_dir: str,
    img_size: List[int] | None = None,
    batch_size: int | None = None,
    img_channels: int | None = None,
    use_pretrained: bool | None = None,
    train_split: float | None = None,
    val_split: float | None = None,
    split_seed: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    构建 train / val / test 三个 DataLoader。

    这里的逻辑是：
    1. 从一个 ImageFolder 根目录读取全部样本
    2. 按配置比例随机切分成 train / val / test
    3. 返回三个 DataLoader 和类别名列表
    """
    if img_size is None:
        img_size = ResNet18_config['img_size']
    if batch_size is None:
        batch_size = ResNet18_config['batch_size']
    if img_channels is None:
        img_channels = ResNet18_config.get('img_channels', 3)
    if use_pretrained is None:
        use_pretrained = ResNet18_config.get('use_pretrained', False)
    if train_split is None:
        train_split = ResNet18_config.get('train_split', 0.5)
    if val_split is None:
        val_split = ResNet18_config.get('val_split', 0.25)
    if split_seed is None:
        split_seed = ResNet18_config.get('split_seed', 42)

    if train_split <= 0 or val_split <= 0 or (train_split + val_split) >= 1:
        raise ValueError(
            f"Invalid split ratio: train_split={train_split}, val_split={val_split}. "
            f"Expected train_split > 0, val_split > 0, and train_split + val_split < 1."
        )

    if img_channels not in (1, 3):
        raise ValueError("This training script currently supports only img_channels=1 or img_channels=3.")

    mean, std = _get_normalization_stats(img_channels, use_pretrained)

    transform_steps = [transforms.Resize((img_size[0], img_size[1]))]
    if img_channels == 1:
        # 如果想用单通道模型，这里把图片先转成灰度图。
        transform_steps.append(transforms.Grayscale(num_output_channels=1))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    transform = transforms.Compose(transform_steps)

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # 用固定 generator 保证“每次重新运行脚本时”切分尽量一致。
    split_generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=split_generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Total samples: {total_size}")
    print(
        "Split sizes -> "
        f"train: {train_size} ({train_split:.0%}), "
        f"val: {val_size} ({val_split:.0%}), "
        f"test: {test_size} ({1 - train_split - val_split:.0%})"
    )
    print(
        "Batches per epoch -> "
        f"train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)} "
        f"(batch_size={batch_size})"
    )

    return train_loader, val_loader, test_loader, dataset.classes


def _get_torchvision_weights(model_name: str, use_pretrained: bool):
    """
    通用方式获取 torchvision 模型权重。

    use_pretrained=False 时返回 None，表示从头训练。
    use_pretrained=True 时尝试取该模型的 DEFAULT 权重。
    """
    if not use_pretrained:
        return None

    try:
        return models.get_model_weights(model_name).DEFAULT
    except Exception as exc:
        raise RuntimeError(
            f"Failed to get pretrained weights for model '{model_name}'. "
            f"If you are offline, set use_pretrained=False. Original error: {exc}"
        ) from exc


def _replace_first_conv_if_needed(
    model: nn.Module,
    img_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    use_pretrained: bool,
) -> None:
    """
    根据配置替换官方模型的第一层卷积。

    为什么要单独写这个函数：
    - torchvision 官方接口本身不直接暴露 stem 参数
    - 但很多实验会想修改输入通道数、卷积核大小、stride、padding
    - 所以这里先构建官方模型，再按需替换 conv1
    """
    old_conv = model.conv1
    needs_replace = (
        old_conv.in_channels != img_channels
        or old_conv.kernel_size != (kernel_size, kernel_size)
        or old_conv.stride != (stride, stride)
        or old_conv.padding != (padding, padding)
    )

    if not needs_replace:
        return

    new_conv = nn.Conv2d(
        in_channels=img_channels,
        out_channels=old_conv.out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )

    # 如果使用了预训练权重，并且只是在通道数上做 3 -> 1 的改动，
    # 同时卷积核大小/步长/padding 没改，我们可以把 RGB 权重取均值后迁移过去。
    can_reuse_pretrained_stem = (
        use_pretrained
        and old_conv.kernel_size == (kernel_size, kernel_size)
        and old_conv.stride == (stride, stride)
        and old_conv.padding == (padding, padding)
    )

    if can_reuse_pretrained_stem and old_conv.in_channels == 3 and img_channels == 1:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    else:
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.conv1 = new_conv


def _replace_maxpool_if_needed(
    model: nn.Module,
    use_maxpool: bool,
    kernel_size: int,
    stride: int,
    padding: int,
) -> None:
    """
    按配置替换或关闭 maxpool。

    小图像任务里，很多人会选择关闭 maxpool，减少前期下采样过快的问题。
    """
    if not use_maxpool:
        model.maxpool = nn.Identity()
        return

    old_pool = model.maxpool
    needs_replace = (
        not isinstance(old_pool, nn.MaxPool2d)
        or old_pool.kernel_size != kernel_size
        or old_pool.stride != stride
        or old_pool.padding != padding
    )

    if needs_replace:
        model.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)


def build_torchvision_resnet(
    model_name: str | None = None,
    num_classes: int = 6,
    use_pretrained: bool | None = None,
    img_channels: int | None = None,
    first_conv_kernel_size: int | None = None,
    first_conv_stride: int | None = None,
    first_conv_padding: int | None = None,
    use_maxpool: bool | None = None,
    maxpool_kernel_size: int | None = None,
    maxpool_stride: int | None = None,
    maxpool_padding: int | None = None,
) -> nn.Module:
    """
    使用 torchvision 官方接口构建 ResNet 家族模型。

    这是你现在最推荐使用的入口。
    好处：
    - 结构稳定，和官方实现保持一致
    - 可以直接切换 resnet18 / resnet34 / resnet50 等
    - 仍然保留你实验里常用的 stem 参数调整能力
    """
    if model_name is None:
        model_name = ResNet18_config.get('model_name', 'resnet18')
    if use_pretrained is None:
        use_pretrained = ResNet18_config.get('use_pretrained', False)
    if img_channels is None:
        img_channels = ResNet18_config.get('img_channels', 3)
    if first_conv_kernel_size is None:
        first_conv_kernel_size = ResNet18_config.get('first_conv_kernel_size', 7)
    if first_conv_stride is None:
        first_conv_stride = ResNet18_config.get('first_conv_stride', 2)
    if first_conv_padding is None:
        first_conv_padding = ResNet18_config.get('first_conv_padding', 3)
    if use_maxpool is None:
        use_maxpool = ResNet18_config.get('use_maxpool', True)
    if maxpool_kernel_size is None:
        maxpool_kernel_size = ResNet18_config.get('maxpool_kernel_size', 3)
    if maxpool_stride is None:
        maxpool_stride = ResNet18_config.get('maxpool_stride', 2)
    if maxpool_padding is None:
        maxpool_padding = ResNet18_config.get('maxpool_padding', 1)

    if model_name not in SUPPORTED_TORCHVISION_RESNETS:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            f"Choose one of: {', '.join(SUPPORTED_TORCHVISION_RESNETS)}"
        )

    weights = _get_torchvision_weights(model_name, use_pretrained)
    model = models.get_model(model_name, weights=weights)

    _replace_first_conv_if_needed(
        model=model,
        img_channels=img_channels,
        kernel_size=first_conv_kernel_size,
        stride=first_conv_stride,
        padding=first_conv_padding,
        use_pretrained=use_pretrained,
    )
    _replace_maxpool_if_needed(
        model=model,
        use_maxpool=use_maxpool,
        kernel_size=maxpool_kernel_size,
        stride=maxpool_stride,
        padding=maxpool_padding,
    )

    # 官方模型默认 num_classes=1000，这里替换成你当前任务的类别数。
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def describe_model(model: nn.Module, model_name: str) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info_lines = [
        f"Model name: {model_name}",
        f"Input channels: {model.conv1.in_channels}",
        f"First conv: kernel={model.conv1.kernel_size}, stride={model.conv1.stride}, padding={model.conv1.padding}",
        f"Using maxpool: {not isinstance(model.maxpool, nn.Identity)}",
        f"Number of classes: {model.fc.out_features}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
    ]
    return "\n".join(info_lines)


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 这里 tqdm 显示的是 batch 数，不是图片总数。
    # 因为 train_loader 每次迭代返回的是一个 batch。
    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion, device, desc: str = "Evaluating"):
    """
    一次遍历同时完成：
    - 平均 loss
    - accuracy
    - preds / labels 收集

    这样测试集就不用跑两遍了。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy, all_preds, all_labels


def train_resnet(num_epochs: int | None = None, batch_size: int | None = None, learning_rate: float | None = None):
    if num_epochs is None:
        num_epochs = ResNet18_config['num_epochs']
    if batch_size is None:
        batch_size = ResNet18_config['batch_size']
    if learning_rate is None:
        learning_rate = ResNet18_config['learning_rate']

    seed = ResNet18_config.get('split_seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = ResNet18_config.get('model_name', 'resnet18')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloader(
        data_dir=ResNet18_config['data_dir'],
        img_size=ResNet18_config['img_size'],
        batch_size=batch_size,
        img_channels=ResNet18_config.get('img_channels', 3),
        use_pretrained=ResNet18_config.get('use_pretrained', False),
        train_split=ResNet18_config.get('train_split', 0.5),
        val_split=ResNet18_config.get('val_split', 0.25),
        split_seed=ResNet18_config.get('split_seed', 42),
    )

    print(f"Classes: {class_names}")
    n_classes = len(class_names)

    print(f"Building torchvision model: {model_name} ...")
    model = build_torchvision_resnet(model_name=model_name, num_classes=n_classes)
    model = model.to(device)
    print(f"Model info:\n{describe_model(model, model_name)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=ResNet18_config['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    checkpoint_dir = ResNet18_config['checkpoint_dir']
    best_model_name = f"best_{model_name}.pth"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    print("\nStart training...")
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device, desc="Validating")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, best_model_name)
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} sec")
    print(f"Best val accuracy: {best_val_acc:.2f}%")

    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }
    history_path = os.path.join(checkpoint_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Saved training history to {history_path}")

    print("\nTesting model...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, best_model_name)))
    test_loss, test_acc, preds, labels = evaluate_model(model, test_loader, criterion, device, desc="Testing")

    plot_training_history(history, save_path=os.path.join(checkpoint_dir, 'training_history.png'))
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(checkpoint_dir, 'confusion_matrix.png'))

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return model


# 兼容你之前的调用方式：原来脚本里调用的是 train_resnet18()
def train_resnet18(num_epochs: int | None = None, batch_size: int | None = None, learning_rate: float | None = None):
    return train_resnet(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)


def plot_training_history(history, save_path=None):
    epochs = range(1, len(history['train_losses']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    train_resnet18()
