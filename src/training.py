from models.vit import (
    VisionTransformer,
    PatchEmbeddings,
    ResNetStemPatchEmbeddings,
    LearnedPositionalEmbeddings,
    ClassificationHead,
)
from models.transformerlayer import TransformerLayer
from models.mha import MultiHeadAttention
from models.ffn import FeedForward, LocalEnhancedFeedForward
from config import config

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import argparse
import json
import subprocess
import sys

class AddGaussianNoise(nn.Module):
    """
    Additive Gaussian noise for image tensors in [0, 1].

    Notes:
    - Intended for robustness experiments on spectrogram-like images.
    - Apply after ToTensor() and before Normalize().
    - noise_std is the standard deviation in [0,1] value space (e.g. 0.05 == 5%).
    """

    def __init__(self, std: float, p: float = 1.0, clamp_min: float = 0.0, clamp_max: float = 1.0):
        super().__init__()
        self.std = float(std)
        self.p = float(p)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0 or self.p <= 0.0:
            return x
        if self.p < 1.0:
            # Use torch RNG so the existing seed logic controls repeatability.
            if torch.rand(1).item() >= self.p:
                return x
        x = x + torch.randn_like(x) * self.std
        return x.clamp(self.clamp_min, self.clamp_max)

def get_dataloader(
    data_dir,
    img_size: list = None,
    batch_size: int = None,
    noise_std: float = 0.0,
    noise_prob: float = 0.0,
    train_split: float = None,
    val_split: float = None,
    split_seed: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if img_size is None:
        img_size = config['img_size']
    if batch_size is None:
        batch_size = config['batch_size']
    if train_split is None:
        train_split = config.get('train_split', 0.2)
    if val_split is None:
        val_split = config.get('val_split', 0.4)
    if split_seed is None:
        split_seed = config.get('seed', config.get('split_seed', 42))
    if train_split <= 0 or val_split <= 0 or (train_split + val_split) >= 1:
        raise ValueError(
            f"Invalid split ratio: train_split={train_split}, val_split={val_split}. "
            "Expected train_split > 0, val_split > 0, and train_split + val_split < 1."
        )
    
    tfs = [
        transforms.Resize((img_size[0], img_size[1])),
        transforms.ToTensor(),
    ]
    # Add noise after ToTensor (values in [0,1]) and before Normalize.
    if noise_std and noise_prob:
        tfs.append(AddGaussianNoise(std=float(noise_std), p=float(noise_prob)))
    tfs.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform = transforms.Compose(tfs)
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
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


def _build_ffn_module(ffn_variant: str, d_model: int, d_ff: int):
    if ffn_variant == "standard":
        return FeedForward(d_model=d_model, d_ff=d_ff)
    if ffn_variant == "local_enhanced":
        return LocalEnhancedFeedForward(d_model=d_model, d_ff=d_ff)
    raise ValueError(
        f"Unsupported ffn_variant='{ffn_variant}'. "
        "Choices: standard, local_enhanced, first_layer_local_enhanced"
    )


def _build_transformer_encoder_layer(
    d_model: int,
    n_heads: int,
    feed_forward: nn.Module,
    local_ffn_first_layer_only: bool = False,
):
    mha = MultiHeadAttention(heads=n_heads, d_model=d_model)
    return TransformerLayer(
        d_model=d_model,
        self_attn=mha,
        feed_forward=feed_forward,
        local_ffn_first_layer_only=local_ffn_first_layer_only,
        dropout_prob=0.1,
    )


def _build_transformer_stack(
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    share_transformer_weights: bool,
    ffn_variant: str,
):
    if ffn_variant == "first_layer_local_enhanced":
        return _build_transformer_encoder_layer(
            d_model=d_model,
            n_heads=n_heads,
            feed_forward=_build_ffn_module("local_enhanced", d_model=d_model, d_ff=d_ff),
            local_ffn_first_layer_only=True,
        )

    return _build_transformer_encoder_layer(
        d_model=d_model,
        n_heads=n_heads,
        feed_forward=_build_ffn_module(ffn_variant, d_model=d_model, d_ff=d_ff),
    )


def build_vit_model(d_model: int = None, n_heads: int = None, n_layers: int = None, 
                   patch_size: int = None, n_classes: int = None, d_ff: int = None,
                    share_transformer_weights: bool = None,
                    use_depth_embeddings: bool = None,
                    ffn_variant: str = None,
                    patch_embed_variant: str = None):
    if d_model is None:
        d_model = config['d_model']
    if n_heads is None:
        n_heads = config['n_heads']
    if n_layers is None:
        n_layers = config['n_layers']
    if patch_size is None:
        patch_size = config['patch_size']
    if d_ff is None:
        d_ff = config['d_ff']
    if share_transformer_weights is None:
        share_transformer_weights = config.get('share_transformer_weights', False)
    if use_depth_embeddings is None:
        use_depth_embeddings = config.get('use_depth_embeddings', False)
    if ffn_variant is None:
        ffn_variant = config.get('ffn_variant', 'standard')
    if patch_embed_variant is None:
        patch_embed_variant = config.get('patch_embed_variant', 'standard')
    
    """
    构建完整的Vision Transformer模型
    
    Args:
        d_model: 隐藏维度
        n_heads: 多头注意力头数
        n_layers: Transformer层数
        patch_size: patch大小
        n_classes: 分类类别数
        d_ff: 前馈网络中间维度
    """
    # 1. Patch Embeddings
    if patch_embed_variant == "standard":
        patch_emb = PatchEmbeddings(d_model=d_model, patch_size=patch_size, in_channels=3)
    elif patch_embed_variant == "resnet_stem":
        patch_emb = ResNetStemPatchEmbeddings(
            d_model=d_model,
            patch_size=patch_size,
            in_channels=3,
            stem_channels=tuple(config.get('conv_stem_channels', [32, 64])),
        )
    else:
        raise ValueError(
            f"Unsupported patch_embed_variant='{patch_embed_variant}'. "
            "Choices: standard, resnet_stem"
        )
    
    # 2. Positional Embeddings
    pos_emb = LearnedPositionalEmbeddings(d_model=d_model, max_len=5000)
    
    # 3. Transformer Layer / Layer Stack
    transformer_layer = _build_transformer_stack(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        share_transformer_weights=share_transformer_weights,
        ffn_variant=ffn_variant,
    )
    
    # 4. Classification Head
    classification_head = ClassificationHead(d_model=d_model, n_hidden=d_model, n_classes=n_classes)
    
    # 5. Vision Transformer
    model = VisionTransformer(
        transformer_layer=transformer_layer,
        n_layers=n_layers,
        patch_emb=patch_emb,
        pos_emb=pos_emb,
        classification=classification_head,
        share_transformer_weights=share_transformer_weights,
        use_depth_embeddings=use_depth_embeddings,
    )
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': total_loss / (progress_bar.n + 1),
            'acc': 100 * correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def _resolve_run_name(run_name: str) -> str:
    checkpoint_root = config["checkpoint_dir"]
    if not os.path.exists(os.path.join(checkpoint_root, run_name)):
        return run_name

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = f"{run_name}_{timestamp}"
    suffix = 1
    while os.path.exists(os.path.join(checkpoint_root, candidate)):
        candidate = f"{run_name}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def _resolve_compare_dir(base_name: str) -> str:
    checkpoint_root = config["checkpoint_dir"]
    candidate = os.path.join(checkpoint_root, base_name)
    if not os.path.exists(candidate):
        return candidate

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = os.path.join(checkpoint_root, f"{base_name}_{timestamp}")
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(checkpoint_root, f"{base_name}_{timestamp}_{suffix}")
        suffix += 1
    return candidate


def train_vit(
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    img_size: list = None,
    train_split: float = None,
    val_split: float = None,
    seed: int = None,
    share_transformer_weights: bool = None,
    use_depth_embeddings: bool = None,
    ffn_variant: str = None,
    patch_embed_variant: str = None,
    run_name: str = None,
    data_dir: str = None,
    noise_std: float = 0.0,
    noise_prob: float = 0.0,
    show_plots: bool = False,
):
    if num_epochs is None:
        num_epochs = config['num_epochs']
    if batch_size is None:
        batch_size = config['batch_size']
    if learning_rate is None:
        learning_rate = config['learning_rate']
    if weight_decay is None:
        weight_decay = config.get('weight_decay', 1e-5)
    if img_size is None:
        img_size = config['img_size']
    if train_split is None:
        train_split = config.get('train_split', 0.5)
    if val_split is None:
        val_split = config.get('val_split', 0.25)
    if seed is None:
        seed = config.get('seed', config.get('split_seed', 42))
    if share_transformer_weights is None:
        share_transformer_weights = config.get('share_transformer_weights', False)
    if use_depth_embeddings is None:
        use_depth_embeddings = config.get('use_depth_embeddings', False)
    if ffn_variant is None:
        ffn_variant = config.get('ffn_variant', 'standard')
    if patch_embed_variant is None:
        patch_embed_variant = config.get('patch_embed_variant', 'standard')
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = "shared" if share_transformer_weights else "unshared"
        depth_mode = "depth" if use_depth_embeddings else "plain"
        run_name = f"{mode}_{depth_mode}_{ffn_variant}_{patch_embed_variant}_{timestamp}"
    run_name = _resolve_run_name(run_name)
    
    """
    训练ViT模型
    """
    # 设置随机种子以确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("加载数据集...")
    effective_data_dir = data_dir if data_dir is not None else config['data_dir']
    train_loader, val_loader, test_loader, class_names = get_dataloader(
        effective_data_dir,
        img_size=img_size,
        batch_size=batch_size,
        noise_std=noise_std,
        noise_prob=noise_prob,
        train_split=train_split,
        val_split=val_split,
        split_seed=seed,
    )
    print(f"类别: {class_names}")
    n_classes = len(class_names)
    
    # 构建模型
    print("构建ViT模型...")
    model = build_vit_model(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        patch_size=config['patch_size'],
        n_classes=n_classes,
        d_ff=config['d_ff'],
        share_transformer_weights=share_transformer_weights,
        use_depth_embeddings=use_depth_embeddings,
        ffn_variant=ffn_variant,
        patch_embed_variant=patch_embed_variant,
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"Transformer参数共享: {share_transformer_weights}")
    
    # 损失函数和优化器
    print(f"Depth embeddings enabled: {use_depth_embeddings}")
    print(f"FFN variant: {ffn_variant}")
    print(f"Patch embedding variant: {patch_embed_variant}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 创建checkpoint目录
    checkpoint_dir = os.path.join(config['checkpoint_dir'], run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint目录: {checkpoint_dir}")
    
    # 初始化历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 训练循环
    best_val_acc = 0.0
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n第 {epoch+1}/{num_epochs} 个Epoch")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        
        print(f"训练损失: {train_loss:.4f}, 训练精度: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证精度: {val_acc:.2f}%")
        
        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, "best_vit_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"保存最好的模型到 {checkpoint_path}")
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
    print(f"最佳验证精度: {best_val_acc:.2f}%")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
    with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"保存训练历史到 {os.path.join(checkpoint_dir, 'training_history.pkl')}")
    
    # 测试
    print("\n测试模型...")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_vit_model.pth")))

        
    # 加载训练历史并绘制
    with open(os.path.join(checkpoint_dir, 'training_history.pkl'), 'rb') as f:
        history = pickle.load(f)
    plot_training_history(
        history,
        save_path=os.path.join(checkpoint_dir, 'training_history.png'),
        show=show_plots,
    )
    
    # 加载最好的模型
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_vit_model.pth")))
    
    
    # 计算预测和标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = get_predictions_and_labels(model, test_loader, device)
    features, feature_labels = extract_penultimate_features(model, test_loader, device)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        cm,
        class_names,
        save_path=os.path.join(checkpoint_dir, 'confusion_matrix.png'),
        show=show_plots,
    )
    plot_feature_embedding(
        features,
        feature_labels,
        class_names,
        save_path=os.path.join(checkpoint_dir, 'feature_tsne.png'),
        show=show_plots,
        method="tsne",
    )
    plot_class_relation_map(
        np.asarray(labels),
        np.asarray(preds),
        class_names,
        save_path=os.path.join(checkpoint_dir, 'class_relation_map.png'),
        show=show_plots,
    )
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"测试损失: {test_loss:.4f}, 测试精度: {test_acc:.2f}%")

    metrics = {
        "run_name": run_name,
        "share_transformer_weights": share_transformer_weights,
        "use_depth_embeddings": use_depth_embeddings,
        "ffn_variant": ffn_variant,
        "patch_embed_variant": patch_embed_variant,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "training_time_sec": float(training_time),
        "seed": int(seed),
        "data_dir": effective_data_dir,
        "img_size": list(img_size),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_split": float(train_split),
        "val_split": float(val_split),
        "test_split": float(1 - train_split - val_split),
        "noise_std": float(noise_std),
        "noise_prob": float(noise_prob),
    }
    with open(os.path.join(checkpoint_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"保存评估指标到 {os.path.join(checkpoint_dir, 'metrics.json')}")
    
    
    return model


def _parse_args():
    parser = argparse.ArgumentParser(description="Train ViT (with optional weight sharing)")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=None)
    parser.add_argument("--train-split", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None, help="Override config['data_dir'] for this run.")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Additive Gaussian noise std in [0,1] tensor space (after ToTensor).")
    parser.add_argument("--noise-prob", type=float, default=0.0, help="Probability to apply noise to each sample (0 disables).")
    parser.add_argument("--show-plots", action="store_true", help="Show matplotlib windows (blocks until closed). Default: off (save PNGs only).")

    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument(
        "--share-transformer-weights",
        dest="share_transformer_weights",
        action="store_true",
        help="Share Transformer layer weights across depth (weight tying).",
    )
    share_group.add_argument(
        "--no-share-transformer-weights",
        dest="share_transformer_weights",
        action="store_false",
        help="Do not share Transformer layer weights (standard ViT).",
    )
    parser.set_defaults(share_transformer_weights=None)

    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--use-depth-embeddings",
        dest="use_depth_embeddings",
        action="store_true",
        help="Add learnable depth identity embeddings before each Transformer layer.",
    )
    depth_group.add_argument(
        "--no-depth-embeddings",
        dest="use_depth_embeddings",
        action="store_false",
        help="Disable depth identity embeddings.",
    )
    parser.set_defaults(use_depth_embeddings=None)

    parser.add_argument(
        "--ffn-variant",
        type=str,
        choices=["standard", "local_enhanced", "first_layer_local_enhanced"],
        default=None,
        help="Choose the FFN implementation inside the Transformer block.",
    )
    parser.add_argument(
        "--patch-embed-variant",
        type=str,
        choices=["standard", "resnet_stem"],
        default=None,
        help="Choose the patch embedding frontend. 'resnet_stem' adds a lightweight CNN-style stem before tokenization.",
    )

    parser.add_argument(
        "--compare-both",
        action="store_true",
        help="Run all four combinations of shared/unshared and depth on/off, then generate comparison outputs.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def plot_training_history(history, save_path=None, show: bool = False):
    """绘制训练历史"""
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失函数
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率
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
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path=None, show: bool = False):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def extract_penultimate_features(model, data_loader, device):
    """
    提取分类头前的高层特征，用于 t-SNE / PCA 可视化。
    对当前 ViT 实现来说，classification.linear1 的输入就是最终分类前的特征表示。
    """
    model.eval()
    captured_features = []
    all_labels = []

    def _hook(_module, inputs, _output):
        captured_features.append(inputs[0].detach().cpu())

    handle = model.classification.linear1.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                _ = model(images)
                all_labels.append(labels.detach().cpu())
    finally:
        handle.remove()

    features = torch.cat(captured_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return features, labels


def plot_feature_embedding(features, labels, class_names, save_path=None, show: bool = False, method: str = "tsne"):
    """
    把高维特征降到二维，画真实的特征散点图。
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={features.shape}")

    reduced_input = features
    if features.shape[1] > 50:
        pca = PCA(n_components=50, random_state=42)
        reduced_input = pca.fit_transform(features)

    if method.lower() == "tsne":
        perplexity = min(30, max(5, (len(features) - 1) // 3))
        projector = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        embedding = projector.fit_transform(reduced_input)
        title = "t-SNE Feature Visualization"
    elif method.lower() == "pca":
        projector = PCA(n_components=2, random_state=42)
        embedding = projector.fit_transform(features)
        title = "PCA Feature Visualization"
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[int(label)] if int(label) < len(class_names) else str(label)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=18,
            alpha=0.75,
            color=cmap(idx),
            label=class_name,
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", fontsize=9, markerscale=1.2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def _greedy_similarity_order(similarity: np.ndarray):
    n = similarity.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]

    remaining = set(range(n))
    start = int(np.argmax(similarity.sum(axis=1)))
    order = [start]
    remaining.remove(start)

    while remaining:
        last = order[-1]
        next_idx = max(remaining, key=lambda j: similarity[last, j])
        order.append(next_idx)
        remaining.remove(next_idx)

    return order


def plot_class_relation_map(labels, preds, class_names, save_path=None, show: bool = False):
    """
    绘制一个更适合展示的类别关系图：
    - 每个类别放在圆周上的一个方向
    - 相互混淆更多的类别尽量相邻
    - 预测错误的点会朝被预测类别方向偏移，表达“类间交融”
    """
    n_classes = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    similarity = cm + cm.T
    np.fill_diagonal(similarity, 0)

    order = _greedy_similarity_order(similarity)
    radius = 10.0
    centers = {}
    for pos, class_idx in enumerate(order):
        angle = 2 * math.pi * pos / max(n_classes, 1)
        centers[class_idx] = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("tab10", n_classes)
    rng = np.random.default_rng(42)

    circle = plt.Circle((0, 0), radius, color="lightgray", fill=False, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.add_patch(circle)

    for class_idx in order:
        center = centers[class_idx]
        ax.plot([0, center[0]], [0, center[1]], color="lightgray", linewidth=0.8, alpha=0.5)
        ax.scatter(center[0], center[1], s=180, color=cmap(class_idx), edgecolors="black", linewidths=0.8, zorder=3)
        label_pos = center * 1.12
        ax.text(
            label_pos[0],
            label_pos[1],
            class_names[class_idx],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    for true_idx in range(n_classes):
        mask = labels == true_idx
        true_preds = preds[mask]
        if mask.sum() == 0:
            continue

        true_center = centers[true_idx]
        xs = []
        ys = []
        for pred_idx in true_preds:
            pred_idx = int(pred_idx)
            pred_center = centers[pred_idx]

            if pred_idx == true_idx:
                base = true_center
                jitter = rng.normal(loc=0.0, scale=0.65, size=2)
            else:
                base = true_center * 0.55 + pred_center * 0.45
                jitter = rng.normal(loc=0.0, scale=0.55, size=2)

            point = base + jitter
            xs.append(point[0])
            ys.append(point[1])

        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.72,
            color=cmap(true_idx),
            edgecolors="none",
            label=class_names[true_idx],
        )

    ax.set_title("Class Relation Map (Confusion-Aware Layout)")
    ax.set_xlabel("Relation axis 1")
    ax.set_ylabel("Relation axis 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    handles, labels_legend = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels_legend):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=9, markerscale=1.2)

    ax.text(
        0.02,
        0.02,
        "Note: stylized relation map based on confusion, not a true geometric embedding.",
        transform=ax.transAxes,
        fontsize=9,
        color="dimgray",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def get_predictions_and_labels(model, test_loader, device):
    """获取模型在测试集上的预测和真实标签"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

if __name__ == "__main__":
    args = _parse_args()
    if args.compare_both:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = args.run_name if args.run_name else f"exp_{timestamp}"
        compare_runs = []
        for share_transformer_weights, use_depth_embeddings in (
            (True, False),
            (True, True),
            (False, False),
            (False, True),
        ):
            share_tag = "shared" if share_transformer_weights else "unshared"
            depth_tag = "depth" if use_depth_embeddings else "plain"
            run_name = _resolve_run_name(f"{base_name}_{share_tag}_{depth_tag}")
            compare_runs.append(run_name)

            train_vit(
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                img_size=args.img_size,
                train_split=args.train_split,
                val_split=args.val_split,
                seed=args.seed,
                share_transformer_weights=share_transformer_weights,
                use_depth_embeddings=use_depth_embeddings,
                ffn_variant=args.ffn_variant,
                patch_embed_variant=args.patch_embed_variant,
                run_name=run_name,
                data_dir=args.data_dir,
                noise_std=args.noise_std,
                noise_prob=args.noise_prob,
                show_plots=args.show_plots,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_dir = _resolve_compare_dir(f"compare_{base_name}")
        subprocess.run(
            [
                sys.executable,
                os.path.join("src", "compare_runs.py"),
                "--checkpoint-root",
                config["checkpoint_dir"],
                "--runs",
                *compare_runs,
                "--out-dir",
                out_dir,
            ],
            check=True,
        )
    else:
        train_vit(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
            train_split=args.train_split,
            val_split=args.val_split,
            seed=args.seed,
            share_transformer_weights=args.share_transformer_weights,
            use_depth_embeddings=args.use_depth_embeddings,
            ffn_variant=args.ffn_variant,
            patch_embed_variant=args.patch_embed_variant,
            run_name=args.run_name,
            data_dir=args.data_dir,
            noise_std=args.noise_std,
            noise_prob=args.noise_prob,
            show_plots=args.show_plots,
        )
