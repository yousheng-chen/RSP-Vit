from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from config import PROJECT_ROOT, ResNet18_config
from models.efficientnet_b0_official import build_efficientnet_b0
from models.googlenet_official import build_googlenet
from models.mobilenet_v2_official import build_mobilenet_v2
from models.mnasnet1_0_manual import build_mnasnet1_0_manual
from models.mnasnet1_0_official import build_mnasnet1_0
from models.mobilenetv3_small_official import build_mobilenet_v3_small
from models.shufflenetv2_official import build_shufflenet_v2_x1_0
from resnet_train import evaluate_model, get_dataloader, train_epoch


MODEL_BUILDERS = {
    "efficientnet_b0": build_efficientnet_b0,
    "googlenet": build_googlenet,
    "mobilenet_v2": build_mobilenet_v2,
    "mnasnet1_0": build_mnasnet1_0,
    "mnasnet1_0_manual": build_mnasnet1_0_manual,
    "shufflenet_v2_x1_0": build_shufflenet_v2_x1_0,
    "mobilenet_v3_small": build_mobilenet_v3_small,
}


def build_arg_parser(default_model_name: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight CNN baseline on this project.")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=sorted(MODEL_BUILDERS.keys()),
        default=default_model_name,
        required=default_model_name is None,
        help="Lightweight CNN model to train.",
    )
    parser.add_argument("--data-dir", type=str, default=ResNet18_config["data_dir"])
    parser.add_argument("--img-size", type=int, nargs=2, default=ResNet18_config["img_size"])
    parser.add_argument("--batch-size", type=int, default=ResNet18_config["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=ResNet18_config["num_epochs"])
    parser.add_argument("--learning-rate", type=float, default=ResNet18_config["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=ResNet18_config["weight_decay"])
    parser.add_argument("--train-split", type=float, default=ResNet18_config.get("train_split", 0.5))
    parser.add_argument("--val-split", type=float, default=ResNet18_config.get("val_split", 0.25))
    parser.add_argument("--seed", type=int, default=ResNet18_config.get("split_seed", 42))
    parser.add_argument("--img-channels", type=int, default=ResNet18_config.get("img_channels", 3))
    parser.add_argument("--use-pretrained", action="store_true", help="Load official ImageNet pretrained weights.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional output directory override.")
    return parser


def plot_training_history(history: dict, save_path: str) -> None:
    epochs = range(1, len(history["train_losses"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(epochs, history["train_losses"], "b-", label="Training Loss")
    ax1.plot(epochs, history["val_losses"], "r-", label="Validation Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_accs"], "b-", label="Training Accuracy")
    ax2.plot(epochs, history["val_accs"], "r-", label="Validation Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def describe_model(model: nn.Module, model_name: str) -> str:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return "\n".join(
        [
            f"Model name: {model_name}",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
        ]
    )


def train_official_cnn(
    model_name: str,
    data_dir: str,
    img_size: list[int],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    train_split: float,
    val_split: float,
    seed: int,
    img_channels: int,
    use_pretrained: bool,
    checkpoint_dir: str | None = None,
) -> nn.Module:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloader(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        img_channels=img_channels,
        use_pretrained=use_pretrained,
        train_split=train_split,
        val_split=val_split,
        split_seed=seed,
    )
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model_name='{model_name}'. Choices: {sorted(MODEL_BUILDERS.keys())}")

    if model_name == "mnasnet1_0_manual":
        model = MODEL_BUILDERS[model_name](
            num_classes=num_classes,
            use_pretrained=use_pretrained,
            img_channels=img_channels,
        )
    else:
        model = MODEL_BUILDERS[model_name](num_classes=num_classes, use_pretrained=use_pretrained)
    model = model.to(device)
    print(f"Model info:\n{describe_model(model, model_name)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 2), gamma=0.1)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoint", f"{model_name}_official")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
    history = {
        "train_losses": [],
        "train_accs": [],
        "val_losses": [],
        "val_accs": [],
    }

    best_val_acc = 0.0
    start_time = time.time()
    print("\nStart training...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device, desc="Validating")

        history["train_losses"].append(train_loss)
        history["train_accs"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.2f} sec")
    print(f"Best val accuracy: {best_val_acc:.2f}%")

    with open(os.path.join(checkpoint_dir, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    plot_training_history(history, save_path=os.path.join(checkpoint_dir, "training_history.png"))

    print("\nTesting model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate_model(model, test_loader, criterion, device, desc="Testing")
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(checkpoint_dir, "confusion_matrix.png"))

    implementation_source = "manual_project_reimplementation" if model_name == "mnasnet1_0_manual" else "torchvision_official"
    adaptation_notes = [
        "Reused the project's ImageFolder dataloader, resize, normalization, and train/val/test split pipeline.",
    ]
    if model_name == "mnasnet1_0_manual":
        adaptation_notes.insert(
            0,
            "Manually reimplemented the MnasNet1.0 stage layout, inverted residual blocks, and initialization logic; tuned BN momentum to 0.1 for small-data stability.",
        )
    else:
        adaptation_notes.insert(0, "Replaced only the final classifier layer to match the dataset class count.")

    metrics = {
        "model_name": model_name,
        "use_pretrained": use_pretrained,
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "training_time_sec": float(training_time),
        "seed": int(seed),
        "data_dir": str(Path(data_dir).resolve()),
        "img_size": list(img_size),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "train_split": float(train_split),
        "val_split": float(val_split),
        "img_channels": int(img_channels),
        "class_names": class_names,
        "official_architecture": model_name != "mnasnet1_0_manual",
        "implementation_source": implementation_source,
        "project_adaptation": adaptation_notes,
    }
    metrics_path = os.path.join(checkpoint_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    print(f"Saved metrics to {metrics_path}")
    return model


def run_from_args(args: argparse.Namespace) -> nn.Module:
    return train_official_cnn(
        model_name=args.model_name,
        data_dir=args.data_dir,
        img_size=list(args.img_size),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        img_channels=args.img_channels,
        use_pretrained=args.use_pretrained,
        checkpoint_dir=args.checkpoint_dir,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
