import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def get_data_loaders(data_dir, batch_size=32, num_workers=4, transform=None):
    """
    加载processed_data数据集，并按80%/10%/10%分割为训练集、测试集和验证集
    
    Args:
        data_dir: 数据集根目录路径
        batch_size: 批次大小，默认32
        num_workers: 数据加载器的工作进程数，默认4
        transform: 数据增强和预处理的transforms，如果为None则使用默认设置
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_names: 类别名称列表
    """
    
    # 默认的数据预处理
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整为Vision Transformer所需尺寸
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    # 使用ImageFolder加载数据
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    
    print(f"数据集信息:")
    print(f"  类别数: {len(class_names)}")
    print(f"  类别: {class_names}")
    print(f"  总样本数: {len(full_dataset)}")
    
    # 获取所有索引
    indices = list(range(len(full_dataset)))
    
    # 第一步：分割为训练集(80%)和临时集(20%)
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42,
        stratify=[full_dataset.targets[i] for i in indices]
    )
    
    # 第二步：将临时集(20%)平均分为验证集和测试集(各10%)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=42,
        stratify=[full_dataset.targets[i] for i in temp_indices]
    )
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 打印数据集分割信息
    print(f"\n数据集分割:")
    print(f"  训练集: {len(train_dataset)} ({len(train_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  验证集: {len(val_dataset)} ({len(val_dataset)/len(full_dataset)*100:.1f}%)")
    print(f"  测试集: {len(test_dataset)} ({len(test_dataset)/len(full_dataset)*100:.1f}%)")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # 测试代码
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, "data", "processed_data")
    
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0  # Windows系统推荐设置为0
    )
    
    # 查看第一个批次的数据
    for images, labels in train_loader:
        print(f"\n批次信息:")
        print(f"  图片形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  标签: {labels}")
        print(f"  对应类别: {[class_names[l] for l in labels]}")
        break
