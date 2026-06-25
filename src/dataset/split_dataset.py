import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


def split_and_organize_dataset(processed_data_dir, output_data_dir):
    """
    将processed_data中的数据集分割为训练集、验证集、测试集，
    并按类别组织到train、val、test目录下
    
    Args:
        processed_data_dir: processed_data的路径
        output_data_dir: data目录的路径（将在其中创建train、val、test目录）
    """
    
    # 创建输出目录结构
    train_dir = os.path.join(output_data_dir, "train")
    val_dir = os.path.join(output_data_dir, "val")
    test_dir = os.path.join(output_data_dir, "test")
    
    # 获取所有类别
    classes = sorted([d for d in os.listdir(processed_data_dir) 
                      if os.path.isdir(os.path.join(processed_data_dir, d))])
    
    print(f"发现类别: {classes}")
    print(f"\n开始分割数据集...")
    
    # 为每个类别创建目录
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    total_files = 0
    total_train = 0
    total_val = 0
    total_test = 0
    
    # 对每个类别分别进行分割
    for class_name in classes:
        class_dir = os.path.join(processed_data_dir, class_name)
        image_files = sorted([f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        class_total = len(image_files)
        total_files += class_total
        
        print(f"\n类别: {class_name}")
        print(f"  总文件数: {class_total}")
        
        # 转换为索引
        indices = np.arange(class_total)
        
        # 第一步：分割为训练集(80%)和临时集(20%)
        train_indices, temp_indices = train_test_split(
            indices,
            test_size=0.2,
            random_state=42
        )
        
        # 第二步：将临时集平均分为验证集和测试集(各10%)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=0.5,
            random_state=42
        )
        
        # 复制文件到对应目录
        for idx in train_indices:
            src = os.path.join(class_dir, image_files[idx])
            dst = os.path.join(train_dir, class_name, image_files[idx])
            shutil.copy2(src, dst)
        total_train += len(train_indices)
        
        for idx in val_indices:
            src = os.path.join(class_dir, image_files[idx])
            dst = os.path.join(val_dir, class_name, image_files[idx])
            shutil.copy2(src, dst)
        total_val += len(val_indices)
        
        for idx in test_indices:
            src = os.path.join(class_dir, image_files[idx])
            dst = os.path.join(test_dir, class_name, image_files[idx])
            shutil.copy2(src, dst)
        total_test += len(test_indices)
        
        print(f"  训练集: {len(train_indices)} ({len(train_indices)/class_total*100:.1f}%)")
        print(f"  验证集: {len(val_indices)} ({len(val_indices)/class_total*100:.1f}%)")
        print(f"  测试集: {len(test_indices)} ({len(test_indices)/class_total*100:.1f}%)")
    
    # 打印总结信息
    print(f"\n" + "="*50)
    print(f"数据集分割完成！")
    print(f"="*50)
    print(f"总文件数: {total_files}")
    print(f"\n目录结构:")
    print(f"  {train_dir}")
    print(f"    ├── {classes[0]}/")
    print(f"    ├── {classes[1]}/")
    print(f"    └── ...")
    print(f"\n数据分割结果:")
    print(f"  训练集 (train): {total_train} ({total_train/total_files*100:.1f}%)")
    print(f"  验证集 (val):  {total_val} ({total_val/total_files*100:.1f}%)")
    print(f"  测试集 (test): {total_test} ({total_test/total_files*100:.1f}%)")
    print(f"\n各目录统计:")
    
    # 统计各目录的文件数
    for split_name, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        total_in_split = sum(len(os.listdir(os.path.join(split_dir, cls))) 
                            for cls in classes)
        print(f"  {split_name}: {total_in_split} 文件")
        for cls in classes:
            cls_count = len(os.listdir(os.path.join(split_dir, cls)))
            print(f"    ├── {cls}: {cls_count}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(project_root, "data", "processed_data")
    output_data_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(processed_data_dir):
        print(f"错误: 找不到 processed_data 目录: {processed_data_dir}")
    else:
        split_and_organize_dataset(processed_data_dir, output_data_dir)
