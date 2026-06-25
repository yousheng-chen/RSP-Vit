import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import shutil


def process_dataset(original_data_dir, processed_data_dir, category, num_augmentations=2):
    """
    处理单个类别的数据集，使用亮度和对比度调整来增大数据集。
    
    Args:
        original_data_dir: 原始数据目录路径
        processed_data_dir: 处理后数据的保存目录
        category: 数据集类别名称
        num_augmentations: 每张图像的增强次数（默认2次）
    """
    
    # 定义原始数据目录和目标保存目录
    category_original_path = os.path.join(original_data_dir, category)
    category_processed_path = os.path.join(processed_data_dir, category)
    
    # 检查原始目录是否存在
    if not os.path.exists(category_original_path):
        print(f"警告: {category} 原始数据目录不存在: {category_original_path}")
        return False
    
    # 创建目标目录
    os.makedirs(category_processed_path, exist_ok=True)
    
    # 获取所有原始图像文件
    image_files = sorted([f for f in os.listdir(category_original_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"警告: {category} 原始数据目录中没有图像文件")
        return False
    
    print(f"\n{'='*60}")
    print(f"处理 {category} 数据集 - 找到 {len(image_files)} 张原始图像")
    print(f"{'='*60}")
    
    image_count = 0
    
    # 处理每张图像
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(category_original_path, image_file)
        
        try:
            # 打开原始图像
            original_image = Image.open(image_path)
            
            # 转换为RGB格式（处理RGBA、灰度等格式）
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # 保存原始图像
            original_save_path = os.path.join(category_processed_path, f"original_{idx}.jpg")
            original_image.save(original_save_path, 'JPEG', quality=95)
            image_count += 1
            
            if idx <= 3 or idx == len(image_files):  # 只打印前3个和最后1个
                print(f"[{idx}/{len(image_files)}] 保存原始图像: original_{idx}.jpg")
            elif idx == 4:
                print(f"... (处理中...) ...")
            
            # 生成增强版本
            for aug_idx in range(num_augmentations):
                # 随机生成亮度倍数 (0.9 ~ 1.1)
                brightness_factor = random.uniform(0.9, 1.1)
                
                # 随机生成对比度倍数 (0.85 ~ 1.15)
                contrast_factor = random.uniform(0.85, 1.15)
                
                # 应用亮度调整
                enhanced_image = ImageEnhance.Brightness(original_image).enhance(brightness_factor)
                
                # 应用对比度调整
                enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(contrast_factor)
                
                # 保存增强后的图像
                augmented_save_path = os.path.join(
                    category_processed_path, 
                    f"aug_{idx}_{aug_idx+1}_b{brightness_factor:.2f}_c{contrast_factor:.2f}.jpg"
                )
                enhanced_image.save(augmented_save_path, 'JPEG', quality=95)
                image_count += 1
        
        except Exception as e:
            print(f"错误: 处理 {image_file} 失败 - {e}")
    
    print(f"\n{category} 处理完成！")
    print(f"  原始图像数量: {len(image_files)}")
    print(f"  增强系数: {num_augmentations}x")
    print(f"  处理后总图像数量: {image_count}")
    print(f"  数据集增长倍数: {image_count / len(image_files):.1f}x")
    
    return True


if __name__ == "__main__":
    # 项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 原始数据目录
    original_data_dir = os.path.join(project_root, 'data', 'original_data')
    
    # 处理后数据目录
    processed_data_dir = os.path.join(project_root, 'data', 'processed_data')
    
    # 其他5个数据集类别（Fall已经处理过了）
    categories = ['Jog', 'Marking time', 'Shadow boxing', 'Walk', 'Wava']
    
    print("=" * 70)
    print("开始处理其他数据集类别...")
    print("=" * 70)
    print(f"原始数据目录: {original_data_dir}")
    print(f"处理后数据目录: {processed_data_dir}")
    print(f"处理类别: {', '.join(categories)}")
    print("=" * 70)
    
    # 统计信息
    successful_categories = 0
    
    # 处理每个类别
    for category in categories:
        if process_dataset(original_data_dir, processed_data_dir, category, num_augmentations=2):
            successful_categories += 1
    
    print("\n" + "=" * 70)
    print(f"处理完成！成功处理 {successful_categories}/{len(categories)} 个类别")
    print("=" * 70)
