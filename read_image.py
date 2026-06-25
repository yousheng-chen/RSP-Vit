import os
from PIL import Image
import numpy as np

# 定义数据路径
data_path = r'd:\Code\python\Micro-Doppler-vit\micro_doppler_ai\data\original_data\Fall\1.jpg'

# 使用PIL读取图像
img = Image.open(data_path)

# 获取图像shape - (宽, 高)
print(f"图像大小（宽，高）: {img.size}")

# 转换为numpy数组查看详细shape
img_array = np.array(img)
print(f"numpy数组shape: {img_array.shape}")
print(f"图像模式: {img.mode}")
print(f"数据类型: {img_array.dtype}")

# 显示图像基本信息
print(f"\n图像详细信息:")
print(f"  宽度: {img.width} px")
print(f"  高度: {img.height} px")
print(f"  通道数: {img_array.shape[2] if len(img_array.shape) > 2 else 1}")
