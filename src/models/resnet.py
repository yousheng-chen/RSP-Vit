"""
Deep Residual Learning for Image Recognition (ResNet)
"""

import torch
import torch.nn as nn
from typing import List, Optional

class ShortcutProjection(nn.Module):
    """
    ## Linear projections for shortcut connection
    ## 直线投影用于捷径连接,用于将输入的尺寸匹配到输出的尺寸,以便我们可以将它们相加.
    This does the $W_s x$ projection described above.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$       输入通道数量
        * `out_channels` is the number of channels in $\mathcal{F}(x, \{W_i\})$       输出通道数量
        We do the same stride on the shortcut connection, to match the feature-map size.
        我们做同样的步长在捷径连接上,以匹配特征图大小
        """
        super().__init__()

        # Convolution layer for linear projection $W_s x$
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)  # 卷积核大小为1，步长为stride
        # Paper suggests adding batch normalization after each convolution operation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        # Convolution and batch normalization
        return self.bn(self.conv(x))
    
     
class ResidualBlock(nn.Module):
    """_summary_
    ## Residual Block
    
    The first convolution layer maps from `in_channels` to `out_channels`,
    where the `out_channels` is higher than `in_channels` when we reduce the
    feature map size with a stride length greater than 1.
    
    The second convolution layer maps from `out_channels` to `out_channels` and
    always has a stride length of 1.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # First $3 \times 3$ convolution layer, this maps to `out_channels` 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()
        
        # Second $3 \times 3$ convolution layer, image size is preserved with stride 1 and padding 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection should be a projection if the stride length is not $1$
        # or if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$, mapping from `in_channels` to `out_channels` or mapping input image size to output image size
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()
        
        # Second activation function (ReLU) after the addition of the shortcut connection
        self.act2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection output
        shortcut = self.shortcut(x)
        # First convolution, batch normalization, and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution and batch normalization
        x = self.bn2(self.conv2(x))
        # Activation function after adding the shortcut connection
        return self.act2(x + shortcut)
    
    
class BottleneckResidualBlock(nn.Module):
    """
    ## Bottleneck Residual Block 
    
    This implements the bottleneck block described in the paper.
    It has three convolution layers, kenel size is $1 \times 1$, $3 \times 3$, and $1 \times 1$ respectively.
    *` The first convolution layer maps from `in_channels` to `out_channels // 4`, kenel size is $1 \times 1$,
    *` The second convolution layer maps from `out_channels // 4` to `out_channels // 4` and kernel size is $3 \times 3$,
       This can have a stride length greater than $1$ when we want to compress the
    *` The third convolution layer maps from `out_channels // 4` to `out_channels` and kernel size is $1 \times 1$.
    
    `bottleneck_channels` is less than `in_channels` and the $3 \times 3$ convolution is performed
    on this shrunk space (hence the bottleneck). The two $1 \times 1$ convolution decreases and increases
    the number of channels.
    
    """
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # First $1 \times 1$ convolution layer, this maps to `bottleneck_channels`
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()
        
        # Second $3 \times 3$ convolution layer, image size is preserved with stride 1 and padding 1
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # Second activation function (ReLU)
        self.act2 = nn.ReLU()
        
        # Third $1 \times 1$ convolution layer, this maps to `out_channels`
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        # Batch normalization after the third convolution
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection should be a projection if the stride length is not $1$
        # or if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$, mapping from `in_channels` to `out_channels` or mapping input image size to output image size
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()
            
        # Third activation function (ReLU) after the addition of the shortcut connection
        self.act3 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection output
        shortcut = self.shortcut(x)
        # First convolution, batch normalization, and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution, batch normalization, and activation
        x = self.act2(self.bn2(self.conv2(x)))
        # Third convolution and batch normalization
        x = self.bn3(self.conv3(x))
        # Activation function after adding the shortcut connection
        return self.act3(x + shortcut)
    
    
class ResNet50(nn.Module):
    def __init__(self, n_blocks: List[int], 
                 n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, 
                 first_kernel_size: int = 7):
        """
        * `n_blocks` is a list of 4 integers, where each integer represents the number of residual blocks in each of the 4 layers. 
            4层，每层的残差块数量. ResNet-50 has [3, 4, 6, 3] blocks in the 4 layers respectively.
        * `n_channels` is a list of 4 integers, where each integer represents the number of channels in each of the 4 layers. 
            4层，每层的通道数, ResNet-50 has [64, 128, 256, 512] channels in the 4 layers respectively.
        * `bottlenecks` is an optional list of 4 integers, where each integer represents the number of channels in the bottleneck convolution for each of the 4 layers. 
            4层，每层的瓶颈卷积通道数, ResNet-50 has [64, 128, 256, 512] channels in the bottleneck convolution for the 4 layers respectively. 
            If `bottlenecks` is `None`, then the model will use regular residual blocks instead of bottleneck blocks. 如果bottlenecks为None，则模型将使用常规残差块而不是瓶颈块。
        * `img_channels` is the number of channels in the input image, default is `3` for RGB images. 
            输入图像的通道数，默认为3（RGB图像）
        * `first_kernel_size` is the kernel size for the first convolution layer, default is `7` as suggested in the paper. 第一卷积层的卷积核大小，默认为7，如论文所建议的那样
        """
        super().__init__()
        # First convolution layer, maps $3 * 3$ to `n_channels[0]`
        # 扩大感受野，减少特征图的尺寸，同时增加通道数到n_channels[0]
        # The paper suggests using a $7 \times 7$ convolution with stride $2$ and padding $3$ for the first layer. 论文建议使用一个7x7的卷积，步长为2，填充为3, 这样可以将输入图像的尺寸缩小一半，同时增加通道数到n_channels[0]
        self.conv1 = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=3)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        # First activation function (ReLU)
        self.act1 = nn.ReLU()
        # MaxPool 3×3, stride=2, padding=1, 进一步缩小特征图的尺寸 112x112 -> 56x56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # channels=n_channels[0](通常为64),56*56
        
        # Residual layers
        self.layer1 = self._make_layer(n_channels[0], n_blocks[0], stride=1, bottleneck_channels=bottlenecks[0] if bottlenecks else None)
        self.layer2 = self._make_layer(n_channels[1], n_blocks[1], stride=2, bottleneck_channels=bottlenecks[1] if bottlenecks else None)
        self.layer3 = self._make_layer(n_channels[2], n_blocks[2], stride=2, bottleneck_channels=bottlenecks[2] if bottlenecks else None)
        self.layer4 = self._make_layer(n_channels[3], n_blocks[3], stride=2, bottleneck_channels=bottlenecks[3] if bottlenecks else None)
        
        # Global average pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(bottlenecks[-1] * 4, 1000)  # Assuming 1000 classes for ImageNet, bottlenecks[-1] * 4 is the number of channels in the last layer's output
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, n_blocks: int, stride: int=2, bottleneck_channels: Optional[int] = None) -> nn.Sequential:   
        """
        *`in_channels` is the number of channels in the input to the layer. 输入到层的通道数
        * `n_blocks` is the number of residual blocks in the layer. 层中的残差块数量
        * `stride` 中间层的步长，通常为2，表示特征图尺寸减半
        * `bottleneck_channels` is the number of channels in the bottleneck convolution for the layer. 层中瓶颈卷积的通道数，如果为None，则使用常规残差块,2*inchannels
        """
        blocks = []
        current_in = in_channels
        for i in range(n_blocks):
            if i == 0:
                current_stride = stride
            else:
                current_stride = 1
            if bottleneck_channels is not None:
                out_ch = bottleneck_channels * 4
                blocks.append(BottleneckResidualBlock(
                    in_channels=current_in,
                    bottleneck_channels=bottleneck_channels,
                    out_channels=out_ch,
                    stride=current_stride
                ))
                current_in = out_ch
            else:
                # For regular residual blocks
                out_ch = current_in  # Assume out_channels = in_channels for regular
                blocks.append(ResidualBlock(
                    in_channels=current_in,
                    out_channels=out_ch,
                    stride=current_stride
                ))
                current_in = out_ch
                
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    
    def forward(self, x: torch.Tensor):
        # First convolution, batch normalization, activation and max pooling
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pooling
        x = self.avgpool(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch size
        # Fully connected layer for classification
        return self.fc(x)
    
    
class ResNet34(nn.Module):
    def __init__(self, img_channels: int = 3, 
                 first_kernel_size: int = 7,
                 n_classes: int = 1000):
        """
        ResNet-34 model with regular residual blocks.
        
        * `img_channels` is the number of channels in the input image, default is `3` for RGB images.
        * `first_kernel_size` is the kernel size for the first convolution layer, default is `7`.
        """
        super().__init__()
        # Configuration for ResNet-34
        n_blocks = [3, 4, 6, 3]
        n_channels = [64, 128, 256, 512]
        
        # First convolution layer
        # size cut in half, channels to 64
        self.conv1 = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.act1 = nn.ReLU()
        # MaxPool 3×3, stride=2, padding=1, 进一步缩小特征图的尺寸 112x112 -> 56x56(假设输入图像为224x224)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(n_channels[0], n_channels[0], n_blocks[0], stride=1)
        self.layer2 = self._make_layer(n_channels[0], n_channels[1], n_blocks[1], stride=2)
        self.layer3 = self._make_layer(n_channels[1], n_channels[2], n_blocks[2], stride=2)
        self.layer4 = self._make_layer(n_channels[2], n_channels[3], n_blocks[3], stride=2)
        
        # Global average pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[-1], n_classes)  # n_classes classes for ImageNet
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        """
        ## Create a layer with n_blocks residual blocks.
        
        * `in_channels` is the number of input channels.
        * `out_channels` is the number of output channels.
        * `n_blocks` is the number of residual blocks.
        * `stride` is the stride for the first block.
        """
        blocks = []
        current_in = in_channels
        for i in range(n_blocks):
            # only the first block in the layer can have a stride greater than 1, 
            # which reduces the feature map size. The rest of the blocks will have stride 1 to preserve the feature map size.
            current_stride = stride if i == 0 else 1
            current_out = out_channels
            blocks.append(ResidualBlock(
                in_channels=current_in,
                out_channels=current_out,
                stride=current_stride
            ))
            current_in = current_out
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        # First convolution, batch normalization, activation and max pooling
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pooling
        x = self.avgpool(x)
        # Flatten the output for the fully connected layer, except batch size
        x = torch.flatten(x, 1)
        # Fully connected layer for classification
        return self.fc(x)
    

class ResNet18(nn.Module):   
    """
    ResNet-18 model with regular residual blocks.
    """
    def __init__(self, img_channels: int = 3,
                 first_kernel_size: int = 7,
                 n_classes: int = 1000):
        super().__init__()
        
        n_blocks = [2,2,2,2]
        n_channels = [64, 128, 256, 512]
        
        self.conv1 = nn.Conv2d(img_channels, n_channels[0], kernel_size=first_kernel_size, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(n_channels[0])
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(n_channels[0], n_channels[0], n_blocks[0], stride=1)
        self.layer2 = self._make_layer(n_channels[0], n_channels[1], n_blocks[1], stride=2)
        self.layer3 = self._make_layer(n_channels[1], n_channels[2], n_blocks[2], stride=2)
        self.layer4 = self._make_layer(n_channels[2], n_channels[3], n_blocks[3], stride=2)

        # Global average pooling and fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[-1], n_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        
        blocks = []
        current_in = in_channels
        for i in range(n_blocks):
            # only the first block in the layer can have a stride greater than 1, 
            # which reduces the feature map size. The rest of the blocks will have stride 1 to preserve the feature map size.            
            current_stride = stride if i == 0 else 1
            blocks.append(ResidualBlock(
                in_channels = current_in,
                out_channels = out_channels,
                stride = current_stride
            ))
            current_in = out_channels
            
        return nn.Sequential(*blocks)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor):
        # First convolution, batch normalization, activation and max pooling
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        # Residual layers        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Global average pooling
        x = self.avgpool(x)
        # Flatten the output for the fully connected layer, except batch size
        x = torch.flatten(x, 1)
        # Fully connected layer for classification
        return self.fc(x)
    
    def info(self):
        """Return model information summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info_str = f"ResNet18 Model Summary:\n"
        info_str += f"Input channels: {self.conv1.in_channels}\n"
        info_str += f"First kernel size: {self.conv1.kernel_size[0]}\n"
        info_str += f"Number of classes: {self.fc.out_features}\n"
        info_str += f"Total parameters: {total_params:,}\n"
        info_str += f"Trainable parameters: {trainable_params:,}\n"
        
        return info_str
    