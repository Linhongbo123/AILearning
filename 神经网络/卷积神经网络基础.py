"""
卷积神经网络(CNN)基础示例
功能: 实现一个简单的CNN模型用于图像分类
"""

import torch
import torch.nn as nn


class simplecnn(nn.Module):
    """
    简单的卷积神经网络模型
    包含特征提取器和分类器两部分
    """
    def __init__(self, num_classes):
        """
        初始化CNN模型
        
        参数:
            num_classes (int): 分类的类别数量，例如10表示10个类别
        """
        super().__init__()  # 调用父类nn.Module的初始化方法
        
        # 特征提取器: 使用卷积层和池化层提取图像特征
        self.feature_extractor = nn.Sequential(
            # 第一个卷积层
            # 参数说明:
            #   3: 输入通道数(RGB图像有3个颜色通道)
            #   16: 输出通道数(学习16个不同的特征图)
            #   kernel_size=3: 卷积核大小为3x3
            #   stride=1: 卷积核每次移动1个像素
            #   padding=1: 在图像边缘填充1圈0，保持输出尺寸不变
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            #卷积后大小  N = (W-F+2P)/S +1 = 224-3+2*1/1 +1 = 224
            # ReLU激活函数: 引入非线性，将负值变为0，防止梯度消失
            nn.ReLU(),
            
            # 第一个最大池化层
            # 参数说明:
            #   kernel_size=2: 池化窗口大小为2x2
            #   stride=2: 窗口每次移动2个像素
            # 作用: 将特征图尺寸减半(224x224 -> 112x112)，减少计算量
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二个卷积层
            # 参数说明:
            #   16: 输入通道数(来自上一层的16个特征图)
            #   32: 输出通道数(学习32个更复杂的特征图)
            #   kernel_size=3: 卷积核大小为3x3
            #   stride=1: 卷积核每次移动1个像素
            #   padding=1: 边缘填充，保持输出尺寸
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            
            # ReLU激活函数
            nn.ReLU(),
            
            # 第二个最大池化层
            # 作用: 将特征图尺寸再减半(112x112 -> 56x56)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器: 使用全连接层进行最终分类
        self.classifier = nn.Sequential(
            # 第一个全连接层
            # 参数说明:
            #   32 * 56 * 56: 输入特征数(32个通道，每个56x56的特征图展平后的总长度)
            #   128: 输出特征数(压缩到128维)
            nn.Linear(32 * 56 * 56, 128),
            
            # ReLU激活函数
            nn.ReLU(),
            
            # 第二个全连接层(输出层)
            # 参数说明:
            #   128: 输入特征数
            #   num_classes: 输出特征数(等于类别数)
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播函数: 定义数据如何在网络中流动
        
        参数:
            x (tensor): 输入图像张量，形状为 (batch_size, 3, 224, 224)
                       batch_size: 批次大小
                       3: RGB三个颜色通道
                       224, 224: 图像的高度和宽度
        
        返回:
            tensor: 输出预测结果，形状为 (batch_size, num_classes)
        """
        # 步骤1: 通过特征提取器，提取图像特征
        # 输入: (batch_size, 3, 224, 224)
        # 输出: (batch_size, 32, 56, 56)
        x = self.feature_extractor(x)
        
        # 步骤2: 展平特征图，将多维张量转换为二维
        # x.size(0): 获取batch_size
        # -1: 自动计算剩余维度(32*56*56)
        # 输入: (batch_size, 32, 56, 56)
        # 输出: (batch_size, 32*56*56)
        x = x.view(x.size(0), -1)
        
        # 步骤3: 通过分类器，得到最终预测结果
        # 输入: (batch_size, 32*56*56)
        # 输出: (batch_size, num_classes)
        x = self.classifier(x)
        
        return x


# ==================== 测试代码 ====================

# 定义分类类别数(例如: 10个类别，可以是0-9的数字或10种物体)
num_classes = 10

# 创建CNN模型实例
model = simplecnn(num_classes)

# 创建随机输入数据用于测试
# 参数说明:
#   64: batch_size，一次处理64张图片
#   3: 通道数(RGB)
#   224, 224: 图像尺寸(高度×宽度)
input = torch.randn(64, 3, 224, 224)

# 将输入数据传入模型，得到输出
# 模型会自动调用forward方法
output = model(input)

# 打印输出形状
# 预期输出: torch.Size([64, 10])
# 表示64张图片，每张图片对应10个类别的预测分数
print(output.shape) 