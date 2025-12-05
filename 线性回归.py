"""
简化的PyTorch线性回归项目
功能：用简单的数据训练一个线性回归模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 定义线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入1个特征，输出1个值
    
    def forward(self, x):
        return self.linear(x)

def main():
    print("简单的线性回归训练")
    print("-" * 30)
    
    # 1. 准备简单的模拟数据
    # 真实关系: y = 2.0 * x + 1.0 + 噪声
    x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], 
                          [6.0], [7.0], [8.0], [9.0], [10.0]])
    y_data = torch.tensor([[3.1], [5.2], [7.0], [8.9], [11.1], 
                          [12.8], [15.2], [17.1], [18.8], [21.0]])
    
    print(f"数据点数量: {len(x_data)}")
    print(f"X范围: {x_data.min():.1f} 到 {x_data.max():.1f}")
    print(f"Y范围: {y_data.min():.1f} 到 {y_data.max():.1f}")
    
    # 2. 创建模型
    model = LinearModel()
    print(f"\n初始参数:")
    print(f"权重: {model.linear.weight.item():.4f}")
    print(f"偏置: {model.linear.bias.item():.4f}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 4. 训练模型
    print("\n开始训练...")
    losses = []
    
    for epoch in range(1000):
        # 前向传播
        predictions = model(x_data)
        loss = criterion(predictions, y_data)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        losses.append(loss.item())
        
        # 打印进度
        if (epoch + 1) % 200 == 0:
            print(f'Epoch {epoch+1}/1000, Loss: {loss.item():.4f}')
    
    # 5. 训练完成后的参数
    print(f"\n训练完成!")
    print(f"最终参数:")
    print(f"权重: {model.linear.weight.item():.4f}")
    print(f"偏置: {model.linear.bias.item():.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    
    # 6. 拟合结果可视化
    plt.figure(figsize=(8, 6))
    x_plot = x_data.numpy()
    y_plot = y_data.numpy()
    
    plt.scatter(x_plot, y_plot, color='blue', label='原始数据', s=50)
    
    # 预测直线
    with torch.no_grad():
        x_line = torch.linspace(0, 11, 100).reshape(-1, 1)
        y_line = model(x_line).numpy()
    
    plt.plot(x_line, y_line, 'r-', label='拟合直线', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()