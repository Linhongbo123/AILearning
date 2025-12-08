"""
简化的PyTorch逻辑回归项目
功能：用简单的数据训练一个逻辑回归分类模型
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
# 定义逻辑回归模型
class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入1个特征，输出1个值
        self.sigmoid = nn.Sigmoid()    # Sigmoid激活函数
    
    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

def main():
    print("简单的逻辑回归分类训练")
    print("-" * 35)
    
    # 1. 准备简单的分类数据
    # 规律：x < 5时多数为类别0，x >= 5时多数为类别1
    x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [4.5], 
                          [5.5], [6.0], [7.0], [8.0], [9.0]])
    y_data = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], 
                          [1.0], [1.0], [1.0], [1.0], [1.0]])
    
    print(f"数据点数量: {len(x_data)}")
    print(f"X范围: {x_data.min():.1f} 到 {x_data.max():.1f}")
    print(f"类别0样本数: {(y_data == 0).sum().item()}")
    print(f"类别1样本数: {(y_data == 1).sum().item()}")
    
    # 2. 创建模型
    model = LogisticModel()
    print(f"\n初始参数:")
    print(f"权重: {model.linear.weight.item():.4f}")
    print(f"偏置: {model.linear.bias.item():.4f}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数（Binary Cross Entropy）
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 4. 训练模型
    print("\n开始训练...")
    losses = []
    
    for epoch in range(2000):
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
        if (epoch + 1) % 400 == 0:
            print(f'Epoch {epoch+1}/2000, Loss: {loss.item():.4f}')
    
    # 5. 训练完成后的参数
    print(f"\n训练完成!")
    print(f"最终参数:")
    print(f"权重: {model.linear.weight.item():.4f}")
    print(f"偏置: {model.linear.bias.item():.4f}")
    print(f"最终损失: {losses[-1]:.4f}")
    
    # 6. 分类结果可视化
    plt.figure(figsize=(10, 6))
    x_plot = x_data.numpy().flatten()
    y_plot = y_data.numpy().flatten()
    
    # 绘制原始数据点
    class_0 = x_plot[y_plot == 0]
    class_1 = x_plot[y_plot == 1]
    
    plt.scatter(class_0, [0]*len(class_0), color='red', s=100, label='类别 0', marker='o')
    plt.scatter(class_1, [1]*len(class_1), color='blue', s=100, label='类别 1', marker='s')
    
    # 绘制预测的概率曲线
    with torch.no_grad():
        x_line = torch.linspace(0, 10, 100).reshape(-1, 1)
        y_prob = model(x_line).numpy()
    
    plt.plot(x_line, y_prob, 'g-', label='概率曲线', linewidth=2)
    
    # 绘制决策边界（概率=0.5的点）
    decision_boundary = -model.linear.bias.item() / model.linear.weight.item()
    plt.axvline(x=decision_boundary, color='orange', linestyle='--', 
                linewidth=2, label=f'决策边界 x={decision_boundary:.2f}')
    
    plt.xlabel('X特征值')
    plt.ylabel('类别 / 概率')
    plt.title('逻辑回归分类结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.show()
    
    # 7. 预测示例
    print("\n预测示例:")
    test_values = [2.0, 5.0, 8.0]
    with torch.no_grad():
        for x_val in test_values:
            x_test = torch.tensor([[x_val]])
            prob = model(x_test).item()
            predicted_class = 1 if prob >= 0.5 else 0
            print(f"输入 {x_val:3.1f} -> 概率 {prob:.3f} -> 预测类别 {predicted_class}")

if __name__ == "__main__":
    main()
