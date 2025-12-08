"""
简化的支持向量机分类项目
功能：用简单的数据训练一个SVM分类模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# 设置随机种子
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("简单的支持向量机分类训练")
    print("-" * 35)
    
    # 1. 准备简单的二分类数据
    # 两个特征：x1和x2，目标是将两类数据分开
    data = {
        'x1': [1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9],
        'x2': [1, 2, 3, 2, 4, 3, 5, 4, 6, 5, 6, 7],
        '类别': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    
    df = pd.DataFrame(data)
    print("数据集:")
    print(df)
    print(f"\n总样本数: {len(df)}")
    print(f"类别0样本数: {sum(df['类别'] == 0)}")
    print(f"类别1样本数: {sum(df['类别'] == 1)}")
    
    # 2. 准备训练数据
    X = df[['x1', 'x2']].values  # 特征矩阵
    y = df['类别'].values        # 标签向量
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签向量形状: {y.shape}")
    
    # 3. 创建SVM模型
    # kernel='linear': 线性核函数（寻找直线分界）
    # C=1.0: 正则化参数，控制分类严格程度
    model = SVC(kernel='linear', C=1.0, random_state=42)
    
    print(f"\n支持向量机参数:")
    print(f"核函数: {model.kernel}")
    print(f"正则化参数C: {model.C}")
    
    # 4. 训练模型
    print("\n开始训练...")
    model.fit(X, y)
    print("训练完成!")
    
    # 5. 模型预测和评估
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"\n模型性能:")
    print(f"训练准确率: {accuracy:.3f}")
    print(f"支持向量数量: {model.n_support_}")
    print(f"支持向量索引: {model.support_}")
    
    # 6. 数据分布和决策边界可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    
    plt.scatter(class_0[:, 0], class_0[:, 1], 
               color='red', s=100, label='类别 0', marker='o')
    plt.scatter(class_1[:, 0], class_1[:, 1], 
               color='blue', s=100, label='类别 1', marker='s')
    
    # 高亮显示支持向量
    support_vectors = X[model.support_]
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
               s=200, facecolors='none', edgecolors='black', 
               linewidth=2, label='支持向量')
    
    # 绘制决策边界
    # 创建网格点
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                          np.linspace(x2_min, x2_max, 100))
    
    # 预测网格点的类别
    grid_points = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx1.shape)
    
    # 绘制决策边界
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='green', 
               linestyles='--', linewidths=2, label='决策边界')
    
    plt.xlabel('特征 x1')
    plt.ylabel('特征 x2')
    plt.title('支持向量机分类结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 7. 显示决策函数的数学表达式
    # 对于线性SVM: f(x) = w1*x1 + w2*x2 + b
    if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        w1, w2 = model.coef_[0]
        b = model.intercept_[0]
        
        print(f"\n决策函数:")
        print(f"f(x) = {w1:.3f}*x1 + {w2:.3f}*x2 + {b:.3f}")
        print(f"决策规则: f(x) > 0 → 类别1, f(x) < 0 → 类别0")
    
    # 8. 预测新样本
    print("\n预测新样本:")
    new_samples = [
        [2, 2],   # 应该是类别0
        [7, 6],   # 应该是类别1
        [4, 4],   # 边界附近
    ]
    
    for sample in new_samples:
        x1, x2 = sample
        prediction = model.predict([[x1, x2]])[0]
        
        # 计算到决策边界的距离
        distance = model.decision_function([[x1, x2]])[0]
        
        print(f"点 ({x1}, {x2}) -> 预测: 类别{prediction}")
        print(f"  到决策边界距离: {distance:.3f}")
        print(f"  {'越远离边界越确信' if abs(distance) > 1 else '接近边界，不太确信'}")
        print()
    
    # 9. 支持向量详细信息
    print("支持向量详细信息:")
    print("-" * 25)
    for i, sv_idx in enumerate(model.support_):
        sv = X[sv_idx]
        sv_label = y[sv_idx]
        print(f"支持向量 {i+1}: ({sv[0]:.1f}, {sv[1]:.1f}), 类别: {sv_label}")
    
    # 10. 可视化支持向量的重要性
    plt.figure(figsize=(8, 6))
    
    # 绘制所有点，支持向量用特殊标记
    for i in range(len(X)):
        if i in model.support_:
            plt.scatter(X[i, 0], X[i, 1], s=200, 
                       c='gold' if y[i] == 0 else 'orange',
                       marker='*', edgecolors='black', linewidth=1,
                       label='支持向量' if i == model.support_[0] else '')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=80,
                       c='lightcoral' if y[i] == 0 else 'lightblue',
                       alpha=0.6)
    
    plt.xlabel('特征 x1')
    plt.ylabel('特征 x2')
    plt.title('支持向量的重要性')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加文本说明
    plt.text(0.5, 7.5, '★ 支持向量决定分界线', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    plt.text(0.5, 7, '其他点不影响决策边界', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.show()

if __name__ == "__main__":
    main()
