"""
简化的决策树分类项目
功能：用简单的数据训练一个决策树分类模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd

# 设置随机种子
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("简单的决策树分类训练")
    print("-" * 30)
    
    # 1. 准备简单的分类数据
    # 规律：年龄和收入决定是否购买产品
    # 年轻且高收入 -> 购买(1)
    # 年老且低收入 -> 不购买(0)
    data = {
        '年龄': [25, 35, 45, 20, 35, 52, 23, 40, 60, 25, 30, 45],
        '收入': [50, 70, 30, 60, 80, 40, 75, 45, 35, 85, 90, 25],
        '购买': [1,  1,  0,  1,  1,  0,  1,  0,  0,  1,  1,  0]
    }
    
    df = pd.DataFrame(data)
    print("数据集:")
    print(df)
    print(f"\n总样本数: {len(df)}")
    print(f"购买样本数: {sum(df['购买'])}")
    print(f"未购买样本数: {len(df) - sum(df['购买'])}")
    
    # 2. 准备训练数据
    X = df[['年龄', '收入']].values  # 特征：年龄和收入
    y = df['购买'].values           # 标签：是否购买
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签向量形状: {y.shape}")
    
    # 3. 创建决策树模型
    # max_depth=3: 限制树的深度，防止过拟合
    # random_state=42: 保证结果可重现
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    
    print(f"\n决策树参数:")
    print(f"最大深度: {model.max_depth}")
    print(f"分裂标准: {model.criterion}")
    
    # 4. 训练模型
    print("\n开始训练...")
    model.fit(X, y)
    print("训练完成!")
    
    # 5. 模型预测
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"\n模型性能:")
    print(f"训练准确率: {accuracy:.3f}")
    
    # 6. 查看决策规则
    print(f"\n决策树信息:")
    print(f"树的深度: {model.get_depth()}")
    print(f"叶子节点数: {model.get_n_leaves()}")
    
    # 7. 可视化决策树
    plt.figure(figsize=(15, 8))
    plot_tree(model, 
             feature_names=['年龄', '收入'], 
             class_names=['不购买', '购买'],
             filled=True, 
             rounded=True, 
             fontsize=10)
    plt.title('决策树结构')
    plt.show()
    
    # 8. 数据分布可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制不同类别的数据点
    buy_data = df[df['购买'] == 1]
    no_buy_data = df[df['购买'] == 0]
    
    plt.scatter(buy_data['年龄'], buy_data['收入'], 
               color='green', s=100, label='购买', marker='o')
    plt.scatter(no_buy_data['年龄'], no_buy_data['收入'], 
               color='red', s=100, label='不购买', marker='x')
    
    plt.xlabel('年龄')
    plt.ylabel('收入 (千元)')
    plt.title('用户购买行为数据分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加决策边界的近似展示
    # 这是一个简化的边界展示，实际决策树的边界是矩形分割
    plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='收入分界线')
    plt.axvline(x=40, color='purple', linestyle='--', alpha=0.7, label='年龄分界线')
    
    plt.legend()
    plt.show()
    
    # 9. 预测新样本
    print("\n预测新样本:")
    new_samples = [
        [28, 65],  # 年轻高收入
        [50, 30],  # 年老低收入
        [35, 75],  # 中年高收入
    ]
    
    for i, sample in enumerate(new_samples):
        age, income = sample
        prediction = model.predict([[age, income]])[0]
        probability = model.predict_proba([[age, income]])[0]
        
        result = "购买" if prediction == 1 else "不购买"
        print(f"年龄 {age}, 收入 {income} -> 预测: {result}")
        print(f"  不购买概率: {probability[0]:.3f}")
        print(f"  购买概率: {probability[1]:.3f}")
        print()
    
    # 10. 特征重要性
    feature_importance = model.feature_importances_
    print("特征重要性:")
    print(f"年龄重要性: {feature_importance[0]:.3f}")
    print(f"收入重要性: {feature_importance[1]:.3f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(8, 5))
    features = ['年龄', '收入']
    plt.bar(features, feature_importance, color=['skyblue', 'lightcoral'])
    plt.title('特征重要性')
    plt.ylabel('重要性分数')
    plt.show()

if __name__ == "__main__":
    main()
