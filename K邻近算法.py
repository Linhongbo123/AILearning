"""
简化的K邻近算法分类项目
功能：用简单的数据训练一个KNN分类模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# 设置随机种子和中文显示
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("简单的K邻近算法分类训练")
    print("-" * 30)
    
    # 1. 准备简单的分类数据
    # 场景：根据身高和体重预测运动类型
    data = {
        '身高': [160, 162, 165, 168, 170, 172, 175, 178, 180, 182, 185, 188],
        '体重': [50, 52, 55, 60, 65, 68, 70, 75, 78, 80, 85, 90],
        '运动类型': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]  # 0:游泳, 1:跑步, 2:篮球
    }
    
    df = pd.DataFrame(data)
    print("数据集:")
    print(df)
    
    sport_names = {0: '游泳', 1: '跑步', 2: '篮球'}
    print(f"\n数据说明:")
    for sport_id, sport_name in sport_names.items():
        count = sum(df['运动类型'] == sport_id)
        print(f"运动类型 {sport_id}({sport_name}): {count}个样本")
    
    # 2. 准备训练数据
    X = df[['身高', '体重']].values  # 特征矩阵
    y = df['运动类型'].values        # 标签向量
    
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"标签向量形状: {y.shape}")
    
    # 3. 创建KNN模型
    k = 3  # 选择最近的3个邻居
    model = KNeighborsClassifier(n_neighbors=k)
    
    print(f"\nK邻近算法参数:")
    print(f"邻居数量K: {k}")
    print(f"距离度量: 欧几里得距离")
    
    # 4. 训练模型（KNN实际上是存储数据，没有真正的"训练"）
    print(f"\n开始训练...")
    model.fit(X, y)
    print("训练完成!")
    print("注意：KNN是懒惰学习算法，只是存储了训练数据")
    
    # 5. 模型预测和评估
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"\n模型性能:")
    print(f"训练准确率: {accuracy:.3f}")
    
    # 6. 测试新数据点
    test_cases = [
        [175, 72],   # 测试点1
        [165, 58],   # 测试点2  
        [185, 88]    # 测试点3
    ]
    
    print(f"\n预测新样本:")
    for i, test_point in enumerate(test_cases):
        # 预测
        pred = model.predict([test_point])[0]
        proba = model.predict_proba([test_point])[0]
        
        # 找最近的K个邻居
        distances, indices = model.kneighbors([test_point])
        
        print(f"\n测试点{i+1}: 身高{test_point[0]}cm, 体重{test_point[1]}kg")
        print(f"预测运动类型: {pred} ({sport_names[pred]})")
        print(f"预测概率: {proba}")
        print("最近的3个邻居:")
        for j, idx in enumerate(indices[0]):
            neighbor_data = df.iloc[idx]
            dist = distances[0][j]
            print(f"  邻居{j+1}: 身高{neighbor_data['身高']}, 体重{neighbor_data['体重']}, "
                  f"运动{sport_names[neighbor_data['运动类型']]}, 距离{dist:.2f}")
    
    # 7. 数据分布可视化
    plt.figure(figsize=(10, 6))
    
    # 创建颜色映射
    colors = ['red', 'blue', 'green']
    sport_labels = ['游泳', '跑步', '篮球']
    
    # 绘制训练数据
    for sport_id in [0, 1, 2]:
        mask = y == sport_id
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[sport_id], s=100, 
                   label=f'{sport_labels[sport_id]} (类别{sport_id})',
                   alpha=0.7, edgecolors='black')
    
    # 绘制测试点
    test_X = np.array(test_cases)
    test_pred = model.predict(test_X)
    plt.scatter(test_X[:, 0], test_X[:, 1], 
               c=[colors[p] for p in test_pred], 
               s=200, marker='*', 
               label='测试点', 
               edgecolors='black', linewidth=2)
    
    plt.xlabel('身高 (cm)')
    plt.ylabel('体重 (kg)')
    plt.title(f'K邻近算法分类结果 (K={k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 8. 距离计算演示
    print(f"\n距离计算演示（以测试点1为例）:")
    test_point = test_cases[0]
    print(f"测试点: 身高{test_point[0]}cm, 体重{test_point[1]}kg")
    print("到各训练点的距离:")
    
    distances_manual = []
    for i, (height, weight) in enumerate(zip(df['身高'], df['体重'])):
        # 欧几里得距离
        dist = np.sqrt((test_point[0] - height)**2 + (test_point[1] - weight)**2)
        distances_manual.append((dist, i, sport_names[df['运动类型'].iloc[i]]))
        print(f"  到点{i+1}: √[(175-{height})² + (72-{weight})²] = {dist:.2f} ({sport_names[df['运动类型'].iloc[i]]})")
    
    # 找最近的K个
    distances_manual.sort()
    print(f"\n最近的{k}个邻居:")
    for i in range(k):
        dist, idx, sport = distances_manual[i]
        print(f"  第{i+1}近: 距离{dist:.2f}, 运动类型{sport}")

if __name__ == "__main__":
    main()