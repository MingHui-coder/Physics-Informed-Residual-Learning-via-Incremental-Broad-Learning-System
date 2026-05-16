"""
@File    :   BLS_Regressor.py
@Time    :   2026/04/22
@Author  :   Minghui Zhang
@Desc    :   使用BLSRegressor进行回归任务的示例
"""
import numpy as np
from BoradLearningSystem import BLSRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ========== 1. 加载sklearn回归数据集 ==========
# 使用sklearn内置的糖尿病数据集（442个样本，10个特征）
dataset = load_diabetes()
data, target = dataset['data'], dataset['target']

print(f"数据集大小: {data.shape}")
print(f"特征维度: {data.shape[1]}")
print(f"目标变量范围: [{target.min():.2f}, {target.max():.2f}]")

# ========== 2. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# ========== 3. 数据标准化 ==========
# BLS模型对输入特征的尺度敏感，标准化有助于提升性能
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 创建BLSRegressor模型 ==========
# 参数说明:
#   NumFeatureNodes: 每个特征窗口的节点数
#   NumWindows:      特征窗口数量
#   NumEnhance:      增强层节点数
#   S:               收缩系数
#   C:               正则化系数（控制模型复杂度）
regressor = BLSRegressor(
    NumFeatureNodes=10,
    NumWindows=10,
    NumEnhance=100,
    S=0.8,
    C=2 ** -30
)

# ========== 5. 训练模型 ==========
print("\n开始训练 BLSRegressor ...")
train_output = regressor.fit(X_train_scaled, y_train)
print(f"训练完成，训练输出形状: {train_output.shape}")

# ========== 6. 预测 ==========
y_pred = np.asarray(regressor.predict(X_test_scaled)).ravel()

# ========== 7. 评估 ==========
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== 评估结果 ==========")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R2):   {r2:.4f}")
print(f"真实值范围: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
