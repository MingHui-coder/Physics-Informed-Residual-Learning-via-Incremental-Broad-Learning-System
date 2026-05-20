import numpy as np
from typing import Tuple, List, Callable, Optional

def generate_orthogonal_weights(in_dim: int, out_dim: int) -> np.ndarray:
    """
    生成随机正交权重矩阵（或半正交矩阵）。
    如果 in_dim >= out_dim，则生成列正交矩阵 W^T W = I。
    如果 in_dim < out_dim，则生成行正交矩阵 W W^T = I。
    """
    # 生成随机高斯矩阵
    W = np.random.randn(in_dim, out_dim)
    # 使用 QR 分解获得正交基
    Q, R = np.linalg.qr(W, mode='reduced')
    # 当 out_dim > in_dim 时，Q 的形状为 (in_dim, in_dim)，需要填充或处理
    if out_dim > in_dim:
        # 补充随机正交列
        extra = np.random.randn(in_dim, out_dim - in_dim)
        Q_extra, _ = np.linalg.qr(extra, mode='reduced')
        Q = np.hstack([Q, Q_extra])
    # 确保输出维度正确
    return Q[:, :out_dim]

def generate_random_bias(out_dim: int) -> np.ndarray:
    """生成随机偏置，形状为 (1, out_dim)"""
    return np.random.uniform(-1, 1, (1, out_dim))

def default_activation(x: np.ndarray) -> np.ndarray:
    """默认激活函数：线性（恒等映射）"""
    return x

def relu_activation(x: np.ndarray) -> np.ndarray:
    """ReLU 激活函数"""
    return np.maximum(0, x)

def tanh_activation(x: np.ndarray) -> np.ndarray:
    """tanh 激活函数"""
    return np.tanh(x)

def sigmoid_activation(x: np.ndarray) -> np.ndarray:
    """sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))

# 可选激活函数字典
ACTIVATIONS = {
    'linear': default_activation,
    'relu': relu_activation,
    'tanh': tanh_activation,
    'sigmoid': sigmoid_activation
}
import numpy as np
from typing import Optional


def physics_activation(
    Q: np.ndarray,
    m: float = 1.0,       # 小腿+脚的总质量 [kg]
    L: float = 1.0,       # 小腿+脚的总长度 [m]
    d: Optional[float] = None,  # 质心到膝关节的长度 [m]，若为None则默认取 L/2
    g: float = 9.81       # 重力加速度 [m/s²]
) -> np.ndarray:
    """
    根据单自由度膝关节动力学模型计算基础物理扭矩 Φ = M(q) * q̈ + G(q)
    
    参数:
        Q: 输入数组，形状 (N, 3)，每一行分别为 [q, q̇, q̈]
        m: 质量 (kg)
        L: 总长度 (m)
        d: 质心到膝关节的长度 (m)，默认 L/2
        g: 重力加速度 (m/s²)
    
    返回:
        Φ: 基础扭矩，形状 (N, 1)
    """
    if d is None:
        d = L / 2.0   # 假设质心位于小腿中点
    
    # 提取物理量
    q = Q[:, 0]          # 角度 (rad)
    # q_dot = Q[:, 1]    # 角速度 (rad/s) – 此处离心力项为零，故未使用
    q_ddot = Q[:, 2]     # 角加速度 (rad/s²)
    
    # 惯性矩 I_cm = (1/12) * m * L^2 (绕质心的转动惯量)
    I_cm = (1.0 / 12.0) * m * (L ** 2)
    
    # 等效惯性 M = I_cm + m * d^2
    M_inertia = I_cm + m * (d ** 2)
    
    # 重力项 G = m * g * d * cos(q)
    G_gravity = m * g * d * np.cos(q)
    
    # 基础扭矩 Φ = M * q̈ + G
    Phi = M_inertia * q_ddot + G_gravity   # shape (N,)
    
    # 返回为列向量 (N, 1)
    return Phi.reshape(-1, 1)

def build_mixed_system_matrix(
    X: np.ndarray,           # 数据驱动输入，形状 (N, D)
    Q: np.ndarray,           # 物理驱动输入，形状 (N, B)
    n: int,                  # 映射特征节点组数
    mapping_neurons_per_group: int,   # 每组映射节点的输出维度（即 Z_i 的列数）
    m: int,                  # 增强特征节点组数
    enhance_neurons_per_group: int,   # 每组增强节点的输出维度（即 H_j 的列数）
    output_dim: int,         # C，输出维度（用于物理节点）
    mapping_activation: str = 'linear',   # 映射节点的激活函数类型
    enhance_activation: str = 'relu',     # 增强节点的激活函数类型
    physics_activation: Optional[Callable] = None,  # 物理节点映射 g 的函数，若为 None 则使用线性变换
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    根据公式 (3)-(8) 构建混合系统矩阵 Ã = [Z | H | Φ]。
    
    参数:
        X: 数据驱动输入，形状 (N, D)
        Q: 物理驱动输入，形状 (N, B)
        n: 映射特征节点组数
        mapping_neurons_per_group: 每组映射特征节点的神经元数（即每个 Z_i 的列数）
        m: 增强特征节点组数
        enhance_neurons_per_group: 每组增强特征节点的神经元数（即每个 H_j 的列数）
        output_dim: 输出维度 C，用于物理节点 Φ 的输出维度，也决定了最终 W 的列数
        mapping_activation: 映射节点 φ_i 的激活函数类型，可选 'linear', 'relu', 'tanh', 'sigmoid'
        enhance_activation: 增强节点 ξ_j 的激活函数类型
        physics_activation: 物理节点 g 的映射函数，若为 None 则使用线性变换: Φ = Q @ W_phy + b_phy
        random_seed: 随机种子，确保可重复性
    
    返回:
        Ã: 混合系统矩阵，形状 (N, total_features)
            total_features = n * mapping_neurons_per_group + m * enhance_neurons_per_group + output_dim
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N, D = X.shape
    N_q, B = Q.shape
    assert N == N_q, "X 和 Q 的样本数必须相同"
    assert output_dim > 0, "输出维度必须为正"
    
    # 选择激活函数
    phi_activation = ACTIVATIONS.get(mapping_activation, default_activation)
    xi_activation = ACTIVATIONS.get(enhance_activation, default_activation)
    
    # ================== 1. 构建映射特征节点 Z = [Z_1, Z_2, ..., Z_n] ==================
    Z_list = []
    for i in range(1, n+1):
        # 生成随机正交权重 W_{m_i}，形状 (D, mapping_neurons_per_group)
        W_mi = generate_orthogonal_weights(D, mapping_neurons_per_group)
        # 生成随机偏置 β_{m_i}，形状 (1, mapping_neurons_per_group)
        beta_mi = generate_random_bias(mapping_neurons_per_group)
        # 线性变换: X @ W_mi + beta_mi，然后应用 φ_i
        linear_out = X @ W_mi + beta_mi
        Z_i = phi_activation(linear_out)   # 形状 (N, mapping_neurons_per_group)
        Z_list.append(Z_i)
    # 横向拼接所有 Z_i
    Z = np.hstack(Z_list)   # 形状 (N, n * mapping_neurons_per_group)
    
    # ================== 2. 构建增强特征节点 H = [H_1, H_2, ..., H_m] ==================
    # 输入 Z 的形状 (N, F)，其中 F = n * mapping_neurons_per_group
    F = Z.shape[1]
    H_list = []
    for j in range(1, m+1):
        # 生成随机正交权重 W_{e_j}，形状 (F, enhance_neurons_per_group)
        # 注意：原公式 (5) 中写为 Z^T W_ej，但根据常见宽度学习应为 Z W_ej，此处采用合理实现
        W_ej = generate_orthogonal_weights(F, enhance_neurons_per_group)
        # 生成随机偏置 β_{e_j}，形状 (1, enhance_neurons_per_group)
        beta_ej = generate_random_bias(enhance_neurons_per_group)
        # 线性变换: Z @ W_ej + beta_ej，然后应用 ξ_j
        linear_out = Z @ W_ej + beta_ej
        H_j = xi_activation(linear_out)   # 形状 (N, enhance_neurons_per_group)
        H_list.append(H_j)
    H = np.hstack(H_list)   # 形状 (N, m * enhance_neurons_per_group)
    
    # ================== 3. 构建物理特征节点 Φ = g(Q) ==================
    # 要求 Φ 的形状为 (N, output_dim) 即 (N, C)
    if physics_activation is None:
        # 使用线性变换: Φ = Q @ W_phy + b_phy
        W_phy = np.random.randn(B, output_dim)   # 线性权重，不严格要求正交
        b_phy = np.random.randn(1, output_dim)
        Phi = Q @ W_phy + b_phy
    else:
        # 用户自定义映射，必须确保输出形状为 (N, output_dim)
        Phi = physics_activation(Q)
        if Phi.shape != (N, output_dim):
            raise ValueError(f"physics_activation 必须返回形状 ({N}, {output_dim}) 的张量，实际得到 {Phi.shape}")
    
    # ================== 4. 拼接得到混合系统矩阵 Ã = [Z | H | Φ] ==================
    A_tilde = np.hstack([Z, H, Phi])   # 形状 (N, total_features)
    
    return A_tilde

def compute_weights(
    A_tilde: np.ndarray,   # 混合系统矩阵，形状 (N, F)，其中 F = d + p, p = C
    Y: np.ndarray,         # 目标输出，形状 (N, C)
    lambda1: float = 1e-3, # 数据部分正则化参数
    lambda2: float = 1e-3  # 物理部分正则化参数
) -> np.ndarray:
    """
    根据公式 (28)-(29) 计算权重矩阵 W = (ÃᵀÃ + Λ)⁻¹ (ÃᵀY + B_phy)

    参数:
        A_tilde: 混合系统矩阵，形状 (N, F)
        Y: 目标输出，形状 (N, C)
        lambda1: 数据部分的正则化系数 λ₁
        lambda2: 物理部分的正则化系数 λ₂

    返回:
        W: 权重矩阵，形状 (F, C)
    """
    N, F = A_tilde.shape
    N_y, C = Y.shape
    if N != N_y:
        raise ValueError("A_tilde 和 Y 的样本数必须相同")
    
    # 物理部分的特征数 p 应等于输出维度 C (因为 Φ ∈ R^{N×C})
    p = C
    d = F - p
    if d < 0:
        raise ValueError("A_tilde 的列数必须大于等于 C")

    # 构建广义正则化矩阵 Λ (F × F 对角阵)
    # 前 d 个对角元为 lambda1, 后 p 个对角元为 lambda2
    diag_lambda = np.array([lambda1] * d + [lambda2] * p)
    Lambda = np.diag(diag_lambda)

    # 构建物理偏置矩阵 B_phy (F × C)
    # 前 d 行全为 0，后 p 行为 lambda2 * I_p
    B_phy = np.zeros((F, C))
    # 后 p 行的第 j 列第 j 行为 lambda2
    B_phy[d:, :] = lambda2 * np.eye(p)  # 注: p = C, 因此 (p, p) 可以赋值给 (p, C)

    # 计算 ÃᵀÃ 和 ÃᵀY
    AtA = A_tilde.T @ A_tilde          # (F, F)
    AtY = A_tilde.T @ Y                # (F, C)

    # 计算 P = (ÃᵀÃ + Λ)⁻¹
    # 数值稳定性: 使用 np.linalg.solve 替代求逆再乘法
    # 但为了严格符合公式 (28)，这里显式计算逆矩阵
    # 实际推荐使用 solve: W = np.linalg.solve(AtA + Lambda, AtY + B_phy)
    try:
        P = np.linalg.inv(AtA + Lambda)   # (F, F)
    except np.linalg.LinAlgError:
        # 若矩阵奇异，加入微小扰动（实际已有正则化，理论上可逆）
        P = np.linalg.inv(AtA + Lambda + 1e-12 * np.eye(F))

    # 计算权重 W = P (ÃᵀY + B_phy)
    W = P @ (AtY + B_phy)                 # (F, C)

    return W


# ================== 示例用法 ==================
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    N, D, B, C = 100, 10, 5, 3
    X = np.random.randn(N, D)
    Q = np.random.randn(N, B)
    
    # 参数设置
    n = 3                     # 3 组映射节点
    mapping_neurons = 8       # 每组 8 个神经元
    m = 2                     # 2 组增强节点
    enhance_neurons = 10      # 每组 10 个神经元
    output_dim = C
    
    # 构建混合系统矩阵
    A_tilde = build_mixed_system_matrix(
        X, Q, n, mapping_neurons, m, enhance_neurons, output_dim,
        mapping_activation='linear',
        enhance_activation='relu',
        physics_activation=None,
        random_seed=42
    )
    
    print(f"混合系统矩阵 Ã 的形状: {A_tilde.shape}")
    expected_features = n * mapping_neurons + m * enhance_neurons + output_dim
    print(f"期望的特征维度: {expected_features}")
    print(f"Ã 的前几行:\n{A_tilde[:3, :10]}")  # 打印部分内容