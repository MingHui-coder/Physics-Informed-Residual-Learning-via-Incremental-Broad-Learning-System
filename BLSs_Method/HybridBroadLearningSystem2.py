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

def build_mixed_system_matrix_train(
    X: np.ndarray,           # 数据驱动输入 (N, D)
    Q: np.ndarray,           # 物理驱动输入 (N, B)
    n: int,                  # 映射节点组数
    mapping_neurons_per_group: int,   # 每组映射节点神经元数
    m: int,                  # 增强节点组数
    enhance_neurons_per_group: int,   # 每组增强节点神经元数
    output_dim: int,         # 输出维度 C（也用于物理节点 Φ 的列数）
    mapping_activation: str = 'linear',
    enhance_activation: str = 'relu',
    physics_activation: Optional[Callable] = None,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    训练阶段构建混合系统矩阵并保存随机参数。
    
    返回:
        A_tilde: 混合系统矩阵 (N, total_features)
        params: 字典，包含以下键:
            - 'mapping_weights': list of W_mi (每个形状 (D, mapping_neurons_per_group))
            - 'mapping_biases': list of beta_mi (每个形状 (1, mapping_neurons_per_group))
            - 'enhance_weights': list of W_ej (每个形状 (F, enhance_neurons_per_group))
            - 'enhance_biases': list of beta_ej (每个形状 (1, enhance_neurons_per_group))
            - 'phy_W': 物理线性映射权重 (B, output_dim) 或 None (若使用自定义函数)
            - 'phy_b': 物理线性映射偏置 (1, output_dim) 或 None
            - 'phy_activation_func': physics_activation (若为自定义函数，测试时需传入相同函数)
            - 'n', 'mapping_neurons_per_group', 'm', 'enhance_neurons_per_group', 'output_dim'
            - 'mapping_activation', 'enhance_activation'  (用于测试时选择激活函数)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N, D = X.shape
    N_q, B = Q.shape
    assert N == N_q
    assert output_dim > 0
    
    # 激活函数
    phi_act = ACTIVATIONS.get(mapping_activation, default_activation)
    xi_act = ACTIVATIONS.get(enhance_activation, default_activation)
    
    # ---------- 1. 映射特征节点 Z ----------
    mapping_weights = []
    mapping_biases = []
    Z_list = []
    for _ in range(n):
        W_mi = generate_orthogonal_weights(D, mapping_neurons_per_group)
        beta_mi = generate_random_bias(mapping_neurons_per_group)
        linear_out = X @ W_mi + beta_mi
        Z_i = phi_act(linear_out)
        Z_list.append(Z_i)
        mapping_weights.append(W_mi)
        mapping_biases.append(beta_mi)
    Z = np.hstack(Z_list)   # (N, n * mapping_neurons_per_group)
    
    # ---------- 2. 增强特征节点 H ----------
    F = Z.shape[1]
    enhance_weights = []
    enhance_biases = []
    H_list = []
    for _ in range(m):
        W_ej = generate_orthogonal_weights(F, enhance_neurons_per_group)
        beta_ej = generate_random_bias(enhance_neurons_per_group)
        linear_out = Z @ W_ej + beta_ej
        H_j = xi_act(linear_out)
        H_list.append(H_j)
        enhance_weights.append(W_ej)
        enhance_biases.append(beta_ej)
    H = np.hstack(H_list)   # (N, m * enhance_neurons_per_group)
    
    # ---------- 3. 物理特征节点 Φ ----------
    if physics_activation is None:
        # 线性映射
        phy_W = np.random.randn(B, output_dim)
        phy_b = np.random.randn(1, output_dim)
        Phi = Q @ phy_W + phy_b
        phy_activation_func = None
    else:
        # 自定义映射
        Phi = physics_activation(Q)
        if Phi.shape != (N, output_dim):
            raise ValueError(f"physics_activation 必须返回 ({N}, {output_dim}) 形状")
        phy_W = None
        phy_b = None
        phy_activation_func = physics_activation   # 保存函数引用（测试时需传入）
    
    # ---------- 4. 拼接 ----------
    A_tilde = np.hstack([Z, H, Phi])
    
    # 保存参数
    params = {
        'mapping_weights': mapping_weights,
        'mapping_biases': mapping_biases,
        'enhance_weights': enhance_weights,
        'enhance_biases': enhance_biases,
        'phy_W': phy_W,
        'phy_b': phy_b,
        'phy_activation_func': phy_activation_func,
        'n': n,
        'mapping_neurons_per_group': mapping_neurons_per_group,
        'm': m,
        'enhance_neurons_per_group': enhance_neurons_per_group,
        'output_dim': output_dim,
        'mapping_activation': mapping_activation,
        'enhance_activation': enhance_activation
    }
    
    return A_tilde, params

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

    return W, P


def build_mixed_system_matrix_test(
    X_test: np.ndarray,
    Q_test: np.ndarray,
    params: dict,
    physics_activation: Optional[Callable] = None
) -> np.ndarray:
    """
    使用训练阶段保存的参数构建测试集的混合系统矩阵。
    
    参数:
        X_test: (N_test, D)
        Q_test: (N_test, B)
        params: 训练函数返回的参数字典
        physics_activation: 若训练时使用了自定义映射函数，则测试时必须传入相同的函数；
                            若训练时使用了线性映射，则该参数被忽略。
    
    返回:
        A_tilde_test: (N_test, total_features)
    """
    # 读取参数
    mapping_weights = params['mapping_weights']
    mapping_biases = params['mapping_biases']
    enhance_weights = params['enhance_weights']
    enhance_biases = params['enhance_biases']
    phy_W = params['phy_W']
    phy_b = params['phy_b']
    n = params['n']
    mapping_neurons_per_group = params['mapping_neurons_per_group']
    m = params['m']
    enhance_neurons_per_group = params['enhance_neurons_per_group']
    output_dim = params['output_dim']
    mapping_act_name = params['mapping_activation']
    enhance_act_name = params['enhance_activation']
    saved_phy_func = params.get('phy_activation_func')  # 训练时保存的函数（可能为None）
    
    # 激活函数
    phi_act = ACTIVATIONS.get(mapping_act_name, default_activation)
    xi_act = ACTIVATIONS.get(enhance_act_name, default_activation)
    
    N_test, D = X_test.shape
    N_q, B = Q_test.shape
    assert N_test == N_q
    
    # ---------- 1. 映射特征节点 ----------
    Z_list = []
    for i in range(n):
        W_mi = mapping_weights[i]
        beta_mi = mapping_biases[i]
        linear_out = X_test @ W_mi + beta_mi
        Z_i = phi_act(linear_out)
        Z_list.append(Z_i)
    Z = np.hstack(Z_list)
    
    # ---------- 2. 增强特征节点 ----------
    F = Z.shape[1]
    H_list = []
    for j in range(m):
        W_ej = enhance_weights[j]
        beta_ej = enhance_biases[j]
        # 注意：增强节点输入是训练时的 Z，测试时需用当前的 Z 计算
        # 但 W_ej 的形状是 (F_train, enhance_neurons)，
        # 测试时 F_test 必须等于训练时的 F（即 n * mapping_neurons_per_group）
        # 若 X_test 维度 D 与训练一致，则 F 相同，断言确保
        assert Z.shape[1] == F, "测试集映射特征维度与训练时不一致"
        linear_out = Z @ W_ej + beta_ej
        H_j = xi_act(linear_out)
        H_list.append(H_j)
    H = np.hstack(H_list)
    
    # ---------- 3. 物理特征节点 ----------
    # 决策：优先使用传入的 physics_activation，若为 None 则尝试使用训练时保存的函数或线性映射
    if physics_activation is not None:
        Phi = physics_activation(Q_test)
    elif saved_phy_func is not None:
        Phi = saved_phy_func(Q_test)
    elif phy_W is not None and phy_b is not None:
        Phi = Q_test @ phy_W + phy_b
    else:
        raise ValueError("无法确定物理映射：请提供 physics_activation 或确保训练时使用了线性映射")
    
    if Phi.shape != (N_test, output_dim):
        raise ValueError(f"物理映射输出形状错误，应为 ({N_test}, {output_dim})，实际 {Phi.shape}")
    
    # ---------- 4. 拼接 ----------
    A_tilde_test = np.hstack([Z, H, Phi])
    return A_tilde_test


def build_mixed_system_matrix_addition(
    X_addition: np.ndarray,
    Q_addition: np.ndarray,
    params: dict,
    physics_activation: Optional[Callable] = None
) -> np.ndarray:
    """
    用于增量学习场景：使用训练阶段保存的参数构建新增样本的混合系统矩阵。
    该函数直接复用 build_mixed_system_matrix_test 的实现，仅函数名语义不同。
    
    参数:
        X_addition: 新增样本的数据驱动输入，形状 (M, D)
        Q_addition: 新增样本的物理驱动输入，形状 (M, B)
        params: 训练函数返回的参数字典
        physics_activation: 若训练时使用了自定义物理映射，则需传入相同函数
    
    返回:
        A_tilde_addition: 新增样本的混合系统矩阵，形状 (M, total_features)
    """
    return build_mixed_system_matrix_test(
        X_test=X_addition,
        Q_test=Q_addition,
        params=params,
        physics_activation=physics_activation
    )

def train(
    X: np.ndarray,                     # 数据驱动输入 (N, D)
    Q: np.ndarray,                     # 物理驱动输入 (N, B)
    Y: np.ndarray,                     # 目标输出 (N, C)
    n: int,                            # 映射特征节点组数
    mapping_neurons_per_group: int,    # 每组映射节点的神经元数
    m: int,                            # 增强特征节点组数
    enhance_neurons_per_group: int,    # 每组增强节点的神经元数
    output_dim: Optional[int] = None,  # 输出维度 C，若为 None 则自动从 Y 获取
    mapping_activation: str = 'linear',
    enhance_activation: str = 'relu',
    physics_activation: Optional[Callable] = None,
    lambda1: float = 1e-3,             # 数据部分正则化系数
    lambda2: float = 1e-3,             # 物理部分正则化系数
    random_seed: Optional[int] = None
) -> Tuple[dict, np.ndarray]:
    """
    训练混合宽度学习模型。

    参数:
        X, Q, Y: 训练数据
        n, mapping_neurons_per_group: 映射节点配置
        m, enhance_neurons_per_group: 增强节点配置
        output_dim: 输出维度（默认从 Y.shape[1] 获得）
        mapping_activation, enhance_activation: 激活函数类型
        physics_activation: 物理节点映射函数（若为 None 则使用线性变换）
        lambda1, lambda2: 正则化参数
        random_seed: 随机种子

    返回:
        params: 训练时生成的随机参数（可用于测试阶段）
        W: 输出层权重矩阵，形状 (total_features, C)
    """
    if output_dim is None:
        output_dim = Y.shape[1]   # C

    # 1. 构建混合系统矩阵并保存随机参数
    A_tilde, params = build_mixed_system_matrix_train(
        X=X,
        Q=Q,
        n=n,
        mapping_neurons_per_group=mapping_neurons_per_group,
        m=m,
        enhance_neurons_per_group=enhance_neurons_per_group,
        output_dim=output_dim,
        mapping_activation=mapping_activation,
        enhance_activation=enhance_activation,
        physics_activation=physics_activation,
        random_seed=random_seed
    )

    # 2. 计算输出层权重
    W, P = compute_weights(
        A_tilde=A_tilde,
        Y=Y,
        lambda1=lambda1,
        lambda2=lambda2
    )

    return params, W,  P


def test(
    X_test: np.ndarray,
    Q_test: np.ndarray,
    params: dict,
    W: np.ndarray,
    physics_activation: Optional[Callable] = None
) -> np.ndarray:
    """
    使用训练好的模型对测试集进行预测。

    参数:
        X_test: 测试集数据驱动输入，形状 (N_test, D)
        Q_test: 测试集物理驱动输入，形状 (N_test, B)
        params: 训练阶段保存的参数字典（由 train 函数返回）
        W: 训练得到的输出层权重矩阵，形状 (total_features, C)
        physics_activation: 若训练时使用了自定义物理映射，测试时需传入相同函数

    返回:
        Y_pred: 预测输出，形状 (N_test, C)
    """
    # 构建测试集的混合系统矩阵
    A_tilde_test = build_mixed_system_matrix_test(
        X_test=X_test,
        Q_test=Q_test,
        params=params,
        physics_activation=physics_activation
    )

    # 预测：Ŷ = Ã * W
    Y_pred = A_tilde_test @ W

    return Y_pred

def addition_update(
    W: np.ndarray,                     # 原始权重矩阵 (F, C)
    P: np.ndarray,                     # 原始矩阵 (F, F) = (ÃᵀÃ + Λ)⁻¹
    X_addition: np.ndarray,            # 新增数据驱动输入 (M, D)
    Q_addition: np.ndarray,            # 新增物理驱动输入 (M, B)
    Y_addition: np.ndarray,            # 新增输出 (M, C)
    params: dict,                      # 训练阶段保存的参数字典
    physics_activation: Optional[Callable] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据公式进行增量更新：
        P* = P - P Aaᵀ (I + Aa P Aaᵀ)⁻¹ Aa P
        W* = W + P* Aaᵀ (Ya - Aa W)

    参数:
        W: 原始输出权重 (F, C)
        P: 原始矩阵 (F, F)
        X_addition, Q_addition: 新增样本输入
        Y_addition: 新增样本输出
        params: 训练时返回的参数（用于构建 Aa）
        physics_activation: 若物理映射为自定义函数，需传入

    返回:
        P_star: 更新后的矩阵 (F, F)
        W_star: 更新后的权重 (F, C)
    """
    # 构建新增样本的混合系统矩阵 Ã_a
    Aa = build_mixed_system_matrix_addition(
        X_addition=X_addition,
        Q_addition=Q_addition,
        params=params,
        physics_activation=physics_activation
    )   # 形状 (M, F)

    # 计算必要的中间矩阵
    # Aa @ P 形状 (M, F)
    AaP = Aa @ P
    # P @ Aaᵀ 形状 (F, M) = (AaP).T，但后续需要 P Aaᵀ，即 (AaP).T
    # 公式中需要 P Aaᵀ (I + Aa P Aaᵀ)⁻¹ Aa P
    # 先计算 AaP 和 P Aaᵀ = (AaP).T

    # 计算 I + Aa P Aaᵀ
    I = np.eye(Aa.shape[0])   # (M, M)
    AaPAaT = Aa @ P @ Aa.T     # (M, M)
    mat_inv = np.linalg.inv(I + AaPAaT)   # (M, M)

    # 计算 P Aaᵀ (I + Aa P Aaᵀ)⁻¹ Aa P
    # 先求 P Aaᵀ = (AaP).T
    PAaT = AaP.T               # (F, M)
    term = PAaT @ mat_inv @ AaP   # (F, F)

    P_star = P - term

    # 更新权重
    # 计算 Ya - Aa W
    residual = Y_addition - Aa @ W   # (M, C)
    # 计算 P_star @ Aaᵀ @ residual
    # 注意：P_star 是 (F, F)，Aaᵀ 是 (F, M)，residual 是 (M, C)
    W_star = W + (P_star @ Aa.T @ residual)   # (F, C)

    return P_star, W_star
# ================== 示例用法 ==================
