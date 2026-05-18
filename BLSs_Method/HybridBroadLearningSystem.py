"""
@File    :   HybridBroadLearningSystem.py
@Time    :   2026/05/17
@Author  :   Minghui Zhang
@Version :   1.0
@Desc    :   混合宽度学习系统，结合数据驱动的BLS模型与物理驱动特征。
            支持增量学习，基于Woodbury恒等式优化计算。

公式来源：README.md 1.2-1.6节
"""
import numpy as np


class HybridBLS:
    """混合宽度学习系统类

    整合数据驱动特征映射、增强层与物理特征节点，支持增量更新。

    参数:
        NumFeatureNodes: 每组映射特征节点数
        NumWindows: 映射特征窗口数
        NumEnhance: 增强层节点数
        S: 缩放因子
        C: 正则化参数
        lambda1: W_d的正则化参数
        lambda2: W_p的正则化参数
    """

    def __init__(self, NumFeatureNodes=10, NumWindows=100, NumEnhance=1000,
                 S=0.5, C=2**-30, lambda1=2**-30, lambda2=2**-30):
        self.FeatureNodes = NumFeatureNodes
        self.FeatureWindows = NumWindows
        self.EnhancementNodes = NumEnhance
        self.S = S
        self.C = C
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    # ── 激活函数 ─────────────────────────────────────────────────────────
    def _tansig(self, x):
        """双曲正切激活函数: (2/(1+exp(-2x)))-1"""
        return (2 / (1 + np.exp(-2 * x))) - 1

    def _relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)

    # ── 伪逆求解 ─────────────────────────────────────────────────────────
    def _pinv(self, A, reg):
        """正则化伪逆: (reg*I + A^T A)^{-1} A^T"""
        M = reg * np.eye(A.shape[1]) + A.T @ A
        return np.linalg.solve(M, A.T)

    # ── 稀疏BLS ─────────────────────────────────────────────────────────
    def _shrinkage(self, a, b):
        """收缩函数: max(a-b, 0) - max(-a-b, 0)"""
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def _sparse_bls(self, A, b):
        """稀疏BLS求解"""
        lam = 0.001
        itrs = 50
        AA = A.T @ A
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.linalg.inv(AA + np.eye(m))
        L2 = L1 @ A.T @ b
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + L1 @ tempc
            ok = self._shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def compute_physics_node(self, Q, m, g, d):
        """计算物理特征节点 (公式9): Φ = g(Q)

        对于膝关节力矩估计:
        Φ = M(q)*q_acc + C(q,q_dot)*q_dot + G(q)
          = m*d^2*q_acc + m*g*d*cos(q)

        参数:
            Q: 物理驱动输入，shape=(N, 3)，包含 [q, q_dot, q_acc]
            m: 质量
            g: 重力加速度
            d: 质心到关节的距离

        返回:
            Phi: 物理特征节点，shape=(N, 1)
        """
        q = Q[:, 0:1]       # 关节角
        q_dot = Q[:, 1:2]   # 关节角速度
        q_acc = Q[:, 2:3]  # 关节角加速度

        I_cm = (1 / 12) * m * (d ** 2)  # ���动惯量 (公式21)
        M_q = I_cm + m * (d ** 2)       # 惯性矩阵

        C_q_qdot = 0  # 单自由度膝关节离心力项为0 (公式22)

        G_q = m * g * d * np.cos(q)  # 重力项 (公式23)

        Phi = M_q * q_acc + C_q_qdot + G_q
        return Phi

    def _solve_hybrid_weights(self, A_tilde, Y):
        """求解混合系统权重 (公式29, 30)

        公式(29): W = P @ (Ã.T @ Y + B_phy)
        其中 P = (Ã.T @ Ã + Λ)^{-1}

        参数:
            A_tilde: 混合系统矩阵 [Z | H | Φ]
            Y: 目标输出

        返回:
            W: 联合权重
            P: 协方差矩阵（用于增量学习）
        """
        d_total = A_tilde.shape[1]  # 总维度 = d_d + d_p
        d_p = 1  # 物理驱动维度

        # 构建正则化矩阵 Λ (公式18)
        Lambda = np.zeros((d_total, d_total))
        Lambda[:d_total-d_p, :d_total-d_p] = self.lambda1 * np.eye(d_total - d_p)
        Lambda[d_total-d_p:, d_total-d_p:] = self.lambda2 * np.eye(d_p)

        # 构建物理偏执向量 B_phy (公式19)
        B_phy = np.zeros((d_total, Y.shape[1]))
        B_phy[d_total-d_p:, :] = self.lambda2 * np.eye(d_p)

        # 公式(30): P = (Ã.T @ Ã + Λ)^{-1}
        P = np.linalg.inv(A_tilde.T @ A_tilde + Lambda + 1e-12 * np.eye(d_total))

        # 公式(29): W = P @ (Ã.T @ Y + B_phy)
        W = P @ (A_tilde.T @ Y + B_phy)

        return W, P

    def _woodbury_inverse(self, P, A_a):
        """Woodbury恒等式 (公式32)

        P* = P - P @ A_a.T @ (I + A_a @ P @ A_a.T)^{-1} @ A_a @ P

        参数:
            P: 旧协方差矩阵
            A_a: 增量混合系统矩阵

        返回:
            P_new: 新协方差矩阵
        """
        A_p = A_a.shape[0]
        I = np.eye(A_p)

        # 避免奇异矩阵
        inner = I + A_a @ P @ A_a.T + 1e-12 * np.eye(A_p)
        inner_inv = np.linalg.inv(inner)

        P_new = P - P @ A_a.T @ inner_inv @ A_a @ P
        return P_new

    def fit(self, train_x, train_y, train_q, m=None, g=9.81, d=None):
        """训练混合BLS模型

        参数:
            train_x: 数据驱动输入, shape=(N, D)
            train_y: 目标输出, shape=(N, C)
            train_q: 物理驱动输入, shape=(N, 3) [q, q_dot, q_acc]
            m: 质量 (默认使用训练数据估计)
            g: 重力加速度 (默认9.81)
            d: 质心距离 (默认使用训练数据估计)

        返回:
            训练输出
        """
        n_samples = train_x.shape[0]
        np.random.seed(2022)

        # ── 步骤1: 初始化权重 ───────────────────────────────────────
        WF = []
        for i in range(self.FeatureWindows):
            WeightFea = 2 * np.random.randn(train_x.shape[1] + 1, self.FeatureNodes) - 1
            WF.append(WeightFea)

        WeightEnhan = 2 * np.random.randn(
            self.FeatureWindows * self.FeatureNodes + 1, self.EnhancementNodes) - 1
        self.WeightEnhan = WeightEnhan
        self.WF = WF  # 保存用于预测

        # ── 步骤2: 计算特征映射层 Z (公式3-4) ────────────────────────
        H1 = np.hstack([train_x, 0.1 * np.ones((n_samples, 1))])
        y = np.zeros((n_samples, self.FeatureWindows * self.FeatureNodes))

        self.WFSparse = []
        self.distOfMaxAndMin = []
        self.meanOfEachWindow = []

        for i in range(self.FeatureWindows):
            WeightFea = WF[i]
            A1 = H1 @ WeightFea

            # MinMaxScaler映射到(-1,1)
            min_vals = A1.min(axis=0, keepdims=True)
            max_vals = A1.max(axis=0, keepdims=True)
            A1 = 2 * (A1 - min_vals) / (max_vals - min_vals + 1e-12) - 1

            # 稀疏BLS求解
            WeightFeaSparse = self._sparse_bls(A1, H1).T
            self.WFSparse.append(WeightFeaSparse)

            T1 = H1 @ WeightFeaSparse
            mean_i = T1.mean(axis=0, keepdims=True)
            dist_i = T1.max(axis=0, keepdims=True) - T1.min(axis=0, keepdims=True)
            self.meanOfEachWindow.append(mean_i)
            self.distOfMaxAndMin.append(dist_i)
            T1 = (T1 - mean_i) / (dist_i + 1e-12)

            y[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = T1

        # ── 步骤3: 计算增强层 H (公式6-7) ───────────────────────────────
        H2 = np.hstack([y, 0.1 * np.ones((n_samples, 1))])
        T2 = self._tansig(H2 @ self.WeightEnhan)
        H = T2  # 增强层输出

        # ── 步骤4: 计算物理特征节点 Φ (公式9, 20) ──────────────────────
        # 估计物理参数
        if m is None:
            m = 1.0  # 默认质量
        if d is None:
            d = 0.3  # 默认质心距离

        self.m = m
        self.g = g
        self.d = d

        Phi = self.compute_physics_node(train_q, m, g, d)
        self.Phi_train = Phi  # 保存用于后续计算

        # ── 步骤5: 拼接混合系统矩阵 Ã = [Z | H | Φ] (公式10-11) ───────────
        Z = y
        A_tilde = np.hstack([Z, H, Phi])

        # ── 步骤6: 求解联合权重 W (公式29) ────────────────────────────────────
        self.W, self.P = self._solve_hybrid_weights(A_tilde, train_y)

        # ── 步骤7: 计算训练输出 ─────────────────────────────────────────────
        NetoutTrain = A_tilde @ self.W
        self.Y_train = NetoutTrain

        return NetoutTrain

    def predict(self, test_x, test_q):
        """预测

        参数:
            test_x: 数据驱动输入, shape=(M, D)
            test_q: 物理驱动输入, shape=(M, 3)

        返回:
            预测输出
        """
        n_samples = test_x.shape[0]

        # ── 步骤1: 特征映射层 ───────────────────────────────────────
        HH1 = np.hstack([test_x, 0.1 * np.ones((n_samples, 1))])
        yy1 = np.zeros((n_samples, self.FeatureWindows * self.FeatureNodes))

        for i in range(self.FeatureWindows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1 @ WeightFeaSparse
            TT1 = (TT1 - self.meanOfEachWindow[i]) / (self.distOfMaxAndMin[i] + 1e-12)
            yy1[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = TT1

        # ── 步骤2: 增强层 ───────────────────────────────────────────────
        HH2 = np.hstack([yy1, 0.1 * np.ones((n_samples, 1))])
        TT2 = self._tansig(HH2 @ self.WeightEnhan)

        # ── 步骤3: 物理节点 ───────────────────────────────────────────────
        Phi_test = self.compute_physics_node(test_q, self.m, self.g, self.d)

        # ── 步骤4: 拼接并预测 ────────────────────────────────────────
        A_tilde_test = np.hstack([yy1, TT2, Phi_test])
        NetoutTest = A_tilde_test @ self.W

        return NetoutTest

    def incremental_fit(self, train_x_new, train_y_new, train_q_new):
        """增量学习 (公式31, 32)

        使用Woodbury���等���高效更新权重

        参数:
            train_x_new: 新增数据驱动输入
            train_y_new: 新增目标输出
            train_q_new: 新增物理驱动输入

        返回:
            增量训练输出
        """
        # 计算新增数据的混合系统矩阵
        n_new = train_x_new.shape[0]

        # 特征映射层
        H1_new = np.hstack([train_x_new, 0.1 * np.ones((n_new, 1))])
        y_new = np.zeros((n_new, self.FeatureWindows * self.FeatureNodes))

        for i in range(self.FeatureWindows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = H1_new @ WeightFeaSparse
            TT1 = (TT1 - self.meanOfEachWindow[i]) / (self.distOfMaxAndMin[i] + 1e-12)
            y_new[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = TT1

        # 增强层
        H2_new = np.hstack([y_new, 0.1 * np.ones((n_new, 1))])
        T2_new = self._tansig(H2_new @ self.WeightEnhan)

        # 物理节点
        Phi_new = self.compute_physics_node(train_q_new, self.m, self.g, self.d)

        # 拼接增量混合矩阵
        A_new = np.hstack([y_new, T2_new, Phi_new])

        # 使用Woodbury恒等式更新P (公式32)
        P_new = self._woodbury_inverse(self.P, A_new)

        # 公式(31): W* = W + P* @ A_a.T @ (Y_a - A_a @ W)
        residual = train_y_new - A_new @ self.W
        W_increment = P_new @ A_new.T @ residual
        self.W = self.W + W_increment

        self.P = P_new

        return train_y_new


def demo():
    """演示函数"""
    np.random.seed(42)

    n_train = 500
    n_test = 100
    D = 10  # 数据输入维度
    B = 3   # 物理输入维度 [q, q_dot, q_acc]
    C = 1   # 输出维度

    # 生成训练数据
    train_x = np.random.randn(n_train, D)
    q = np.random.randn(n_train, 1) * 0.5  # 关节角
    q_dot = np.random.randn(n_train, 1) * 0.3  # 角速度
    q_acc = np.random.randn(n_train, 1) * 0.2  # 角加速度
    train_q = np.hstack([q, q_dot, q_acc])

    m, g, d = 2.0, 9.81, 0.3
    I_cm = (1/12) * m * (d ** 2)
    M_q = I_cm + m * (d ** 2)
    train_y = M_q * q_acc + m * g * d * np.cos(q) + 0.1 * np.random.randn(n_train, 1)

    # 测试数据
    test_x = np.random.randn(n_test, D)
    q_test = np.random.randn(n_test, 1) * 0.5
    q_dot_test = np.random.randn(n_test, 1) * 0.3
    q_acc_test = np.random.randn(n_test, 1) * 0.2
    test_q = np.hstack([q_test, q_dot_test, q_acc_test])
    test_y = M_q * q_acc_test + m * g * d * np.cos(q_test)

    print("=" * 50)
    print("混合宽度学习系统 (HybridBLS) 演示")
    print("=" * 50)

    # ── 训练 ───────────────────────────────────────────────────────────────
    print("\n[1] 训练...")
    model = HybridBLS(NumFeatureNodes=5, NumWindows=50, NumEnhance=100,
                    lambda1=2**-30, lambda2=2**-30)
    train_out = model.fit(train_x, train_y, train_q, m=m, g=g, d=d)
    pred_test = model.predict(test_x, test_q)

    mse = np.mean((pred_test - test_y) ** 2)
    print(f"    训练完成，测试MSE: {mse:.6f}")

    # ── 增量学习 ─────────────────────────────────────────────
    print("\n[2] 增量学习...")
    n_incr = 100
    train_x_incr = np.random.randn(n_incr, D)
    q_incr = np.random.randn(n_incr, 1) * 0.5
    q_dot_incr = np.random.randn(n_incr, 1) * 0.3
    q_acc_incr = np.random.randn(n_incr, 1) * 0.2
    train_q_incr = np.hstack([q_incr, q_dot_incr, q_acc_incr])
    train_y_incr = M_q * q_acc_incr + m * g * d * np.cos(q_incr)

    model.incremental_fit(train_x_incr, train_y_incr, train_q_incr)
    pred_incr = model.predict(test_x, test_q)
    mse_incr = np.mean((pred_incr - test_y) ** 2)
    print(f"    增量学习后测试MSE: {mse_incr:.6f}")

    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    demo()