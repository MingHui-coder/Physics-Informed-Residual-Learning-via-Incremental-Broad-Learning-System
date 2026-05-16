"""
@File    :   BroadLearningSystem.py
@Time    :   2026/04/22
@Author  :   Minghui Zhang
@Version :   1.0
@Contact :   [你的邮箱]
@Desc    :   负责BLS的回归、分类。包含模型的初始化、训练、预测。
"""
import numpy as np
import pandas as pd
import torch

from scipy import linalg as LA
from scipy import io as scio
from numpy import random
from sklearn import preprocessing

#网格搜索，搜索得到最佳参数
class GridSearchCV:
    def __init__(self) -> None:
        pass

    def run():
        return None

class BLS:
    def __init__(self, NumFeatureNodes=10, NumWindows=100, NumEnhance=1000, S=0.5, C=2**-30, is_argmax=True):
        self.FeatureNodes = NumFeatureNodes
        self.FeatureWindows = NumWindows  
        self.EnhancementNodes = NumEnhance
        self.S = S
        self.C = C
        self.is_argmax = is_argmax

    def _tansig(self, x):
        return (2/(1+np.exp(-2*x)))-1
    
    def _relu(self, x):
        return np.maximum(0, x)

    def _pinv(self, A, reg):
        return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
    
    def _pinv_cls(self, matrix):
        return np.mat(self.C * np.eye(matrix.shape[1]) + matrix.T.dot(matrix)).I.dot(matrix.T)

    def _shrinkage(self, a, b):
        z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
        return z

    def _sparse_bls(self, A, b):
        lam = 0.001
        itrs = 50
        AA = np.dot(A.T, A)
        m = A.shape[1]
        n = b.shape[1]
        wk = np.zeros([m, n], dtype='double')
        ok = np.zeros([m, n], dtype='double')
        uk = np.zeros([m, n], dtype='double')
        L1 = np.mat(AA + np.eye(m)).I
        L2 = np.dot(np.dot(L1, A.T), b)
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + np.dot(L1, tempc)
            ok = self._shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

class BLSRegressor(BLS):
    """BLS 回归模型，支持 CPU / CUDA GPU 训练。

    通过 `device` 参数切换后端：'cpu' 使用 PyTorch CPU 张量，
    'cuda' 或 'cuda:N' 使用 GPU 加速。
    """
    def __init__(self, NumFeatureNodes=10, NumWindows=100, NumEnhance=1000,
                 S=0.5, C=2**-30, device='cpu'):
        super().__init__(NumFeatureNodes, NumWindows, NumEnhance, S, C)
        self.device = torch.device(device)

    # ── torch 后端辅助函数 ────────────────────────────────────────
    @staticmethod
    def _tansig(x):
        return (2 / (1 + torch.exp(-2 * x))) - 1

    @staticmethod
    def _pinv(A, reg):
        """正则化伪逆: (reg*I + A^T A)^{-1} A^T"""
        I = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        M = A.T @ A + reg * I
        return torch.linalg.solve(M, A.T)

    @staticmethod
    def _shrinkage(a, b):
        return torch.clamp(a - b, min=0) - torch.clamp(-a - b, min=0)

    @staticmethod
    def _sparse_bls(A, b):
        lam = 0.001
        itrs = 50
        m = A.shape[1]
        n = b.shape[1]
        AA = A.T @ A
        wk = torch.zeros(m, n, device=A.device, dtype=A.dtype)
        ok = torch.zeros(m, n, device=A.device, dtype=A.dtype)
        uk = torch.zeros(m, n, device=A.device, dtype=A.dtype)

        M = AA + torch.eye(m, device=A.device, dtype=A.dtype)
        L1_inv = torch.linalg.inv(M)          # (AA + I)^{-1}
        L2 = L1_inv @ (A.T @ b)               # L1_inv @ A^T @ b
        for i in range(itrs):
            tempc = ok - uk
            ck = L2 + L1_inv @ tempc
            ok = BLSRegressor._shrinkage(ck + uk, lam)
            uk += ck - ok
            wk = ok
        return wk

    def _to_tensor(self, x):
        """统一将 ndarray / Tensor 转为 device 上的 float32 张量。"""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=self.device, dtype=torch.float32)
        return x.to(device=self.device, dtype=torch.float32)

    # ── 训练 ──────────────────────────────────────────────────────
    def fit(self, train_x, train_y):
        train_x = self._to_tensor(train_x)
        train_y = self._to_tensor(train_y)
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(1)

        u = 0
        WF = []
        for i in range(self.FeatureWindows):
            torch.manual_seed(i + u)
            WeightFea = 2 * torch.randn(
                train_x.shape[1] + 1, self.FeatureNodes,
                device=self.device, dtype=torch.float32) - 1
            WF.append(WeightFea)

        self.WeightEnhan = 2 * torch.randn(
            self.FeatureWindows * self.FeatureNodes + 1, self.EnhancementNodes,
            device=self.device, dtype=torch.float32) - 1

        H1 = torch.cat([
            train_x,
            0.1 * torch.ones(train_x.shape[0], 1, device=self.device, dtype=torch.float32)
        ], dim=1)
        y = torch.zeros(train_x.shape[0], self.FeatureWindows * self.FeatureNodes,
                        device=self.device, dtype=torch.float32)
        self.WFSparse = []
        self.distOfMaxAndMin = torch.zeros(self.FeatureWindows, device=self.device)
        self.meanOfEachWindow = torch.zeros(self.FeatureWindows, device=self.device)

        for i in range(self.FeatureWindows):
            WeightFea = WF[i]
            A1 = H1 @ WeightFea

            # MinMaxScaler 映射到 (-1, 1)
            min_vals = A1.min(dim=0, keepdim=True)[0]
            max_vals = A1.max(dim=0, keepdim=True)[0]
            A1 = 2 * (A1 - min_vals) / (max_vals - min_vals + 1e-12) - 1

            WeightFeaSparse = self._sparse_bls(A1, H1).T
            self.WFSparse.append(WeightFeaSparse)

            T1 = H1 @ WeightFeaSparse
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / (self.distOfMaxAndMin[i] + 1e-12)
            y[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = T1

        H2 = torch.cat([
            y,
            0.1 * torch.ones(y.shape[0], 1, device=self.device, dtype=torch.float32)
        ], dim=1)
        T2 = self._tansig(H2 @ self.WeightEnhan)
        T3 = torch.cat([y, T2], dim=1)

        # WeightTop = pinv(T3) @ train_y
        self.WeightTop = self._pinv(T3, self.C) @ train_y
        NetoutTrain = T3 @ self.WeightTop
        return NetoutTrain

    # ── 预测 ──────────────────────────────────────────────────────
    def predict(self, test_x):
        test_x = self._to_tensor(test_x)

        HH1 = torch.cat([
            test_x,
            0.1 * torch.ones(test_x.shape[0], 1, device=self.device, dtype=torch.float32)
        ], dim=1)
        yy1 = torch.zeros(test_x.shape[0], self.FeatureWindows * self.FeatureNodes,
                          device=self.device, dtype=torch.float32)

        for i in range(self.FeatureWindows):
            WeightFeaSparse = self.WFSparse[i]
            TT1 = HH1 @ WeightFeaSparse
            TT1 = (TT1 - self.meanOfEachWindow[i]) / (self.distOfMaxAndMin[i] + 1e-12)
            yy1[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = TT1

        HH2 = torch.cat([
            yy1,
            0.1 * torch.ones(yy1.shape[0], 1, device=self.device, dtype=torch.float32)
        ], dim=1)
        TT2 = self._tansig(HH2 @ self.WeightEnhan)
        TT3 = torch.cat([yy1, TT2], dim=1)
        NetoutTest = TT3 @ self.WeightTop
        return NetoutTest

    # ── 设备迁移 ──────────────────────────────────────────────────
    def to(self, device):
        """将模型所有张量移至指定设备。"""
        self.device = torch.device(device)
        self.WeightEnhan = self.WeightEnhan.to(self.device)
        self.WeightTop = self.WeightTop.to(self.device)
        self.WFSparse = [w.to(self.device) for w in self.WFSparse]
        self.distOfMaxAndMin = self.distOfMaxAndMin.to(self.device)
        self.meanOfEachWindow = self.meanOfEachWindow.to(self.device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device='cuda'):
        return self.to(device)

class BLSClassifier(BLS):
    def _init_(self, NumFeatureNodes=10, NumWindows=10, NumEnhance=10, S=0.5, C=2**-30, is_argmax=True):
        super()._init_()
        self.is_argmax = is_argmax
    def fit(self, train_x, train_y, is_excel_label=False):
        """模型本体"""

        if is_excel_label:
            train_y = [[i] for i in train_y]
            encoder = preprocessing.OneHotEncoder()
            encoder.fit(train_y)
            train_y = encoder.transform(train_y).toarray()

        # --Train--
        train_x = preprocessing.scale(train_x, axis=1)  # 标准化处理样本
        Feature_InputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])  # 将输入矩阵进行行链接，即平铺展开整个矩阵
        Output_FeatureMappingLayer = np.zeros([train_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        self.Beta1_EachWindow = []
        self.Dist_MaxAndMin = []
        self.Min_EachWindow = []
        self.ymin = 0
        self.ymax = 1

        # 特征层
        for i in range(self.FeatureWindows):
            random.seed(i + 2022)
            W_EachWindow = 2 * random.randn(train_x.shape[1] + 1, self.FeatureNodes) - 1  # 随机化特征层初始权重
            Feature_EachWindow = np.dot(Feature_InputDataWithBias, W_EachWindow)  # 计算每个特征映射中间态

            # scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Feature_EachWindow)                      # 对上述结果归一化处理
            # Feature_EachWindowAfterPreprocess = scaler1.transform(Feature_EachWindow)                               # 进行标准化

            Feature_EachWindowAfterPreprocess = Feature_EachWindow  # 进行标准化
            Beta_EachWindow = self._sparse_bls(Feature_EachWindowAfterPreprocess, Feature_InputDataWithBias).T  # 随机化特征映射初始偏置
            self.Beta1_EachWindow.append(Beta_EachWindow)
            Output_EachWindow = np.dot(Feature_InputDataWithBias, Beta_EachWindow)  # 计算每个特征映射最终输出

            self.Dist_MaxAndMin.append(np.max(Output_EachWindow, axis=0) - np.min(Output_EachWindow, axis=0))  # 计算损失函数
            self.Min_EachWindow.append(np.min(Output_EachWindow, axis=0))
            Output_EachWindow = (Output_EachWindow - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i]
            # 计算特征层最终输出
            Output_FeatureMappingLayer[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = Output_EachWindow

        # 增强层
        train_ori_hance = np.hstack([train_x])
        train_x_enhance = preprocessing.scale(train_ori_hance, axis=1)
        Input_EnhanceLayerWithBias = np.hstack([train_x_enhance, 0.1 * np.ones((train_x_enhance.shape[0], 1))])

        self.W_EnhanceLayer = LA.orth(2 * random.randn(train_x_enhance.shape[1] + 1, self.EnhancementNodes)) - 1

        Temp_Output_EnhanceLayer = np.dot(Input_EnhanceLayerWithBias, self.W_EnhanceLayer)  # 计算增强层中间态
        self.Parameter_Shrink = self.S / np.max(Temp_Output_EnhanceLayer)
        Output_EnhanceLayer = self._relu(Temp_Output_EnhanceLayer * self.Parameter_Shrink)  # 计算增强层最终输出

        # 输出层
        Input_OutputLayer = np.hstack([Output_FeatureMappingLayer, Output_EnhanceLayer])  # 合并特征层和增强层作为输出层输入
        _pinv_Output = self._pinv_cls(Input_OutputLayer)  # 计算伪逆

        # 计算系统总权重
        self.W = np.dot(_pinv_Output, train_y)

        OutputOfTrain = np.dot(Input_OutputLayer, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTrain.argmax(axis=1)
            # print(predlabel)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTrain
        
        return predlabel
    
    
    def predict(self, test_x):
        test_x = preprocessing.scale(test_x, axis=1)
        Feature_InputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        Output_FeatureMappingLayerTest = np.zeros([test_x.shape[0], self.FeatureWindows * self.FeatureNodes])

        for i in range(self.FeatureWindows):
            Output_EachWindowTest = np.dot(Feature_InputDataWithBiasTest, self.Beta1_EachWindow[i])
            Output_FeatureMappingLayerTest[:, self.FeatureNodes * i:self.FeatureNodes * (i + 1)] = (self.ymax - self.ymin) * (Output_EachWindowTest - self.Min_EachWindow[i]) / self.Dist_MaxAndMin[i] - self.ymin

        test_ori_hance = np.hstack([test_x])
        test_x_enhance = test_ori_hance
        Input_EnhanceLayerWithBiasTest = np.hstack([test_x_enhance, 0.1 * np.ones((test_x_enhance.shape[0], 1))])
        Temp_Output_EnhanceLayerTest = np.dot(Input_EnhanceLayerWithBiasTest, self.W_EnhanceLayer)
        Output_EnhanceLayerTest = self._relu(Temp_Output_EnhanceLayerTest * self.Parameter_Shrink)
        Input_OutputLayerTest = np.hstack([Output_FeatureMappingLayerTest, Output_EnhanceLayerTest])  # 合并特征层和增强层作为测试输出层输入

        OutputOfTest = np.dot(Input_OutputLayerTest, self.W)  # 计算预测输出

        if self.is_argmax:
            predlabel = OutputOfTest.argmax(axis=1)
            # 预测标签解嵌套
            predlabel = [int(i) for j in predlabel for i in j]
        else:
            predlabel = OutputOfTest
        
        return predlabel
