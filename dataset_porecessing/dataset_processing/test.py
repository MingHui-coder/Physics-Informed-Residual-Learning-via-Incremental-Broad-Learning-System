"""
test.py

描述:   交互式对T_physics进行滤波，计算新的knee_moment_minus_T_physics，
       并可视化对比滤波效果。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d


# ===== 以下滤波函数来自 final_moment_prediction/data_processing_and_graphing.py =====
def smooth_savgol(y, window):
    """Savitzky-Golay 滤波。"""
    order = min(2, window - 2)
    return savgol_filter(y, window, order)


def smooth_gaussian(y, sigma):
    """高斯滤波。"""
    return gaussian_filter1d(y, sigma, mode='reflect')


def smooth_butterworth(y, cutoff, fs=1.0, order=4):
    """零相位 Butterworth 低通滤波。"""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, y)


def smooth_median(y, window):
    """中值滤波：专门去除毛刺/脉冲噪声。"""
    return medfilt(y, kernel_size=window)


def smooth_cascade(y, filters):
    """级联滤波：先中值去毛刺，再平滑"""
    for name, kwargs in filters:
        if name == "median":
            y = smooth_median(y, **kwargs)
        elif name == "savgol":
            y = smooth_savgol(y, **kwargs)
        elif name == "gaussian":
            y = smooth_gaussian(y, **kwargs)
        elif name == "butter":
            y = smooth_butterworth(y, **kwargs)
    return y
# ===== 滤波函数结束 =====


# 可用的滤波器列表
FILTERS = {
    '1': ('Butterworth (低通)', smooth_butterworth),
    '2': ('Savitzky-Golay (window=101)', smooth_savgol),
    '3': ('高斯滤波', smooth_gaussian),
    '4': ('中值滤波', smooth_median),
    '5': ('级联: 中值+Savitzky-Golay (强制平滑)', 'cascade_savgol'),
}


def load_data(filepath):
    """加载CSV数据"""
    return pd.read_csv(filepath)


def get_filter_params(filter_name):
    """获取滤波器参数"""
    if filter_name == '2':
        window = int(input("请输入窗口大小 (window, 默认101): ") or "101")
        return {'window': window}
    elif filter_name == '3':
        sigma = float(input("请输入sigma: "))
        return {'sigma': sigma}
    elif filter_name == '4':
        window = int(input("请输入窗口大小 (window): "))
        return {'window': window}
    elif filter_name == '5':
        median_window = int(input("请输入中值窗口大小 (median_window, 默认5): ") or "5")
        savgol_window = int(input("请输入Savitzky-Golay窗口大小 (savgol_window, 默认101): ") or "101")
        return {'median_window': median_window, 'savgol_window': savgol_window}
    else:
        fs = float(input("请输入采样频率 (fs, Hz): "))
        fc = float(input("请输入截止频率 (cutoff, Hz): "))
        order = int(input("请输入滤波器阶数 (order, 默认4): ") or "4")
        return {'fs': fs, 'cutoff': fc, 'order': order}


def apply_filter(data, filter_func, params):
    """应用滤波器"""
    if filter_func == 'cascade_savgol':
        filters = [
            ('median', {'window': params['median_window']}),
            ('savgol', {'window': params['savgol_window']}),
        ]
        return smooth_cascade(data, filters)
    return filter_func(data, **params)


def plot_comparison(time, T_raw, T_filtered, KM_raw, KM_filtered, filter_name, show_raw=True):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    if show_raw:
        axes[0].plot(time, T_raw, 'b-', label='T_physics (原始)', alpha=0.7)
    axes[0].plot(time, T_filtered, 'r-', label=f'T_physics ({filter_name})', linewidth=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('T_physics')
    axes[0].set_title('T_physics: 滤波后')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if show_raw:
        axes[1].plot(time, KM_raw, 'b-', label='knee_moment_minus_T_physics (原始)', alpha=0.7)
    axes[1].plot(time, KM_filtered, 'r-', label=f'knee_moment_minus_T_physics ({filter_name})', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('knee_moment_minus_T_physics')
    axes[1].set_title('knee_moment_minus_T_physics: 滤波后')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    filepath = r'C:\Users\LENOVO\Desktop\吴博士论文\code\dataset_porecessing\final_dataset\datasetA\level_walking\AB06\levelground_ccw_fast_01_01_L.csv'

    print("=" * 50)
    print("数据滤波与可视化工具")
    print("=" * 50)

    df = load_data(filepath)
    print(f"\n数据已加载，共 {len(df)} 行")

    T_physics_raw = df['T_physics'].values
    knee_moment = df['knee_moment'].values
    knee_moment_minus_T_raw = knee_moment - T_physics_raw
    n_samples = len(T_physics_raw)

    time = np.arange(n_samples) / 100

    print(f"\n原始T_physics范围: [{T_physics_raw.min():.4f}, {T_physics_raw.max():.4f}]")
    print(f"原始knee_moment_minus_T_physics范围: [{knee_moment_minus_T_raw.min():.4f}, {knee_moment_minus_T_raw.max():.4f}]")

    print("\n" + "=" * 50)
    filter_choice = input("是否对T_physics进行滤波? (y/n): ").strip().lower()

    T_physics_filtered = None
    knee_moment_minus_T_filtered = None

    if filter_choice == 'y':
        print("\n请选择滤波算法:")
        for key, (name, _) in FILTERS.items():
            print(f"  {key}: {name}")

        filter_select = input("\n请输入滤波器编号: ").strip()

        if filter_select in FILTERS:
            filter_name, filter_func = FILTERS[filter_select]
            print(f"\n选择: {filter_name}")

            params = get_filter_params(filter_select)

            print("正在滤波...")
            T_physics_filtered = apply_filter(T_physics_raw, filter_func, params)

            # knee_moment_minus_T_physics = knee_moment - T_physics
            knee_moment_minus_T_filtered = knee_moment - T_physics_filtered

            print(f"\n滤波后T_physics范围: [{T_physics_filtered.min():.4f}, {T_physics_filtered.max():.4f}]")
            print(f"滤波后knee_moment_minus_T_physics范围: [{knee_moment_minus_T_filtered.min():.4f}, {knee_moment_minus_T_filtered.max():.4f}]")
        else:
            print("无效的选择，将使用原始数据")

    print("\n" + "=" * 50)
    plot_choice = input("是否绘制对比图? (y/n): ").strip().lower()

    if plot_choice == 'y':
        filter_label = filter_name if T_physics_filtered is not None else "原始"
        show_raw = input("是否显示原始数据? (y/n, 默认y): ").strip().lower()
        if show_raw == 'n':
            show_raw = False
        else:
            show_raw = True

        if T_physics_filtered is not None:
            plot_comparison(time, T_physics_raw, T_physics_filtered,
                          knee_moment_minus_T_raw, knee_moment_minus_T_filtered, filter_label, show_raw)
        else:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            if show_raw:
                axes[0].plot(time, T_physics_raw, 'b-', label='T_physics (原始)', alpha=0.7)
                axes[0].legend()
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('T_physics')
            axes[0].set_title('T_physics (原始)')
            axes[0].grid(True, alpha=0.3)

            if show_raw:
                axes[1].plot(time, knee_moment_minus_T_raw, 'b-', label='knee_moment_minus_T_physics (原始)', alpha=0.7)
                axes[1].legend()
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('knee_moment_minus_T_physics')
            axes[1].set_title('knee_moment_minus_T_physics (原始)')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    print("\n完成!")


if __name__ == "__main__":
    main()