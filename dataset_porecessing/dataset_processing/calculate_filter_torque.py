"""
calculate_filter_torque.py

描述:   对dataset_porecessing\final_dataset\datasetA中所有CSV文件进行滤波处理，
       滤波后的T_physics替代原值，用knee_moment-T_physics计算新的knee_moment_minus_T_physics，
       保存到dataset_porecessing\filter_final_dataset。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d
import shutil


# ===== 滤波函数 =====
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


FILTERS = {
    '1': ('Butterworth (低通)', smooth_butterworth, 'butter'),
    '2': ('Savitzky-Golay', smooth_savgol, 'savgol'),
    '3': ('高斯滤波', smooth_gaussian, 'gaussian'),
    '4': ('中值滤波', smooth_median, 'median'),
    '5': ('级联: 中值+Savitzky-Golay', 'cascade_savgol', 'cascade'),
}


def get_filter_params(filter_type):
    """获取滤波器参数"""
    if filter_type == 'savgol':
        window = int(input("请输入窗口大小 (window, 默认101): ") or "101")
        return {'window': window}
    elif filter_type == 'gaussian':
        sigma = float(input("请输入sigma: "))
        return {'sigma': sigma}
    elif filter_type == 'median':
        window = int(input("请输入窗口大小 (window): "))
        return {'window': window}
    elif filter_type == 'cascade':
        median_window = int(input("请输入中值窗口大小 (median_window, 默认5): ") or "5")
        savgol_window = int(input("请输入Savitzky-Golay窗口大小 (savgol_window, 默认101): ") or "101")
        return {'median_window': median_window, 'savgol_window': savgol_window}
    else:  # butter
        fs = float(input("请输入采样频率 (fs, Hz): "))
        fc = float(input("请输入截止频率 (cutoff, Hz): "))
        order = int(input("请输入滤波器阶数 (order, 默认4): ") or "4")
        return {'fs': fs, 'cutoff': fc, 'order': order}


def apply_filter(data, filter_type, params):
    """应用滤波器"""
    if filter_type == 'savgol':
        return smooth_savgol(data, **params)
    elif filter_type == 'gaussian':
        return smooth_gaussian(data, **params)
    elif filter_type == 'median':
        return smooth_median(data, **params)
    elif filter_type == 'butter':
        return smooth_butterworth(data, **params)
    elif filter_type == 'cascade':
        filters = [
            ('median', {'window': params['median_window']}),
            ('savgol', {'window': params['savgol_window']}),
        ]
        return smooth_cascade(data, filters)
    return data


def process_file(src_path, dst_path, filter_type, params):
    """处理单个CSV文件"""
    df = pd.read_csv(src_path)

    # 获取原始数据
    T_physics_raw = df['T_physics'].values
    knee_moment = df['knee_moment'].values

    # 滤波T_physics
    T_physics_filtered = apply_filter(T_physics_raw, filter_type, params)

    # 计算新的knee_moment_minus_T_physics
    knee_moment_minus_T_new = knee_moment - T_physics_filtered

    # 更新DataFrame
    df['T_physics'] = T_physics_filtered
    df['knee_moment_minus_T_physics'] = knee_moment_minus_T_new

    # 保存
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)

    return src_path.name, T_physics_filtered.min(), T_physics_filtered.max()


def main():
    src_dir = Path(r'C:\Users\LENOVO\Desktop\吴博士论文\code\dataset_porecessing\final_dataset\datasetA')
    dst_dir = Path(r'C:\Users\LENOVO\Desktop\吴博士论文\code\dataset_porecessing\filter_final_dataset\datasetA')

    print("=" * 50)
    print("批量滤波处理工具")
    print("=" * 50)

    # 选择滤波器
    print("\n请选择滤波算法:")
    for key, (name, _, _) in FILTERS.items():
        print(f"  {key}: {name}")

    filter_select = input("\n请输入滤波器编号: ").strip()

    if filter_select not in FILTERS:
        print("无效的选择，退出。")
        return

    filter_name, _, filter_type = FILTERS[filter_select]
    print(f"\n选择: {filter_name}")

    # 获取参数
    params = get_filter_params(filter_type)

    # 收集所有CSV文件
    csv_files = list(src_dir.rglob("*.csv"))
    print(f"\n找到 {len(csv_files)} 个CSV文件")

    if len(csv_files) == 0:
        print("未找到CSV文件，退出。")
        return

    # 确认
    confirm = input(f"\n将对所有文件应用 {filter_name} 滤波，确认处理? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return

    # 处理所有文件
    print("\n开始处理...")
    success_count = 0

    for src_path in csv_files:
        # 计算目标路径，保持文件夹结构
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path

        try:
            name, t_min, t_max = process_file(src_path, dst_path, filter_type, params)
            print(f"  [OK] {name} - T_physics范围: [{t_min:.4f}, {t_max:.4f}]")
            success_count += 1
        except Exception as e:
            print(f"  [FAIL] {src_path.name} - {e}")

    print(f"\n完成! 成功处理 {success_count}/{len(csv_files)} 个文件")
    print(f"输出目录: {dst_dir}")


if __name__ == "__main__":
    main()