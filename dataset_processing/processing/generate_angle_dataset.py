"""
@File    :   generate_angle_dataset.py
@Time    :   2026/04/21
@Author  :   Minghui Zhang
@Version :   1.0
@Contact :   [你的邮箱]
@Desc    :   得到中间状态数据集。计算 垂直于地面为背景的小腿角度 与 小腿角度的二阶导数 
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_csv_file(filepath):
    """加载CSV文件：第一行是列名，后面是逗号分隔的数据"""
    # 使用pandas读取CSV文件
    df = pd.read_csv(filepath)
    # 索引从1开始（原文件格式）
    df.index = range(1, len(df) + 1)
    return df

def process_shank_column(df):
    """创建shank_new列：(shank - 180)度转换为弧度"""
    if 'shank' not in df.columns:
        raise ValueError("DataFrame中未找到'shank'列")

    # 角度转弧度：弧度 = 角度 * π / 180
    # 所以 (shank - 180)度 转弧度 = (shank - 180) * π / 180
    df['shank_new'] = (df['shank'] - 180.0) * np.pi / 180.0
    return df

def compute_second_derivative(df, freq=200.0):
    """
    计算shank_new对时间的二阶导数（角加速度，单位：弧度/秒²）

    参数：
        df: 包含'shank_new'列的DataFrame
        freq: 采样频率，默认200 Hz

    计算方法：
        1. 时间步长 dt = 1/freq 秒（200Hz对应dt=0.005秒）
        2. 使用numpy.gradient计算一阶导数（角速度，单位：弧度/秒）
           numpy.gradient使用中心差分法，对于均匀采样数据：
           f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        3. 对一阶导数再次使用numpy.gradient计算二阶导数（角加速度）
        4. 结果保存为新列'shank_new_second derivative'

    注意：这是数值二阶导数，近似于解析二阶导数d²θ/dt²
    """
    if 'shank_new' not in df.columns:
        raise ValueError("DataFrame中未找到'shank_new'列")

    dt = 1.0 / freq  # 时间步长（秒）
    # 一阶导数：角速度 = dθ/dt （弧度/秒）
    first_deriv = np.gradient(df['shank_new'].values, dt)
    # 二阶导数：角加速度 = d²θ/dt² （弧度/秒²）
    second_deriv = np.gradient(first_deriv, dt)
    df['shank_new_second derivative'] = second_deriv
    return df

def save_processed_file(df, original_path, output_root):
    """
    保存处理后的DataFrame，保持目录结构

    输出路径结构：output_root/datasetA/... (保持datasetA之后的原始目录结构)
    例如：
      输入：original_dataset/datasetA/level_walking/AB06/file.csv
      输出：output_root/datasetA/level_walking/AB06/file.csv
    """
    original_path = Path(original_path)
    output_root = Path(output_root)

    # 获取原始路径的各个部分
    parts = original_path.parts

    # 查找datasetA在路径中的位置
    try:
        datasetA_idx = parts.index('datasetA')
    except ValueError:
        # 如果找不到datasetA，记录警告并假设datasetA应该在路径中
        print(f"警告：路径中未找到'datasetA'目录：{original_path}")
        # 创建一个以datasetA为根目录的路径
        # 使用原始路径的最后一部分作为文件名，其余部分作为子目录
        if len(parts) >= 2:
            # 保留最后一部分作为文件名，其余作为目录
            dir_parts = parts[:-1]
            file_name = parts[-1]
            relative_parts = ('datasetA',) + tuple(dir_parts) + (file_name,)
        else:
            # 如果只有一个部分，直接作为文件名
            relative_parts = ('datasetA', parts[0])
    else:
        # 找到datasetA，使用datasetA之后的部分
        relative_parts = parts[datasetA_idx:]  # 从datasetA开始的部分

    # 构建输出路径
    output_path = output_root.joinpath(*relative_parts)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 确定列顺序：保持原始列顺序 + 新列
    # 原始列顺序从文件读取，新列添加到末尾
    original_cols = [col for col in df.columns if col not in ['shank_new', 'shank_new_second derivative']]
    new_cols = ['shank_new', 'shank_new_second derivative']
    all_cols = original_cols + new_cols

    # 选择要保存的列（按顺序）
    df_to_save = df[all_cols].copy()

    # 保存为CSV，保持与原文件相同格式（逗号分隔，没有索引列）
    df_to_save.to_csv(output_path, index=False)

    print(f"已保存：{output_path}")
    return output_path

def process_all_files(input_root, output_root, max_files=None):
    """处理input_root下的所有CSV文件"""
    input_root = Path(input_root)
    output_root = Path(output_root)

    # 查找所有CSV文件
    csv_files = list(input_root.rglob('*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件需要处理。")

    if max_files is not None:
        csv_files = csv_files[:max_files]
        print(f"只处理前 {max_files} 个文件进行测试")

    for i, csv_file in enumerate(csv_files):
        print(f"正在处理 ({i+1}/{len(csv_files)})：{csv_file}")
        try:
            df = load_csv_file(csv_file)
            df = process_shank_column(df)
            df = compute_second_derivative(df, freq=200.0)
            save_processed_file(df, csv_file, output_root)
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错：{e}")
            import traceback
            traceback.print_exc()

def test_single_file():
    """测试单个文件处理"""
    print("=== 测试单个文件处理 ===")
    test_file = Path('original_dataset/datasetA/level_walking/AB06/levelground_ccw_fast_01_01_L.csv')

    if not test_file.exists():
        print(f"测试文件不存在：{test_file}")
        return

    try:
        # 加载文件
        df = load_csv_file(test_file)
        print(f"原始数据形状：{df.shape}")
        print(f"原始列名：{list(df.columns)}")
        print(f"前3行数据：")
        print(df.head(3))

        # 处理shank列
        df = process_shank_column(df)
        print(f"\n添加shank_new列后的数据形状：{df.shape}")
        print(f"shank_new前3个值：{df['shank_new'].head(3).values}")

        # 计算二阶导数
        df = compute_second_derivative(df, freq=200.0)
        print(f"\n添加二阶导数列后的数据形状：{df.shape}")
        print(f"shank_new_second derivative前3个值：{df['shank_new_second derivative'].head(3).values}")

        # 保存文件
        output_root = Path('angle_second_order_derivative_dataset_test')
        output_root.mkdir(parents=True, exist_ok=True)
        save_processed_file(df, test_file, output_root)

        # 读取保存的文件验证
        saved_file = output_root / 'datasetA' / 'level_walking' / 'AB06' / 'levelground_ccw_fast_01_01_L.csv'
        if saved_file.exists():
            print(f"\n文件已保存到：{saved_file}")
            # 检查文件前几行
            with open(saved_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("保存的文件前3行：")
                for i in range(min(3, len(lines))):
                    print(f"  {lines[i].strip()}")
        else:
            print(f"保存的文件不存在：{saved_file}")

    except Exception as e:
        print(f"测试失败：{e}")
        import traceback
        traceback.print_exc()

def main():
    # 路径设置
    current_dir = Path(__file__).parent
    project_root = current_dir.parent  # 假设脚本在 dataset_processing/ 目录下
    datasetA_root = project_root / 'original_dataset' / 'datasetA'
    output_root = project_root / 'angle_second_order_derivative_dataset'

    if not datasetA_root.exists():
        print(f"数据集目录不存在：{datasetA_root}")
        return

    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    # 显式创建datasetA子目录，确保输出结构明确
    datasetA_output_dir = output_root / 'datasetA'
    datasetA_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录结构：{output_root}/datasetA/...")

    # 处理所有文件
    process_all_files(datasetA_root, output_root)

    print("处理完成。")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_single_file()
    else:
        main()