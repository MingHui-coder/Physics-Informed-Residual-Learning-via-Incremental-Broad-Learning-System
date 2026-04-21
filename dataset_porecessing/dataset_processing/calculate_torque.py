"""
@File    :   calculate_torque.py
@Time    :   2026/04/21
@Author  :   Minghui Zhang
@Version :   1.0
@Contact :   [你的邮箱]
@Desc    :   得到最终的数据集。计算 惯性力矩 + 重力力矩 
                 以及 真实力矩-(惯性力矩 + 重力力矩)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 物理参数
M = 3.0  # 质量 (kg)
L = 0.4  # 长度 (m) - 40cm转换为0.4m
L_COM = 0.2  # 质心位置 (m) - 20cm转换为0.2m
G = 9.8  # 重力加速度 (m/s^2)

# 计算转动惯量 I_s = (1/3) * m * L^2
I_S = (1.0/3.0) * M * (L ** 2)
print(f"物理参数:")
print(f"  质量 m = {M} kg")
print(f"  长度 L = {L} m (40cm)")
print(f"  质心位置 l_com = {L_COM} m (20cm)")
print(f"  重力加速度 g = {G} m/s^2")
print(f"  转动惯量 I_s = (1/3)*m*L^2 = {I_S:.6f} kg·m^2")
print()

def load_csv_file(filepath):
    """加载CSV文件"""
    df = pd.read_csv(filepath)
    return df

def calculate_t_physics(df):
    """
    计算物理扭矩 T_physics

    公式: T_physics = I_s * θ_s_ddot + m * g * l_com * sin(θ_s)

    其中:
      θ_s = shank_new (小腿角度，弧度)
      θ_s_ddot = shank_new_second derivative (小腿角加速度，弧度/秒^2)
      I_s = (1/3) * m * L^2
      m = 3 kg
      L = 0.4 m
      l_com = 0.2 m
      g = 9.8 m/s^2
    """
    # 检查必要的列是否存在
    required_cols = ['shank_new', 'shank_new_second derivative']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame中未找到必需的列: '{col}'")

    # 获取角度和角加速度
    theta_s = df['shank_new'].values  # 弧度
    theta_s_ddot = df['shank_new_second derivative'].values  # 弧度/秒^2

    # 计算 T_physics
    # 第一部分: I_s * θ_s_ddot
    inertial_term = I_S * theta_s_ddot

    # 第二部分: m * g * l_com * sin(θ_s)
    gravity_term = M * G * L_COM * np.sin(theta_s)

    # 总扭矩
    t_physics = inertial_term + gravity_term

    # 添加到DataFrame
    df['T_physics'] = t_physics

    return df

def calculate_torque_difference(df):
    """
    计算膝关节力矩与物理扭矩的差值

    公式: knee_moment - T_physics
    """
    if 'knee_moment' not in df.columns:
        raise ValueError("DataFrame中未找到必需的列: 'knee_moment'")
    if 'T_physics' not in df.columns:
        raise ValueError("DataFrame中未找到必需的列: 'T_physics'")

    df['knee_moment_minus_T_physics'] = df['knee_moment'] - df['T_physics']
    return df

def save_processed_file(df, original_path, output_root):
    """
    保存处理后的DataFrame，保持目录结构

    输出路径结构：output_root/datasetA/... (保持datasetA之后的原始目录结构)
    """
    original_path = Path(original_path)
    output_root = Path(output_root)

    # 获取原始路径的各个部分
    parts = original_path.parts

    # 查找datasetA在路径中的位置
    try:
        datasetA_idx = parts.index('datasetA')
        relative_parts = parts[datasetA_idx:]  # 从datasetA开始的部分
    except ValueError:
        # 如果找不到datasetA，记录警告
        print(f"警告：路径中未找到'datasetA'目录：{original_path}")
        # 创建一个以datasetA为根目录的路径
        if len(parts) >= 2:
            dir_parts = parts[:-1]
            file_name = parts[-1]
            relative_parts = ('datasetA',) + tuple(dir_parts) + (file_name,)
        else:
            relative_parts = ('datasetA', parts[0])

    # 构建输出路径
    output_path = output_root.joinpath(*relative_parts)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为CSV
    df.to_csv(output_path, index=False)

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
            # 加载文件
            df = load_csv_file(csv_file)

            # 计算T_physics
            df = calculate_t_physics(df)

            # 计算膝关节力矩与T_physics的差值
            df = calculate_torque_difference(df)

            # 保存文件
            save_processed_file(df, csv_file, output_root)

        except Exception as e:
            print(f"处理文件 {csv_file} 时出错：{e}")
            import traceback
            traceback.print_exc()

def test_single_file():
    """测试单个文件处理"""
    print("=== 测试单个文件处理 ===")

    # 使用测试文件
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    test_file = project_root / 'angle_second_order_derivative_dataset' / 'datasetA' / 'level_walking' / 'AB06' / 'levelground_ccw_fast_01_01_L.csv'

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

        # 计算T_physics
        df = calculate_t_physics(df)
        print(f"\n计算T_physics后的数据形状：{df.shape}")
        print(f"T_physics前3个值：{df['T_physics'].head(3).values}")

        # 计算差值
        df = calculate_torque_difference(df)
        print(f"\n计算差值后的数据形状：{df.shape}")
        print(f"knee_moment_minus_T_physics前3个值：{df['knee_moment_minus_T_physics'].head(3).values}")

        # 保存文件
        output_root = project_root / 'final_dataset_test'
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
    input_root = project_root / 'angle_second_order_derivative_dataset' / 'datasetA'
    output_root = project_root / 'final_dataset'

    if not input_root.exists():
        print(f"输入目录不存在：{input_root}")
        print("请先运行 generate_angle_dataset.py 生成角度和二阶导数数据")
        return

    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    # 显式创建datasetA子目录
    datasetA_output_dir = output_root / 'datasetA'
    datasetA_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录结构：{output_root}/datasetA/...")

    # 处理所有文件
    process_all_files(input_root, output_root)

    print("处理完成。")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_single_file()
    else:
        main()