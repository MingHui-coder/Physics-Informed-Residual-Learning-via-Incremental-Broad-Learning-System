"""
data_utils_final.py

描述:   基于 final_dataset 的数据预处理工具集。读取 CSV 文件，通过滑动窗口提取
       thigh/shank 特征，可选择预测 knee_moment 或 knee_moment_minus_T_physics，
       生成并保存训练/测试 PyTorch 张量。
作者:   Minghui Zhang
创建:   2026-04
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 目标列名 → 文件标签映射
TARGET_LABEL_MAP = {
    "knee_moment": "BLS",
    "knee_moment_minus_T_physics": "Ph-BLS",
}


def select_action_interactive():
    """
    终端交互：让用户选择动作类别。
    返回 "level_walking"、"stair_ascent" 或 "stair_descent"。
    """
    print("\n" + "="*50)
    print("[*] 请选择动作类别：")
    print("    1 - level_walking (平地行走)")
    print("    2 - stair_ascent (上楼梯)")
    print("    3 - stair_descent (下楼梯)")
    print("="*50)
    while True:
        try:
            choice = input("\n[*] 请输入编号 (1/2/3): ").strip()
            if choice == "1":
                return "level_walking"
            elif choice == "2":
                return "stair_ascent"
            elif choice == "3":
                return "stair_descent"
            else:
                print("[!] 输入无效，请输入 1、2 或 3。")
        except Exception:
            print("[!] 输入无效，请重新输入。")


def select_target_column_interactive():
    """
    终端交互：让用户选择预测目标。
    返回 "knee_moment" 或 "knee_moment_minus_T_physics"。
    """
    print("\n" + "="*50)
    print("[*] 请选择预测目标列：")
    print("    1 - knee_moment (原始膝关节力矩)")
    print("    2 - knee_moment_minus_T_physics (力矩与物理扭矩的差值)")
    print("="*50)
    while True:
        try:
            choice = input("\n[*] 请输入编号 (1 或 2): ").strip()
            if choice == "1":
                return "knee_moment"
            elif choice == "2":
                return "knee_moment_minus_T_physics"
            else:
                print("[!] 输入无效，请输入 1 或 2。")
        except Exception:
            print("[!] 输入无效，请重新输入。")


def process_single_csv(csv_path, target_column="knee_moment", seq_len=20):
    """
    处理单个 CSV 文件，提取滑动窗口特征和标签。
    通过 target_column 参数选择预测目标：
      - "knee_moment" : 原始膝关节力矩
      - "knee_moment_minus_T_physics" : 力矩与物理扭矩的差值
    """
    df = pd.read_csv(csv_path)

    # 提取 'thigh' 和 'shank' 列作为输入特征
    features = df[['thigh', 'shank']].values

    # 提取目标列
    if target_column not in df.columns:
        raise ValueError(f"目标列 '{target_column}' 不存在于 CSV 文件中。"
                         f"可用列: {list(df.columns)}")
    targets = df[target_column].values

    X_local = []
    y_local = []

    for i in range(len(features) - seq_len + 1):
        X_local.append(features[i : i + seq_len])
        y_local.append(targets[i + seq_len - 1])

    return X_local, y_local


def build_dataset_from_directory(root_dir, target_column="knee_moment",
                                 exclude_subject="A25", seq_len=20):
    """
    遍历整个数据集文件夹，过滤掉指定受试者，将所有数据拼接为一个大型数据集。
    """
    root_path = Path(root_dir)
    all_X = []
    all_y = []

    csv_files = list(root_path.rglob("*.csv"))
    print(f"[*] 在目录 '{root_dir}' 下共扫描到 {len(csv_files)} 个 CSV 文件。")

    valid_files_count = 0
    skipped_files_count = 0

    for csv_file in csv_files:
        parent_folder_name = csv_file.parent.name

        if exclude_subject in parent_folder_name:
            skipped_files_count += 1
            continue

        X_local, y_local = process_single_csv(
            csv_file, target_column=target_column, seq_len=seq_len
        )

        all_X.extend(X_local)
        all_y.extend(y_local)

        valid_files_count += 1

    print(f"[*] 提取完毕！处理了 {valid_files_count} 个文件，"
          f"跳过了 {skipped_files_count} 个包含 '{exclude_subject}' 的文件。")

    print("[*] 正在将海量数据拼合为 PyTorch Tensor，请稍候...")
    X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_y), dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor


def generate_training_tensor(dataset_name, action,
                             target_column="knee_moment", seq_len=20,
                             base_dir="dataset_porecessing/final_dataset",
                             dataset_type="unfiltered"):
    """
    根据数据集和动作，自动路由对应的路径和排除规则，并固化保存张量。

    参数:
        dataset_name (str): 数据集名称，例如 "datasetA" 或 "datasetB"
        action (str): 动作类别，例如 "level_walking", "stair_ascent", "stair_descent"
        target_column (str): 预测目标列名
                             "knee_moment" 或 "knee_moment_minus_T_physics"
        seq_len (int): 时间窗口长度
        base_dir (str): 数据集根目录路径
        dataset_type (str): "filtered" 或 "unfiltered"
    """
    if dataset_name == "datasetA":
        exclude_subject = "AB25"
    elif dataset_name == "datasetB":
        exclude_subject = "AB02"
    else:
        raise ValueError(f"未知的数据集: {dataset_name}。目前仅支持 datasetA 或 datasetB")

    target_dir = Path(base_dir) / dataset_name / action

    if not target_dir.exists():
        raise FileNotFoundError(f"找不到指定的目录: {target_dir}，请检查文件夹结构。")

    print("\n" + "="*50)
    print(f"[*] 任务启动: {dataset_name} | 动作: {action}")
    print(f"[*] 目标列: {target_column}")
    print(f"[*] 工作路径: {target_dir}")
    print(f"[*] 排除规则: 剔除受试者 {exclude_subject}")
    print("="*50)

    X_train, y_train = build_dataset_from_directory(
        root_dir=target_dir,
        target_column=target_column,
        exclude_subject=exclude_subject,
        seq_len=seq_len
    )

    print(f"[*] 张量维度: X shape: {X_train.shape}, y shape: {y_train.shape}")

    out_path = Path("tensor_generating/tensors/train")
    out_path.mkdir(parents=True, exist_ok=True)
    label = TARGET_LABEL_MAP.get(target_column, target_column)

    # filtered数据集在文件名最前加 "filtered_"
    if dataset_type == "filtered":
        save_filename = out_path / f"Train_filtered_{dataset_name}__{action}__{label}.pt"
    else:
        save_filename = out_path / f"Train_{dataset_name}__{action}__{label}.pt"

    torch.save({"X_data": X_train, "y_label": y_train}, save_filename)

    print(f"[*] 恭喜！专属数据包已保存至: {save_filename}\n")
    return save_filename


def build_test_dataset_interactive(root_dir, target_subject,
                                   target_column="knee_moment", seq_len=20):
    """
    专门查找目标受试者的文件夹，列出所有 CSV 文件，并在终端等待用户选择。
    """
    root_path = Path(root_dir)

    all_csv_files = list(root_path.rglob("*.csv"))

    target_files = []
    for f in all_csv_files:
        if target_subject in f.parent.name:
            target_files.append(f)

    if len(target_files) == 0:
        raise ValueError(f"[!] 没有找到属于受试者 '{target_subject}' 的 CSV 文件！请检查目录。")

    print(f"\n[*] 发现目标受试者 {target_subject} 的以下测试数据：")
    for idx, f in enumerate(target_files):
        print(f"    [{idx}] {f.name}  (目录: {f.parent.name})")

    while True:
        try:
            user_input = input(
                f"\n[*] 请输入你要构建测试集的 CSV 文件编号 "
                f"(0 - {len(target_files)-1}): "
            )
            choice_idx = int(user_input)

            if 0 <= choice_idx < len(target_files):
                selected_file = target_files[choice_idx]
                break
            else:
                print(f"[!] 编号越界，请重新输入 0 到 {len(target_files)-1} 之间的数字。")
        except ValueError:
            print("[!] 格式错误，请输入纯数字编号。")

    print(f"\n[*] 你选择了: {selected_file.name}，正在提取滑动窗口数据...")

    X_local, y_local = process_single_csv(
        selected_file, target_column=target_column, seq_len=seq_len
    )

    print("[*] 正在转换为 PyTorch Tensor...")
    X_test_tensor = torch.tensor(np.array(X_local), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_local), dtype=torch.float32).unsqueeze(1)

    return X_test_tensor, y_test_tensor, selected_file.stem


def generate_test_tensor(dataset_name, action,
                         target_column="knee_moment", seq_len=20,
                         base_dir="dataset_porecessing/final_dataset",
                         output_dir="tensor_generating/tensors/test",
                         dataset_type="unfiltered"):
    """
    测试集路由器：定位留出受试者，调用交互选择，并保存测试集张量。
    """
    if dataset_name == "datasetA":
        target_subject = "AB25"
    elif dataset_name == "datasetB":
        target_subject = "AB02"
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

    target_dir = Path(base_dir) / dataset_name / action
    if not target_dir.exists():
        raise FileNotFoundError(f"找不到指定的目录: {target_dir}")

    print("\n" + "="*50)
    print(f"[*] 【测试集构建】任务启动: {dataset_name} | 动作: {action}")
    print(f"[*] 目标列: {target_column}")
    print(f"[*] 目标受试者: {target_subject}")
    print("="*50)

    X_test, y_test, selected_csv_name = build_test_dataset_interactive(
        root_dir=target_dir,
        target_subject=target_subject,
        target_column=target_column,
        seq_len=seq_len
    )

    print(f"[*] 测试集张量维度: X shape: {X_test.shape}, y shape: {y_test.shape}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    label = TARGET_LABEL_MAP.get(target_column, target_column)
    # filtered数据集在文件名最前加 "filtered_"
    if dataset_type == "filtered":
        save_filepath = out_path / f"Test_filtered_{dataset_name}__{action}__{label}__{selected_csv_name}.pt"
    else:
        save_filepath = out_path / f"Test_{dataset_name}__{action}__{label}__{selected_csv_name}.pt"

    torch.save({"X_data": X_test, "y_label": y_test}, save_filepath)
    print(f"[*] 恭喜！测试集数据包已安全保存至: {save_filepath}\n")

    return str(save_filepath)
