"""
BLS_model_training.py

描述:   终端交互选择训练集张量 (.pt)，使用 BLSRegressor 进行训练，
       并将训练好的模型保存至 Model_Training_Testing/models/。
作者:   Minghui Zhang
创建:   2026-04
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from BLSs_Method.BoradLearningSystem import BLSRegressor


def list_data_files(data_dir):
    """扫描目录下所有 .pt 文件，返回排序后的列表。"""
    p = Path(data_dir)
    return sorted(p.glob("*.pt"))


def select_file_interactive(files):
    """终端交互：让用户选择训练集文件。"""
    print("\n" + "=" * 50)
    print("[*] 请选择训练集文件：")
    for i, f in enumerate(files):
        print(f"    [{i}] {f.name}")
    print("=" * 50)
    while True:
        try:
            choice = input(f"\n[*] 请输入编号 (0 - {len(files) - 1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print(f"[!] 编号越界，请输入 0 到 {len(files) - 1} 之间的数字。")
        except ValueError:
            print("[!] 格式错误，请输入纯数字编号。")


def main():
    # ========================================
    # 1. 交互选择训练集
    # ========================================
    train_dir = Path("tensor_generating/tensors/train")
    if not train_dir.exists():
        print(f"[!] 训练集目录不存在: {train_dir}")
        return

    data_files = list_data_files(train_dir)
    if not data_files:
        print(f"[!] 目录 '{train_dir}' 下没有找到任何 .pt 文件。")
        return

    selected_file = select_file_interactive(data_files)
    print(f"\n[*] 你选择了: {selected_file.name}")

    # ========================================
    # 2. 加载数据
    # ========================================
    print("[*] 正在加载训练数据...")
    data = torch.load(selected_file)
    X_data = data["X_data"].numpy()   # shape: (n, seq_len, n_features)
    y_data = data["y_label"].numpy()  # shape: (n, 1)

    print(f"[*] 原始数据形状 — X: {X_data.shape}, y: {y_data.shape}")

    # 将 X 从 (n, seq_len, n_features) 展平为 (n, seq_len * n_features)
    n_samples = X_data.shape[0]
    X_flat = X_data.reshape(n_samples, -1)
    print(f"[*] 展平后 X 形状: {X_flat.shape}")

    # ========================================
    # 3. 训练 BLS 模型
    # ========================================
    print("[*] 正在初始化 BLSRegressor...")
    model = BLSRegressor(
        NumFeatureNodes=10,
        NumWindows=100,
        NumEnhance=1000,
        S=0.5,
        C=2 ** -30,
    )

    print("[*] 开始训练 (可能需要较长时间)...")
    model.fit(X_flat, y_data)
    print("[*] 训练完成！")

    # ========================================
    # 4. 保存模型
    # ========================================
    model_dir = Path("Model_Training_Testing/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    stem = selected_file.stem
    model_path = model_dir / f"{stem}_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"[*] 模型已保存至: {model_path}")


if __name__ == "__main__":
    main()
