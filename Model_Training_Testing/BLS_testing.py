"""
BLS_testing.py

描述:   终端交互选择测试集张量 (.pt) 和训练好的 BLS 模型 (.pkl)，
       对测试数据进行预测，计算 MSE 和 R²，导出 CSV，可选绘制折线图。
作者:   Minghui Zhang
创建:   2026-04
"""

import sys
import os
import pickle
import csv
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from BLSs_Method.BoradLearningSystem import BLSRegressor


def list_files(data_dir, pattern):
    """扫描目录下所有匹配文件，返回排序后的列表。"""
    p = Path(data_dir)
    return sorted(p.glob(pattern))


def select_file_interactive(files, title):
    """终端交互：让用户选择文件。"""
    print("\n" + "=" * 50)
    print(f"[*] {title}")
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


def compute_r2(y_true, y_pred):
    """计算 R² 决定系数。"""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    # ========================================
    # 1. 交互选择测试集
    # ========================================
    test_dir = Path("tensor_generating/tensors/test")
    if not test_dir.exists():
        print(f"[!] 测试集目录不存在: {test_dir}")
        return

    test_files = list_files(test_dir, "*.pt")
    if not test_files:
        print(f"[!] 目录 '{test_dir}' 下没有找到任何 .pt 文件。")
        return

    selected_test = select_file_interactive(test_files, "请选择测试集文件：")
    print(f"\n[*] 你选择了测试集: {selected_test.name}")

    # ========================================
    # 2. 交互选择模型
    # ========================================
    model_dir = Path("Model_Training_Testing/models")
    if not model_dir.exists():
        print(f"[!] 模型目录不存在: {model_dir}")
        return

    model_files = list_files(model_dir, "*.pkl")
    if not model_files:
        print(f"[!] 目录 '{model_dir}' 下没有找到任何 .pkl 模型文件。")
        return

    selected_model = select_file_interactive(model_files, "请选择模型文件：")
    print(f"\n[*] 你选择了模型: {selected_model.name}")

    # ========================================
    # 3. 加载模型和测试数据
    # ========================================
    print("[*] 正在加载模型...")
    with open(selected_model, "rb") as f:
        model = pickle.load(f)

    # 确保模型在 CPU 上进行测试
    model.cpu()
    print(f"[*] 模型加载完成 (device={model.device})")

    print("[*] 正在加载测试数据...")
    data = torch.load(selected_test, map_location='cpu')
    X_data = data["X_data"]   # shape: (n, seq_len, n_features)
    y_data = data["y_label"]  # shape: (n, 1)

    print(f"[*] 测试数据形状 — X: {X_data.shape}, y: {y_data.shape}")

    # 将 X 从 (n, seq_len, n_features) 展平为 (n, seq_len * n_features)
    n_samples = X_data.shape[0]
    X_flat = X_data.reshape(n_samples, -1)
    print(f"[*] 展平后 X 形状: {X_flat.shape}")

    # ========================================
    # 4. 预测
    # ========================================
    print("[*] 正在进行预测...")
    y_pred = model.predict(X_flat)

    # 转为 numpy 并拉平
    y_true_np = y_data.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()

    # ========================================
    # 5. 计算评估指标
    # ========================================
    mse = np.mean((y_true_np - y_pred_np) ** 2)
    r2 = compute_r2(torch.from_numpy(y_true_np), torch.from_numpy(y_pred_np))

    print(f"\n[*] 评估结果:")
    print(f"    MSE = {mse:.6f}")
    print(f"    R²  = {r2:.6f}")

    # ========================================
    # 6. 保存 CSV
    # ========================================
    result_dir = Path("Model_Training_Testing/test_result")
    result_dir.mkdir(parents=True, exist_ok=True)

    csv_name = f"result_{selected_test.stem}__{selected_model.stem}.csv"
    csv_path = result_dir / csv_name

    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Truth", "Predicted"])
        for t, p in zip(y_true_np, y_pred_np):
            writer.writerow([t, p])

    print(f"[*] 结果已保存至: {csv_path}")

    # ========================================
    # 7. 询问是否画折线图
    # ========================================
    while True:
        choice = input("\n[*] 是否绘制真实值与预测值的折线图？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 5))
                plt.plot(y_true_np, label='Truth', alpha=0.8, linewidth=1)
                plt.plot(y_pred_np, label='Predicted', alpha=0.8, linewidth=1)
                plt.xlabel('Sample')
                plt.ylabel('Value')
                plt.title(f'BLS Prediction Comparison  (MSE={mse:.6f}, R²={r2:.6f})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plot_path = result_dir / f"plot_{selected_test.stem}__{selected_model.stem}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"[*] 折线图已保存至: {plot_path}")

                plt.show()
            except ImportError:
                print("[!] matplotlib 未安装，无法绘图。")
            break
        elif choice == 'n':
            print("[*] 跳过绘图。")
            break
        else:
            print("[!] 请输入 y 或 n。")

    print("\n[*] 测试完成！")


if __name__ == "__main__":
    main()
