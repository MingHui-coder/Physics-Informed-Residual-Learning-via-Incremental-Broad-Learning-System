"""
final_moment.py

描述:   交互选择 BLS 测试结果 CSV，自动匹配 physics-based 分量，融合得
       到最终力矩预测，计算 RMSE / R²，导出 CSV，可选绘制折线图。
作者:   Minghui Zhang
创建:   2026-04
"""

import sys
import csv
from pathlib import Path

import numpy as np


def list_csv_files(data_dir):
    """扫描目录下所有 .csv 文件，返回排序后的列表。"""
    return sorted(Path(data_dir).glob("*.csv"))


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


def load_csv(path):
    """加载 CSV 文件，返回 (headers, rows)，rows 为列表的列表。"""
    with open(path, "r", newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


# 数据集 → 测试受试者映射（与 data_utils_final.py 保持一致）
TEST_SUBJECT_MAP = {
    "datasetA": "AB25",
    "datasetB": "AB02",
}


def parse_test_result_stem(test_csv_stem):
    """从 test_result 文件名解析 dataset / action / trial_token，以及是否filtered。"""
    stem = test_csv_stem
    if stem.startswith("result_"):
        stem = stem[7:]

    # 检测是否filtered
    is_filtered = "filtered" in stem.lower()

    test_part = stem.split("__Train_")[0]
    tokens = test_part.split("__")

    if len(tokens) < 3:
        return None, None, None, is_filtered

    # dataset可能是 "filtered_datasetA" 或 "datasetA"，需要去掉 "filtered_" 前缀
    dataset = tokens[0].replace("Test_", "", 1)
    if dataset.startswith("filtered_"):
        dataset = dataset.replace("filtered_", "")

    action = tokens[1]
    trial_token = tokens[-1]
    return dataset, action, trial_token, is_filtered


def find_csv_by_trial(search_base, trial_token):
    """在目录树中按试次名（大小写不敏感）查找 CSV 文件。"""
    if not search_base.exists():
        return None
    trial_lower = trial_token.lower()
    for f in sorted(search_base.rglob("*.csv")):
        if f.stem.lower() == trial_lower:
            return f
    return None


def load_t_physics(csv_path):
    """从 CSV 中读取 T_physics 列，返回 numpy 数组。"""
    headers, rows = load_csv(csv_path)
    try:
        idx = headers.index("T_physics")
    except ValueError:
        return None
    return np.array([float(row[idx]) for row in rows])


def sliding_window_align(values, seq_len=20):
    """
    对 T_physics 施加与测试集张量生成时相同的滑动窗口对齐。
    label = targets[i + seq_len - 1]，即取窗口最后一个值。
    """
    if len(values) < seq_len:
        return None
    return values[seq_len - 1:]


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    # ========================================
    # 1. 交互选择 test_result CSV
    # ========================================
    test_result_dir = Path("Model_Training_Testing/test_result")
    if not test_result_dir.exists():
        print(f"[!] 目录不存在: {test_result_dir}")
        return

    csv_files = list_csv_files(test_result_dir)
    if not csv_files:
        print(f"[!] '{test_result_dir}' 下没有找到任何 .csv 文件。")
        return

    selected = select_file_interactive(csv_files, "请选择 BLS 测试结果 CSV 文件：")
    print(f"\n[*] 你选择了: {selected.name}")

    # ========================================
    # 2. 自动匹配 CSV 并提取 T_physics（滑动窗口对齐）
    # ========================================
    dataset, action, trial_token, is_filtered = parse_test_result_stem(selected.stem)
    if dataset is None:
        print("[!] 无法解析 test_result 文件名。")
        return

    test_subject = TEST_SUBJECT_MAP.get(dataset)
    print(f"[*] 解析结果: dataset={dataset}, action={action}, trial={trial_token}, filtered={is_filtered}")
    print(f"[*] 测试受试者: {test_subject}")

    # 读取 test_result 的 Truth / Predicted
    test_headers, test_rows = load_csv(selected)
    try:
        truth_idx = test_headers.index("Truth")
        pred_idx = test_headers.index("Predicted")
    except ValueError:
        print("[!] test_result CSV 缺少 Truth 或 Predicted 列。")
        return
    n_test = len(test_rows)
    truth_vals = np.array([float(test_rows[i][truth_idx]) for i in range(n_test)])
    pred_vals = np.array([float(test_rows[i][pred_idx]) for i in range(n_test)])

    # 根据是否filtered决定搜索路径
    if is_filtered:
        base_dir = Path("dataset_porecessing/filter_final_dataset")
    else:
        base_dir = Path("dataset_porecessing/final_dataset")

    # 自动查找 T_physics 来源 CSV
    t_physics = None

    search_candidates = [
        ("final_dataset_test", Path("dataset_porecessing/final_dataset_test") / dataset / action),
        (f"{base_dir.name}/{test_subject}", base_dir / dataset / action / test_subject),
        (base_dir.name, base_dir / dataset / action),
    ]

    for label, search_base in search_candidates:
        csv_file = find_csv_by_trial(search_base, trial_token)
        if csv_file is None:
            continue
        raw = load_t_physics(csv_file)
        if raw is None:
            continue
        aligned = sliding_window_align(raw, seq_len=20)
        if aligned is None:
            print(f"[!] '{csv_file}' 数据不足 {20} 个样本，跳过。")
            continue
        if len(aligned) != n_test:
            print(f"[!] '{label}' 下窗口对齐后 {len(aligned)} 行，"
                  f"与 test_result {n_test} 行不匹配，继续搜索...")
            continue
        t_physics = aligned
        print(f"[*] 匹配成功: {csv_file}  (滑动窗口对齐后 {len(aligned)} 行)")
        break

    if t_physics is None:
        print(f"[!] 未找到行数匹配的 T_physics 源文件。"
              f"test_result 共 {n_test} 行（seq_len=20 滑动窗口）。")
        return

    # ========================================
    # 3. 融合计算
    # ========================================
    print("[*] 正在融合计算最终力矩...")

    final_truth = truth_vals + t_physics
    final_pred = pred_vals + t_physics

    # ========================================
    # 5. 评估指标
    # ========================================
    rmse = compute_rmse(final_truth, final_pred)
    r2 = compute_r2(final_truth, final_pred)

    print(f"\n[*] 融合后评估结果:")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    R²   = {r2:.6f}")

    # ========================================
    # 6. 保存 CSV
    # ========================================
    out_dir = Path("final_moment_prediction/reult")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"final_{selected.stem}.csv"
    out_path = out_dir / out_name

    with open(out_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Final_Truth", "Final_Predicted"])
        for t, p in zip(final_truth, final_pred):
            writer.writerow([t, p])

    print(f"[*] 结果已保存至: {out_path}")

    # ========================================
    # 7. 询问是否画图
    # ========================================
    while True:
        choice = input("\n[*] 是否绘制最终力矩的对比折线图？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                plt.plot(final_truth, label='Final Truth  (w/ T_physics)', alpha=0.8, linewidth=1)
                plt.plot(final_pred, label='Final Predicted  (w/ T_physics)', alpha=0.8, linewidth=1)
                plt.xlabel('Sample')
                plt.ylabel('Knee Moment')
                plt.title(f'Final Moment Prediction  (RMSE={rmse:.6f}, R²={r2:.6f})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                plot_path = out_dir / f"plot_{selected.stem}.png"
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

    print("\n[*] 处理完成！")


if __name__ == "__main__":
    main()
