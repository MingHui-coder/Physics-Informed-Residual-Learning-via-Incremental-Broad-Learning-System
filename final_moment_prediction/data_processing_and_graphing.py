"""
data_processing_and_graphing.py

描述:   交互选择 CSV 文件（test_result 或 final_moment result），
       可选平滑滤波 Predicted 数据，计算 RMSE / R²，可选画图。
作者:   Minghui Zhang
创建:   2026-04
"""

import csv
from pathlib import Path

import numpy as np


def list_csv_files(data_dir):
    return sorted(Path(data_dir).glob("*.csv"))


def select_file_interactive(files, title):
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
    with open(path, "r", newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader])
    return headers, rows


def smooth_savgol(y, window):
    """Savitzky-Golay 滤波。"""
    from scipy.signal import savgol_filter
    order = min(2, window - 2)  # 低阶 = 更强平滑
    return savgol_filter(y, window, order)


def smooth_gaussian(y, sigma):
    """高斯滤波。"""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(y, sigma, mode='reflect')


def smooth_butterworth(y, cutoff, fs=1.0, order=4):
    """零相位 Butterworth 低通滤波。"""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, y)


def smooth_median(y, window):
    """中值滤波：专门去除毛刺/脉冲噪声。"""
    from scipy.signal import medfilt
    return medfilt(y, kernel_size=window)


def smooth_cascade(y, filters):
    """级联滤波：按顺序执行多个滤波步骤。"""
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


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def run_smoothing_once(y_pred, label_prefix=""):
    """执行一次交互式平滑配置，返回 (smoothed, label_suffix)。"""
    has_scipy = True
    try:
        import scipy.signal  # noqa: F401
        import scipy.ndimage  # noqa: F401
    except ImportError:
        has_scipy = False

    if has_scipy:
        print(f"\n{label_prefix}[*] 请选择平滑方案：")
        print(f"    {label_prefix}[1] 中值 + SavGol    — 先中值去毛刺，再 SG 平滑（推荐）")
        print(f"    {label_prefix}[2] 中值 + Gaussian   — 先中值去毛刺，再高斯平滑")
        print(f"    {label_prefix}[3] Butterworth 低通  — 零相位低通滤波")
        print(f"    {label_prefix}[4] 仅中值滤波        — 只去除毛刺")
        print(f"    {label_prefix}[5] 仅 Savitzky-Golay — 只做 SG 平滑")
        methods = {"1": "cascade_sg", "2": "cascade_gau",
                   "3": "but", "4": "median_only", "5": "sg_only"}
        while True:
            m = input(f"{label_prefix}[*] 请输入编号 (1-5): ").strip()
            if m in methods:
                method = methods[m]
                break
            print(f"{label_prefix}[!] 请输入 1-5。")

        if method == "cascade_sg":
            while True:
                try:
                    mw = int(input(f"{label_prefix}[*] 中值窗口 (奇数, 建议 5-15): ").strip())
                    if mw >= 3 and mw % 2 == 1:
                        break
                    print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入整数。")
            while True:
                try:
                    sw = int(input(f"{label_prefix}[*] SG 窗口 (奇数, 建议 21-101): ").strip())
                    if sw >= 3 and sw % 2 == 1:
                        break
                    print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入整数。")
            result = smooth_cascade(y_pred, [("median", {"window": mw}), ("savgol", {"window": sw})])
            label = f"Median{mw}+SG{sw}"

        elif method == "cascade_gau":
            while True:
                try:
                    mw = int(input(f"{label_prefix}[*] 中值窗口 (奇数, 建议 5-15): ").strip())
                    if mw >= 3 and mw % 2 == 1:
                        break
                    print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入整数。")
            while True:
                try:
                    s = float(input(f"{label_prefix}[*] Gaussian sigma (越大越平滑, 建议 3-15): ").strip())
                    if s > 0:
                        break
                    print(f"{label_prefix}[!] sigma 必须 >0。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入数字。")
            result = smooth_cascade(y_pred, [("median", {"window": mw}), ("gaussian", {"sigma": s})])
            label = f"Median{mw}+Gaussσ{s}"

        elif method == "but":
            while True:
                try:
                    c = float(input(f"{label_prefix}[*] 截止频率/采样率 (0-1, 越小越平滑, 建议 0.02-0.1): ").strip())
                    if 0 < c < 1:
                        break
                    print(f"{label_prefix}[!] 请输入 0-1 之间的值。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入数字。")
            result = smooth_butterworth(y_pred, c)
            label = f"Butter fc={c}"

        elif method == "median_only":
            while True:
                try:
                    mw = int(input(f"{label_prefix}[*] 中值窗口 (奇数, 建议 5-21): ").strip())
                    if mw >= 3 and mw % 2 == 1:
                        break
                    print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入整数。")
            result = smooth_median(y_pred, mw)
            label = f"Median w={mw}"

        else:  # sg_only
            while True:
                try:
                    sw = int(input(f"{label_prefix}[*] SG 窗口 (奇数, 建议 21-101): ").strip())
                    if sw >= 3 and sw % 2 == 1:
                        break
                    print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
                except ValueError:
                    print(f"{label_prefix}[!] 请输入整数。")
            result = smooth_savgol(y_pred, sw)
            label = f"SavGol w={sw}"
    else:
        # 无 scipy 回退：中值 + 5 次移动平均
        while True:
            try:
                mw = int(input(f"{label_prefix}[*] 中值窗口 (奇数, 建议 5-15): ").strip())
                if mw >= 3 and mw % 2 == 1:
                    break
                print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
            except ValueError:
                print(f"{label_prefix}[!] 请输入整数。")
        while True:
            try:
                sw = int(input(f"{label_prefix}[*] 移动平均窗口 (奇数, 建议 21-101): ").strip())
                if sw >= 3 and sw % 2 == 1:
                    break
                print(f"{label_prefix}[!] 必须为 ≥3 的奇数。")
            except ValueError:
                print(f"{label_prefix}[!] 请输入整数。")
        from scipy.signal import medfilt
        result = medfilt(y_pred, kernel_size=mw)
        half = sw // 2
        for _ in range(5):
            nxt = np.copy(result)
            for i in range(half, len(result) - half):
                nxt[i] = np.mean(result[i - half:i + half + 1])
            result = nxt
        label = f"Median{mw}+MA×5 w={sw}"

    return result, label


def main():
    # ========================================
    # 1. 选择数据来源目录
    # ========================================
    dirs = {
        "1": ("test_result 目录 (Truth / Predicted / 残差)",
              Path("Model_Training_Testing/test_result")),
        "2": ("final_moment reult 目录 (Final_Truth / Final_Predicted / 完整力矩)",
              Path("final_moment_prediction/reult")),
    }

    print("\n" + "=" * 50)
    print("[*] 请选择数据来源：")
    for k, (desc, _) in dirs.items():
        print(f"    [{k}] {desc}")
    print("=" * 50)
    while True:
        choice = input("\n[*] 请输入编号 (1 或 2): ").strip()
        if choice in dirs:
            _, src_dir = dirs[choice]
            break
        print("[!] 请输入 1 或 2。")

    if not src_dir.exists():
        print(f"[!] 目录不存在: {src_dir}")
        return

    csv_files = list_csv_files(src_dir)
    if not csv_files:
        print(f"[!] '{src_dir}' 下没有 .csv 文件。")
        return

    # ========================================
    # 2. 选择 CSV 文件
    # ========================================
    selected = select_file_interactive(csv_files, "请选择 CSV 文件：")
    print(f"\n[*] 你选择了: {selected.name}")

    # ========================================
    # 3. 加载数据
    # ========================================
    print("[*] 正在加载数据...")
    headers, data = load_csv(selected)

    # 自动识别列名（兼容 test_result 和 reult）
    col_map = {}
    for h in headers:
        hl = h.lower()
        if hl in ("truth", "final_truth"):
            col_map["truth"] = h
        elif hl in ("predicted", "final_predicted"):
            col_map["predicted"] = h

    if "truth" not in col_map or "predicted" not in col_map:
        print(f"[!] 无法识别 Truth / Predicted 列。当前列: {headers}")
        return

    truth_idx = headers.index(col_map["truth"])
    pred_idx = headers.index(col_map["predicted"])
    y_true = data[:, truth_idx]
    y_pred = data[:, pred_idx]

    print(f"[*] 数据行数: {len(y_true)}, 列: {headers}")

    # ========================================
    # 4. 可选平滑滤波（支持 1-2 条曲线）
    # ========================================
    while True:
        choice = input("\n[*] 是否对 Predicted 数据进行平滑滤波？(y/n): ").strip().lower()
        if choice == 'y':
            while True:
                try:
                    n_curves = int(input("[*] 生成几条平滑曲线进行对比？（1 或 2）: ").strip())
                    if n_curves in (1, 2):
                        break
                    print("[!] 请输入 1 或 2。")
                except ValueError:
                    print("[!] 请输入数字。")

            preds = []  # [(data, label), ...]
            for i in range(n_curves):
                prefix = f"[曲线 {i + 1}/{n_curves}] "
                print(f"\n{'─' * 40}")
                print(f"{prefix}配置平滑参数")
                smoothed, label = run_smoothing_once(y_pred, prefix)
                preds.append((smoothed, label))
            break
        elif choice == 'n':
            preds = [(y_pred, "Raw")]
            break
        else:
            print("[!] 请输入 y 或 n。")

    # ========================================
    # 5. 计算指标（逐条输出）
    # ========================================
    print(f"\n{'=' * 50}")
    print("[*] 评估结果：")
    for data, label in preds:
        rmse = compute_rmse(y_true, data)
        r2 = compute_r2(y_true, data)
        print(f"    {label:30s}  RMSE = {rmse:.6f}   R² = {r2:.6f}")
    print(f"{'=' * 50}")

    # ========================================
    # 6. 可选画图
    # ========================================
    while True:
        choice = input("\n[*] 是否绘制折线图？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 5))
                plt.plot(y_true, label='Truth', alpha=0.7, linewidth=1, color='black')

                colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
                linestyles = ['-', '--', '-.', ':']
                for idx, (data, label) in enumerate(preds):
                    c = colors[idx % len(colors)]
                    ls = linestyles[idx % len(linestyles)]
                    rmse = compute_rmse(y_true, data)
                    r2 = compute_r2(y_true, data)
                    plt.plot(data, label=f'{label}  (RMSE={rmse:.4f}, R²={r2:.4f})',
                             alpha=0.8, linewidth=1, color=c, linestyle=ls)

                plt.xlabel('Sample')
                plt.ylabel('Value')
                plt.title('Prediction Comparison')
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
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
