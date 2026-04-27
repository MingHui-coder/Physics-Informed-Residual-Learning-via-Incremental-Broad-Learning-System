"""
generate_train_final.py

描述:   基于 final_dataset 的训练集生成脚本，调用 data_utils_final 为指定数据集
       和动作生成训练集张量。预测目标通过终端交互选择。
作者:   Minghui Zhang
创建:   2026-04
"""

from data_utils_final import generate_training_tensor, select_target_column_interactive, select_action_interactive


def select_dataset_type():
    """交互选择数据集类型：filter还是unfilter"""
    print("\n" + "="*50)
    print("[*] 请选择数据集类型：")
    print("    1 - 未滤波 (使用 dataset_porecessing/final_dataset)")
    print("    2 - 滤波后 (使用 dataset_porecessing/filter_final_dataset)")
    print("="*50)
    while True:
        try:
            choice = input("\n[*] 请输入编号 (1/2): ").strip()
            if choice == "1":
                return "unfiltered", "dataset_porecessing/final_dataset"
            elif choice == "2":
                return "filtered", "dataset_porecessing/filter_final_dataset"
            else:
                print("[!] 输入无效，请输入 1 或 2。")
        except Exception:
            print("[!] 输入无效，请重新输入。")


if __name__ == "__main__":
    # ========================================
    # 业务参数配置
    # ========================================
    TARGET_DATASET = "datasetA"          # 可选: "datasetA" 或 "datasetB"

    # 终端交互选择数据集类型
    DATASET_TYPE, BASE_DIR = select_dataset_type()

    # 终端交互选择动作和目标列
    TARGET_ACTION = select_action_interactive()
    TARGET_COLUMN = select_target_column_interactive()

    # 基于选择的数据集生成训练张量
    generate_training_tensor(
        dataset_name=TARGET_DATASET,
        action=TARGET_ACTION,
        target_column=TARGET_COLUMN,
        seq_len=20,
        base_dir=BASE_DIR,
        dataset_type=DATASET_TYPE
    )
