"""
generate_test_final.py

描述:   基于 final_dataset 的测试集生成脚本，调用 data_utils_final 为指定数据集
       和动作生成测试集张量。预测目标通过终端交互选择。
作者:   Minghui Zhang
创建:   2026-04
"""

from data_utils_final import generate_test_tensor, select_target_column_interactive, select_action_interactive

if __name__ == "__main__":
    # ========================================
    # 业务参数配置
    # ========================================
    TARGET_DATASET = "datasetA"          # 可选: "datasetA" 或 "datasetB"

    # 终端交互选择动作和目标列
    TARGET_ACTION = select_action_interactive()
    TARGET_COLUMN = select_target_column_interactive()

    # 启动交互式的测试集生成流程
    generate_test_tensor(
        dataset_name=TARGET_DATASET,
        action=TARGET_ACTION,
        target_column=TARGET_COLUMN,
        seq_len=20,
        base_dir="dataset_porecessing/final_dataset"
    )
