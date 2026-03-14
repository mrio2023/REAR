import torch
import os
from nn.starter import Starter  # 使用已适配纯字典传参的Starter
from nn.tool import set_seed

# 设置工作目录
current_file = os.path.abspath(__file__)
sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
os.chdir(sci_dir)


# 数据集参数配置表（所有参数集中管理，已适配新版Expander）
DATASET_CONFIGS = {
    "ibm": {
        # 基础参数
        "dfname": "ibm",
        # 数据层面：减少无关节点干扰
        "normal_node_ratio": 3,
        "expand_hop": 2,
        "min_community_size": 7,
        # 奖励层面（新参数）
        "p_bias": 0,
        "r_bias": 1.0,                     # 召回倾斜系数
        "repeat_penalty_coeff": 0.5,        # 重复选点惩罚
        "entropy_coeff": 0.01,              # 熵正则系数
        "grad_norm": 1.0,                   # 梯度裁剪阈值
        "target_recall": 0.8,               # 目标召回率
        "stop_reward_scale": 2.0,            # 停止奖励缩放因子
        # 训练层面
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 50,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 10,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "elliptic": {
        # 基础参数
        "dfname": "elliptic",
        # 数据层面
        "normal_node_ratio": 2,
        "expand_hop": 2,
        "min_community_size": 2,
        # 奖励层面
        "p_bias": 0,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.8,
        "stop_reward_scale": 2.0,
        # 训练层面
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 40,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 40,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "elliptic2_full": {
        # 基础参数
        "dfname": "elliptic2_full",
        # 数据层面
        "normal_node_ratio": 2,
        "expand_hop": 2,
        "min_community_size": 5,
        # 奖励层面
        "p_bias": 0.3,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.8,
        "stop_reward_scale": 2.0,
        # 训练层面
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 30,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 8,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def train_single_dataset(dfname: str, seed: int = 2026):
    """训练单个数据集，返回测试指标（纯字典传参）"""
    print(f"\n{'='*70}\n🚀 开始训练数据集：{dfname} (种子={seed})\n{'='*70}")
    set_seed(seed=seed)

    print("种子是", seed)
    # 检查配置是否存在
    if dfname not in DATASET_CONFIGS:
        print(f"❌ 数据集 {dfname} 无配置参数，终止训练")
        return None

    # 直接获取参数字典
    params = DATASET_CONFIGS[dfname]

    # 初始化Starter（直接传参数字典）
    starter = Starter(params=params)
    try:
        # 调用run方法（参数完全解耦）
        test_metrics = starter.run(dfname=dfname, seed=seed)
        return test_metrics
    except Exception as e:
        print(f"\n❌ 数据集 {dfname} 运行失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_all_datasets(datasets: list, seed: int = 2026):
    """批量训练多个数据集，返回汇总结果"""
    all_results = {}
    for dfname in datasets:
        all_results[dfname] = train_single_dataset(dfname, seed=seed)
    return all_results


if __name__ == "__main__":
    # 可调整训练的数据集列表
    dataset_list = ["elliptic"]
    final_results = train_all_datasets(dataset_list, seed=2026)