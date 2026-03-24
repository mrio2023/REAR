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
    "ibm_h_small": {
        # 基础参数
        "dfname": "ibm_h_small",
        "normal_node_ratio": 3,
        "expand_hop": 2,
        "min_community_size": 5,
        "p_bias": 0.2,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.8,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 70,
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 12,  # 可能比small稍大
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
        "p_bias": 1.0,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.5,
        "stop_reward_scale": 2.0,
        # 训练层面
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 60,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 12,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "ibm_h_medium": {
        "dfname": "ibm_h_medium",
        "normal_node_ratio": 3,
        "expand_hop": 2,
        "min_community_size": 5,
        "p_bias": 0.8,
        "r_bias": 1.2,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.8,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 70,
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 13,  # 可能比small稍大
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "ibm_l_medium": {
        "dfname": "ibm_l_medium",
        "normal_node_ratio": 3,
        "expand_hop": 2,
        "min_community_size": 5,
        "p_bias": 1.0,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 1.0,
        "target_recall": 0.8,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 70,
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 12,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "dgraph": {
        "dfname": "dgraph",
        "normal_node_ratio": 5,  # 正常节点比例可能更高
        "expand_hop": 2,  # 控制子图大小
        "min_community_size": 5,  # 最小社区规模
        "p_bias": 1.0,
        "r_bias": 1.0,
        "repeat_penalty_coeff": 0.5,
        "entropy_coeff": 0.01,
        "grad_norm": 10.0,
        "target_recall": 0.5,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 60,  # 大图训练适当减少或保持
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 9,  # 可能需更长路径
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
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
    # dataset_list = [ "elliptic","dgraph","ibm_l_medium","ibm_h_medium","ibm_h_small"]
    dataset_list = [ "dgraph"]
    final_results = train_all_datasets(dataset_list, seed=2026)
