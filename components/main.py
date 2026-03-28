import torch
import os
from nn.starter import Starter
from nn.tool import set_seed


DATASET_CONFIGS = {
    "ibm_h_small": {
        "min_community_size": 5,
        "p_bias": 0.2,
        "r_bias": 1.0,
        "grad_norm": 1.0,
        "target_f1": 0.8,
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
    "elliptic": {
        "min_community_size": 2,
        # 奖励层面
        "p_bias": 1.0,
        "r_bias": 1.0,
        "grad_norm": 1.0,
        "target_f1": 0.5,
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
        "min_community_size": 5,
        "p_bias": 0.8,
        "r_bias": 1.2,
        "grad_norm": 1.0,
        "target_f1": 0.8,
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
        "min_community_size": 5,
        "p_bias": 1.0,
        "r_bias": 1.0,
        "grad_norm": 1.0,
        "target_f1": 0.5,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 60,
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 12,
        "hidden_size": 128,
        "device":  "cpu",
    },
    "dgraph": {
        "min_community_size": 3,
        "p_bias": 1.0,
        "r_bias": 1.3,
        "grad_norm": 10.0,
        "target_f1": 0.4,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 70,  # 大图训练适当减少或保持
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 10,  # 可能需更长路径
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def train_single_dataset(dfname: str, seed: int = 2026):
    print(f"开始训练数据集：{dfname} (种子={seed})")
    set_seed(seed=seed)
    print("种子是", seed)
    if dfname not in DATASET_CONFIGS:
        print(f"数据集 {dfname} 无配置参数，终止训练")
        return None
    params = DATASET_CONFIGS[dfname]
    params["dfname"] = dfname
    starter = Starter(params=params)
    try:
        starter.run(dfname=dfname, seed=seed)
    except Exception as e:
        print(f"\n❌ 数据集 {dfname} 运行失败：{str(e)}")
        import traceback

        traceback.print_exc()


def train_all_datasets(datasets: list, seed: int = 2026):
    for dfname in datasets:
        train_single_dataset(dfname, seed=seed)


if __name__ == "__main__":
    # 可调整训练的数据集列表
    # dataset_list = [ "elliptic","dgraph","ibm_l_medium","ibm_h_medium","ibm_h_small"]
    dataset_list = ["dgraph"]
    train_all_datasets(dataset_list, seed=2026)
