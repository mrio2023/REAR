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
        "maxLen": 13,
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
        "device": "cpu",
    },
    "dgraph": {
        "min_community_size": 1,
        "p_bias": 1.0,
        "r_bias": 1.0,
        "grad_norm": 10.0,
        "target_f1": 0.6,
        "stop_reward_scale": 2.0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 40,
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 2,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def train_single_dataset(dfname: str, seed: int = 2026):
    print(f"Training dataset: {dfname} (seed={seed})")
    set_seed(seed=seed)
    if dfname not in DATASET_CONFIGS:
        print(f"Dataset {dfname} has no configuration. Aborting.")
        return None
    params = DATASET_CONFIGS[dfname]
    params["dfname"] = dfname
    starter = Starter(params=params)
    try:
        starter.run(dfname=dfname, seed=seed)
    except Exception as e:
        print(f"\nDataset {dfname} failed: {str(e)}")
        import traceback
        traceback.print_exc()


def train_all_datasets(datasets: list, seed: int = 2026):
    for dfname in datasets:
        train_single_dataset(dfname, seed=seed)


if __name__ == "__main__":
    # Adjust the list of datasets to train as needed
    dataset_list= [ "elliptic","dgraph","ibm_l_medium","ibm_h_medium","ibm_h_small"]
    train_all_datasets(dataset_list, seed=2026)