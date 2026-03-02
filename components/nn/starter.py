from typing import Union, Optional, List, Set
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
from .dataProcess import dataProcess
from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .configure import Configure
import random
from sklearn.metrics import average_precision_score  # 仅新增：AUC-PR依赖

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"✅ 所有随机种子已固定为：{seed}")

# 仅新增：极简Recall@TopK计算
def recall_at_top_k(pred_com, true_com, k_percent=1.0):
    if not pred_com or not true_com:
        return 0.0
    top_k = max(1, int(len(pred_com) * k_percent / 100))
    hit = len(set(pred_com[:top_k]) & set(true_com))
    return hit / len(true_com) if len(true_com) > 0 else 0.0

# 核心修改：仅补充指标收集和计算，其余逻辑不变
def eval_model(
    expander: Expander, test_g: Graph, test_seeds: List, conf: Configure
) -> dict:
    expander.model.eval()
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "auc_pr": [],  # 新增：AUC-PR
        "recall_top1": [],  # 新增：Recall@Top1%
        "recall_top5": []   # 新增：Recall@Top5%
    }

    with torch.no_grad():
        pred_coms, _ = expander.sample_bs_trajectories(test_seeds)

        for idx, seed in enumerate(test_seeds):
            true_com = test_g.sampleTrajectory(seed, traj_length=conf.maxTraLen)
            pred_com = [node for node in pred_coms[idx] if node != "Stp"]
            p, r, f1 = expander.eval_scores(pred_com, true_com)

            # 仅新增：计算AUC-PR（复用Expander的逻辑）
            ap = expander.eval_auc_pr(pred_com, set(true_com)) if hasattr(expander, 'eval_auc_pr') else 0.0
            # 仅新增：计算Recall@TopK
            r1 = recall_at_top_k(pred_com, true_com, 1.0)
            r5 = recall_at_top_k(pred_com, true_com, 5.0)

            # 仅新增：收集新指标
            all_metrics["auc_pr"].append(ap)
            all_metrics["recall_top1"].append(r1)
            all_metrics["recall_top5"].append(r5)
            all_metrics["precision"].append(p)
            all_metrics["recall"].append(r)
            all_metrics["f1"].append(f1)

    # 仅新增：新指标的均值/标准差
    avg_metrics = {
        "avg_precision": round(np.mean(all_metrics["precision"]), 4),
        "avg_recall": round(np.mean(all_metrics["recall"]), 4),
        "avg_f1": round(np.mean(all_metrics["f1"]), 4),
        "std_f1": round(np.std(all_metrics["f1"]), 4),
        "avg_auc_pr": round(np.mean(all_metrics["auc_pr"]), 4),  # 新增
        "avg_recall_top1": round(np.mean(all_metrics["recall_top1"]), 4),  # 新增
        "avg_recall_top5": round(np.mean(all_metrics["recall_top5"]), 4)   # 新增
    }

    # 仅新增：打印新指标
    print("\n" + "=" * 60)
    print("📊 模型测试集评估结果")
    print("=" * 60)
    print(f"测试种子数量：{len(test_seeds)}")
    print(f"平均精度(P)：{avg_metrics['avg_precision']}")
    print(f"平均召回(R)：{avg_metrics['avg_recall']}")
    print(f"平均F1分数：{avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})")
    print(f"平均AUC-PR：{avg_metrics['avg_auc_pr']}")  # 新增
    print(f"平均Recall@Top1%：{avg_metrics['avg_recall_top1']}")  # 新增
    print(f"平均Recall@Top5%：{avg_metrics['avg_recall_top5']}")  # 新增
    print("=" * 60)

    return avg_metrics

def run(dfname, conf: Configure, seed: int = 42):
    set_seed(seed)
    try:
        dp = dataProcess(
            dfname=conf.dfname,
            normal_node_ratio=conf.normal_node_ratio,
            expand_hop=conf.expand_hop,
            min_community_size=conf.min_community_size,
        )
        print("✅ dataProcess初始化成功")

        print("tes_n", len(dp.test_nodes))
        print("tes_h", len(dp.test_hacker))
        print("tra_n", len(dp.train_nodes))
        print("tra_h", len(dp.train_hacker))

        g = Graph(
            dfnode=dp.train_nodes,
            dffeature=dp.train_feature,
            dfhacker=dp.train_hacker,
            dfedge=dp.train_edge,
        )
        print(f"✅ Graph类初始化成功（基于{dfname}训练集）")

        device = torch.device(conf.device)
        model = Agent(input_size=g.embedsize, hidden_size=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        expander = Expander(
            graph=g,
            model=model,
            optimizer=optimizer,
            device=device,
            maxLen=conf.maxTraLen,
            gamma=conf.gamma,
            reward_weight_abs=conf.reward_weight_abs,
            reward_weight_delta=conf.reward_weight_delta,
            len_penalty_base=conf.len_penalty_base,

        )
        print("✅ Expander初始化成功")

        origin_seeds = dp.train_hacker["address"].values.tolist()
        for i in range(conf.epoch):
            seeds = random.sample(origin_seeds, k=conf.seedNum)
            true_coms = [
                g.sampleTrajectory(s, traj_length=conf.maxTraLen) for s in seeds
            ]

            print(f"\n===================== 第{i}次训练结果 =====================")
            loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
            print(f"✅ trainReward执行成功")
            print(f"   本次训练损失值：{loss:.4f}")

        print("\n🔍 开始在测试集上评估模型...")
        test_g = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge,
        )
        expander.graph = test_g
        test_seeds = dp.test_hacker["address"].values.tolist()
        test_seeds = test_seeds[: min(conf.seedNum * 2, len(test_seeds))]
        test_metrics = eval_model(expander, test_g, test_seeds, conf)

        return test_metrics

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return None