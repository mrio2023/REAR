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

def eval_model(
    expander: Expander, test_g: Graph, test_seeds: List, conf: Configure
) -> dict:
    expander.model.eval()
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": []
    }

    print(f"\n🔧 测试参数：")
    print(f"   种子数：{len(test_seeds)}")
    print(f"   maxTraLen：{conf.maxTraLen}")
    print(f"   device：{conf.device}")
    print("-" * 60)

    with torch.no_grad():
        pred_coms, _ = expander.sample_bs_trajectories(test_seeds)

        batch_size = 10
        for idx, seed in enumerate(test_seeds):
            true_com = test_g.sampleTrajectory(seed, traj_length=conf.maxTraLen)
            pred_com = [node for node in pred_coms[idx] if node != "Stp"]
            p, r, f1 = expander.eval_scores(pred_com, true_com)

            all_metrics["precision"].append(p)
            all_metrics["recall"].append(r)
            all_metrics["f1"].append(f1)

            if (idx + 1) % batch_size == 0 or idx == len(test_seeds) - 1:
                print(f"📈 进度：{idx+1}/{len(test_seeds)}")
                print(f"   P: {np.mean(all_metrics['precision'][-batch_size:]):.4f}")
                print(f"   R: {np.mean(all_metrics['recall'][-batch_size:]):.4f}")
                print(f"   F1: {np.mean(all_metrics['f1'][-batch_size:]):.4f}")
                print("-" * 40)

    avg_metrics = {
        "avg_precision": round(np.mean(all_metrics["precision"]), 4),
        "avg_recall": round(np.mean(all_metrics["recall"]), 4),
        "avg_f1": round(np.mean(all_metrics["f1"]), 4),
        "std_f1": round(np.std(all_metrics["f1"]), 4)
    }

    print("\n" + "=" * 60)
    print("📊 测试集最终结果")
    print("=" * 60)
    print(f"P: {avg_metrics['avg_precision']}")
    print(f"R: {avg_metrics['avg_recall']}")
    print(f"F1: {avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})")
    print("=" * 60)

    return avg_metrics

def run(dfname, conf: Configure, seed: int = 42):
    set_seed(seed)
    # 新增：给Configure补充epoch参数（避免KeyError）
    if not hasattr(conf, 'epoch'):
        conf.epoch = 30  # 默认30轮训练，和你的日志一致
    
    try:
        print(f"\n{'='*70}")
        print(f"📌 数据集：{dfname}  seed={seed}")
        print(f"{'='*70}")
        print(f"参数：")
        print(f"  normal_node_ratio   {conf.normal_node_ratio}")
        print(f"  expand_hop          {conf.expand_hop}")
        print(f"  min_community_size  {conf.min_community_size}")
        print(f"  seedNum             {conf.seedNum}")
        print(f"  maxTraLen           {conf.maxTraLen}")
        print(f"  gamma               {conf.gamma}")
        print(f"  f1_base_weight      {conf.f1_base_weight}")
        print(f"  p_bias              {conf.p_bias}")
        print(f"  min_f1_threshold    {conf.min_f1_threshold}")
        print(f"  len_penalty_coeff   {conf.len_penalty_coeff}")
        print(f"  epoch               {conf.epoch}")
        print("-" * 70)

        dp = dataProcess(
            dfname=conf.dfname,
            normal_node_ratio=conf.normal_node_ratio,
            expand_hop=conf.expand_hop,
            min_community_size=conf.min_community_size,
        )
        print("✅ dataProcess 完成")

        print(f"\n数据规模：")
        print(f"  训练节点：{len(dp.train_nodes)}")
        print(f"  训练黑客：{len(dp.train_hacker)}")
        print(f"  测试节点：{len(dp.test_nodes)}")
        print(f"  测试黑客：{len(dp.test_hacker)}")
        print("-" * 70)

        g = Graph(
            dfnode=dp.train_nodes,
            dffeature=dp.train_feature,
            dfhacker=dp.train_hacker,
            dfedge=dp.train_edge,
        )
        print("✅ 训练图构建完成")

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
            f1_base_weight=conf.f1_base_weight,
            p_bias=conf.p_bias,
            min_f1_threshold=conf.min_f1_threshold,
            len_penalty_coeff=conf.len_penalty_coeff
        )
        print("✅ Expander 初始化完成")

        origin_seeds = dp.train_hacker["address"].values.tolist()
        print(f"\n🚀 开始训练，每轮采样 {conf.seedNum} 个种子，共 {conf.epoch} 轮")

        for i in range(conf.epoch):
            seeds = random.sample(origin_seeds, k=conf.seedNum)
            true_coms = [
                g.sampleTrajectory(s, traj_length=conf.maxTraLen) for s in seeds
            ]
            print(f"\n📝 Epoch {i}")
            loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
            print(f"loss: {loss:.4f}")

        print(f"\n{'='*70}")
        print("🧪 开始测试")
        print(f"{'='*70}")

        test_g = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge,
        )
        expander.graph = test_g
        test_seeds = dp.test_hacker["address"].values.tolist()
        test_seeds = test_seeds[: min(conf.seedNum * 2, len(test_seeds))]
        print(f"测试种子数：{len(test_seeds)}")

        test_metrics = eval_model(expander, test_g, test_seeds, conf)
        
        # 修复：只打印存在的指标，删除AUC-PR/TopK相关
        print(f"\n✅ 数据集 {dfname} 训练完成！")
        print(f"   最终测试集F1：{test_metrics['avg_f1']}")
        print(f"   最终测试集P：{test_metrics['avg_precision']}")
        print(f"   最终测试集R：{test_metrics['avg_recall']}")

        return test_metrics

    except Exception as e:
        print(f"\n❌ 运行失败：{e}")
        import traceback
        traceback.print_exc()
        return None
