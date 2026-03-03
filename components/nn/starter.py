from typing import Union, Optional, List, Set, Dict
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
    expander: Expander, test_g: Graph, conf: Configure
) -> dict:
    """
    最终修正版：完全基于sample_bs_trajectories，无虚构方法
    1. 基于真实社区（test_g.community_seeds）评估，无采样生成社区
    2. 大社区动态采样多个种子（1 + len(data)/maxTraLen），避免单种子偏差
    3. 合并多个种子的扩展路径，去重后作为最终预测社区
    4. 所有评估对比真实社区节点，不与采样社区对比
    """
    expander.model.eval()
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": []
    }

    # 1. 获取测试图中的真实社区（正式数据，无采样）
    true_coms = test_g.community_seeds  # 原始真实社区：[(name_tag, data), ...]
    if not isinstance(true_coms, list):
        # 兼容dict格式：转换为list[(name_tag, data), ...]
        true_coms = list(test_g.community_seeds.items())

    # 2. 按社区大小动态采样种子（核心保留你的逻辑）
    test_seeds = {}
    for name_tag, data in true_coms:
        if len(data) == 0:
            continue
        # 动态计算采样数：1 + 社区大小/最大扩展步数（大社区多采，小社区少采）
        sample_num = int(1 + len(data) / conf.maxTraLen)
        # 确保采样数不超过社区本身大小
        sample_num = min(sample_num, len(data))
        # 随机采样种子
        s = random.sample(data, k=sample_num)
        test_seeds[name_tag] = s

    print(f"\n🔧 测试参数（真实社区+动态采样种子+路径合并）：")
    print(f"   真实社区数：{len(true_coms)}")
    print(f"   采样种子总数：{sum(len(seeds) for seeds in test_seeds.values())}")
    print(f"   maxTraLen：{conf.maxTraLen}")
    print(f"   device：{conf.device}")
    print("-" * 60)

    with torch.no_grad():
        # 3. 遍历每个社区的采样种子，合并路径后评估模型表现
        batch_size = 10
        processed_count = 0
        total_communities = len(test_seeds)

        for name_tag, seeds in test_seeds.items():
            # 获取当前社区的真实节点列表
            true_com = [d for nt, d in true_coms if nt == name_tag][0]
            if len(true_com) == 0:
                continue

            # ========== 核心：用真实的sample_bs_trajectories批量扩展种子 ==========
            # 调用你实际的批量采样方法（无虚构方法）
            pred_coms, _ = expander.sample_bs_trajectories(seeds)  # 返回：[[种子1扩展节点], [种子2扩展节点], ...]
            
            # ========== 合并当前社区所有种子的扩展路径 ==========
            merged_pred_com = set()  # 用集合自动去重
            for pred_com in pred_coms:
                # 过滤停止符，添加到合并集合
                valid_nodes = [node for node in pred_com if node != "Stp"]
                merged_pred_com.update(valid_nodes)  # 合并并去重

            # 转换为列表（适配eval_scores输入格式）
            merged_pred_com = list(merged_pred_com)

            # ========== 计算合并后的指标 ==========
            p, r, f1 = expander.eval_scores(merged_pred_com, true_com)

            all_metrics["precision"].append(p)
            all_metrics["recall"].append(r)
            all_metrics["f1"].append(f1)

            # 进度打印
            processed_count += 1
            if processed_count % batch_size == 0 or processed_count == total_communities:
                print(f"📈 进度：{processed_count}/{total_communities}")
                print(f"   社区：{name_tag} | 采样种子数：{len(seeds)}")
                print(f"   合并后预测节点数：{len(merged_pred_com)} | 真实节点数：{len(true_com)}")
                print(f"   P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
                print(f"   累计平均F1：{np.mean(all_metrics['f1']):.4f}")
                print("-" * 40)

    # 4. 计算整体平均指标
    avg_metrics = {
        "avg_precision": round(np.mean(all_metrics["precision"]), 4),
        "avg_recall": round(np.mean(all_metrics["recall"]), 4),
        "avg_f1": round(np.mean(all_metrics["f1"]), 4),
        "std_f1": round(np.std(all_metrics["f1"]), 4)
    }

    print("\n" + "=" * 60)
    print("📊 测试集最终结果（真实社区+路径合并）")
    print("=" * 60)
    print(f"平均精度(P)：{avg_metrics['avg_precision']}")
    print(f"平均召回(R)：{avg_metrics['avg_recall']}")
    print(f"平均F1：{avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})")
    print(f"评估社区数：{len(test_seeds)} | 合并路径数：{processed_count}")
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
        print("🧪 开始测试（真实社区+动态采样种子）")
        print(f"{'='*70}")

        # 构建测试图（保留真实社区信息）
        test_g = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge,
        )
        expander.graph = test_g

        # 调用修改后的eval_model（真实社区+动态采样种子）
        test_metrics = eval_model(expander, test_g, conf)
        
        # 打印最终结果
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