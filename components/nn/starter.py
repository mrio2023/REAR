from typing import Union, Optional, List, Set, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import time  # 导入计时模块
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


def eval_model(expander: Expander, test_g: Graph, conf: Configure) -> dict:
    expander.model.eval()
    all_metrics = {"precision": [], "recall": [], "f1": []}

    # ========== 核心修正1：适配community_seeds的{name_tag: set}结构 ==========
    true_coms = test_g.community_seeds  # 原始是dict：{name_tag: set(地址)}
    # 转为(name_tag, 地址列表)的列表，同时过滤空社区
    true_coms_list = []
    for name_tag, addr_set in true_coms.items():
        if not isinstance(addr_set, (set, list)) or len(addr_set) == 0:
            continue
        # 统一转为list，方便后续采样
        addr_list = list(addr_set)
        true_coms_list.append((name_tag, addr_list))
    true_coms = true_coms_list  # 覆盖为修正后的列表

    # ========== 计时：测试阶段总开始 ==========
    start_eval = time.time()

    # ========== 计时：种子采样环节 ==========
    start_sample = time.time()
    # 按社区大小采种子（修正：基于list采样，避免random.sample不支持set）
    test_seeds = {}
    for name_tag, addr_list in true_coms:
        if len(addr_list) == 0:
            continue
        sample_num = int(1 + len(addr_list) / conf.maxTraLen)
        sample_num = min(sample_num, len(addr_list))
        # 现在addr_list是list，random.sample可以正常使用
        s = random.sample(addr_list, k=sample_num)
        test_seeds[name_tag] = s
    end_sample = time.time()
    print(f"\n⏱️  测试种子采样耗时：{end_sample - start_sample:.2f} 秒")

    print(
        f"\n🔧 测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}"
    )
    print("-" * 60)

    # ===================== 核心提速：一次性把所有种子压一批跑 =====================
    start_batch_prep = time.time()
    all_seeds_flat = []
    community_map = []  # 记录每个种子属于哪个社区
    for name_tag, seeds in test_seeds.items():
        all_seeds_flat.extend(seeds)
        community_map.extend([name_tag] * len(seeds))
    end_batch_prep = time.time()
    print(f"⏱️  测试批量种子整理耗时：{end_batch_prep - start_batch_prep:.2f} 秒")

    pred_coms_flat = []
    # ========== 计时：核心推理环节（最可能慢的地方） ==========
    start_infer = time.time()
    with torch.no_grad():
        if all_seeds_flat:
            # 整批一次性跑完，不循环一个个社区跑（最大提速点）
            pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)
    end_infer = time.time()
    print(f"⏱️  测试批量推理耗时（核心）：{end_infer - start_infer:.2f} 秒")
    print(f"   推理种子数：{len(all_seeds_flat)}，单种子平均耗时：{(end_infer - start_infer)/len(all_seeds_flat):.4f} 秒/种子")

    # ========== 计时：结果整理环节 ==========
    start_result = time.time()
    # 按社区把结果收回来
    community_pred = {name_tag: set() for name_tag in test_seeds.keys()}
    for idx, pred_com in enumerate(pred_coms_flat):
        name_tag = community_map[idx]
        valid = [n for n in pred_com if n != "Stp"]
        community_pred[name_tag].update(valid)
    end_result = time.time()
    print(f"⏱️  测试结果整理耗时：{end_result - start_result:.2f} 秒")

    # ========== 计时：指标计算环节 ==========
    start_metric = time.time()
    # 统一算指标（修正：匹配true_coms的list结构）
    for name_tag, true_addr_list in true_coms:
        if name_tag not in community_pred:
            continue
        merged = list(community_pred[name_tag])
        # 计算指标时，true_com可以是list/set，eval_scores会处理
        p, r, f1 = expander.eval_scores(merged, true_addr_list)

        all_metrics["precision"].append(p)
        all_metrics["recall"].append(r)
        all_metrics["f1"].append(f1)
    end_metric = time.time()
    print(f"⏱️  测试指标计算耗时：{end_metric - start_metric:.2f} 秒")

    # 汇总
    avg_metrics = {
        "avg_precision": round(np.mean(all_metrics["precision"]), 4),
        "avg_recall": round(np.mean(all_metrics["recall"]), 4),
        "avg_f1": round(np.mean(all_metrics["f1"]), 4),
        "std_f1": round(np.std(all_metrics["f1"]), 4),
    }

    # ========== 总耗时统计 ==========
    end_eval = time.time()
    print(f"\n⏱️  测试阶段总耗时：{end_eval - start_eval:.2f} 秒")
    
    print("\n" + "=" * 60)
    print("📊 测试结果（批量加速版）")
    print("=" * 60)
    print(f"P: {avg_metrics['avg_precision']}")
    print(f"R: {avg_metrics['avg_recall']}")
    print(f"F1: {avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})")
    print("=" * 60)

    return avg_metrics


def run(dfname, conf: Configure, seed: int = 42):
    # ========== 全局计时：整个run函数 ==========
    start_run = time.time()
    
    set_seed(seed)
    # 新增：给Configure补充epoch参数（避免KeyError）
    if not hasattr(conf, "epoch"):
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

        # ========== 计时：数据处理环节 ==========
        start_data = time.time()
        dp = dataProcess(
            dfname=conf.dfname,
            normal_node_ratio=conf.normal_node_ratio,
            expand_hop=conf.expand_hop,
            min_community_size=conf.min_community_size,
        )
        end_data = time.time()
        print(f"✅ dataProcess 完成 | 耗时：{end_data - start_data:.2f} 秒")

        print(f"\n数据规模：")
        print(f"  训练节点：{len(dp.train_nodes)}")
        print(f"  训练黑客：{len(dp.train_hacker)}")
        print(f"  测试节点：{len(dp.test_nodes)}")
        print(f"  测试黑客：{len(dp.test_hacker)}")
        print("-" * 70)

        # ========== 计时：训练图构建 ==========
        start_train_graph = time.time()
        g = Graph(
            dfnode=dp.train_nodes,
            dffeature=dp.train_feature,
            dfhacker=dp.train_hacker,
            dfedge=dp.train_edge,
        )
        end_train_graph = time.time()
        print(f"✅ 训练图构建完成 | 耗时：{end_train_graph - start_train_graph:.2f} 秒")

        # ========== 计时：模型初始化 ==========
        start_model_init = time.time()
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
            len_penalty_coeff=conf.len_penalty_coeff,
        )
        end_model_init = time.time()
        print(f"✅ Expander 初始化完成 | 耗时：{end_model_init - start_model_init:.2f} 秒")

        origin_seeds = dp.train_hacker["address"].values.tolist()
        print(f"\n🚀 开始训练，每轮采样 {conf.seedNum} 个种子，共 {conf.epoch} 轮")

        # ========== 计时：训练环节 ==========
        start_train = time.time()
        for i in range(conf.epoch):
            seeds = random.sample(origin_seeds, k=conf.seedNum)
            true_coms = [
                g.sampleTrajectory(s, traj_length=conf.maxTraLen) for s in seeds
            ]
            print(f"\n📝 Epoch {i}")
            loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
            print(f"loss: {loss:.4f}")
        end_train = time.time()
        print(f"\n✅ 训练完成 | 总耗时：{end_train - start_train:.2f} 秒 | 单轮平均：{(end_train - start_train)/conf.epoch:.2f} 秒/轮")

        print(f"\n{'='*70}")
        print("🧪 开始测试（真实社区+动态采样种子）")
        print(f"{'='*70}")

        # ========== 计时：测试图构建 ==========
        start_test_graph = time.time()
        # 构建测试图（保留真实社区信息）
        test_g = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge,
        )
        expander.graph = test_g
        end_test_graph = time.time()
        print(f"✅ 测试图构建完成 | 耗时：{end_test_graph - start_test_graph:.2f} 秒")

        # 调用修改后的eval_model（真实社区+动态采样种子）
        test_metrics = eval_model(expander, test_g, conf)

        # 打印最终结果
        print(f"\n✅ 数据集 {dfname} 训练完成！")
        print(f"   最终测试集F1：{test_metrics['avg_f1']}")
        print(f"   最终测试集P：{test_metrics['avg_precision']}")
        print(f"   最终测试集R：{test_metrics['avg_recall']}")

        # ========== 全局总耗时 ==========
        end_run = time.time()
        print(f"\n⏱️  整个run函数（训练+测试）总耗时：{end_run - start_run:.2f} 秒")

        return test_metrics

    except Exception as e:
        print(f"\n❌ 运行失败：{e}")
        import traceback

        traceback.print_exc()
        return None