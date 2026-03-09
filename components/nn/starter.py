from typing import Dict, List
import pandas as pd
import numpy as np
import torch
import random
import os
import time
from .dataProcess import dataProcess
from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .tool import eval_scores, eval_f1


class Starter:
    def __init__(self, params: Dict):
        self.params = params  # 强制依赖传入的params，无任何默认值

    def eval_model(self, expander: Expander, test_g: Graph) -> Dict:
        expander.model.eval()

        # 提取真实社区
        true_coms = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.community_seeds.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]
        print("有", len(true_coms), "个社区")

        # 生成测试种子
        test_seeds = {}
        for name_tag, addr_list in true_coms:
            sample_num = min(
                int(1 + len(addr_list) / self.params["maxLen"]),
                len(addr_list),
            )
            test_seeds[name_tag] = random.sample(addr_list, k=sample_num)

        print(
            f"🔧 测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}"
        )
        print("-" * 60)

        # 扁平化种子和社区映射
        all_seeds_flat = []
        community_map = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        # 批量推理得到初始预测
        pred_coms_flat = []
        if all_seeds_flat:
            with torch.no_grad():
                pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)
            print(f"   推理种子数：{len(all_seeds_flat)}")

        # 聚合扩展前预测结果
        community_pred_before = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            valid_nodes = [n for n in pred_com if n != "Stp"]
            community_pred_before[community_map[idx]].update(valid_nodes)

        # ========== 修改点：用模型迭代扩展代替相似度扩展 ==========
        community_pred_after = {}
        max_iter = 2  # 迭代次数，可根据需要调整（或从 self.params 获取）
        for name_tag in test_seeds.keys():
            current_com = set(community_pred_before[name_tag])
            if not current_com:
                community_pred_after[name_tag] = set()
                continue

            # 迭代扩展
            for _ in range(max_iter):
                # 以当前社区所有节点作为种子（去重后转为列表）
                seeds = list(current_com)
                with torch.no_grad():
                    # 注意：sample_bs_trajectories 接收的是列表，返回对应长度的预测列表
                    preds, _ = expander.sample_bs_trajectories(seeds)
                new_nodes = set()
                for p in preds:
                    new_nodes.update([n for n in p if n != "Stp"])
                # 如果没有新增节点，提前停止
                if new_nodes.issubset(current_com):
                    break
                current_com.update(new_nodes)

            community_pred_after[name_tag] = current_com
        # ==========================================================

        # 计算指标
        metrics_before = {"precision": [], "recall": [], "f1": []}
        metrics_after = {"precision": [], "recall": [], "f1": []}

        for name_tag, true_addr in true_coms:
            p_b, r_b, f1_b = eval_scores(
                list(community_pred_before[name_tag]), true_addr
            )
            metrics_before["precision"].append(p_b)
            metrics_before["recall"].append(r_b)
            metrics_before["f1"].append(f1_b)

            p_a, r_a, f1_a = eval_scores(
                list(community_pred_after[name_tag]), true_addr
            )
            metrics_after["precision"].append(p_a)
            metrics_after["recall"].append(r_a)
            metrics_after["f1"].append(f1_a)

        # 汇总指标
        avg_metrics = {
            "before_avg_precision": round(np.mean(metrics_before["precision"]), 4),
            "before_avg_recall": round(np.mean(metrics_before["recall"]), 4),
            "before_avg_f1": round(np.mean(metrics_before["f1"]), 4),
            "before_std_f1": round(np.std(metrics_before["f1"]), 4),
            "after_avg_precision": round(np.mean(metrics_after["precision"]), 4),
            "after_avg_recall": round(np.mean(metrics_after["recall"]), 4),
            "after_avg_f1": round(np.mean(metrics_after["f1"]), 4),
            "after_std_f1": round(np.std(metrics_after["f1"]), 4),
            "f1_improvement": round(
                (np.mean(metrics_after["f1"]) - np.mean(metrics_before["f1"]))
                / np.mean(metrics_before["f1"])
                * 100,
                2,
            ),
            "recall_improvement": round(
                (np.mean(metrics_after["recall"]) - np.mean(metrics_before["recall"]))
                / np.mean(metrics_before["recall"])
                * 100,
                2,
            ),
            "precision_change": round(
                (
                    np.mean(metrics_after["precision"])
                    - np.mean(metrics_before["precision"])
                )
                / np.mean(metrics_before["precision"])
                * 100,
                2,
            ),
        }

        # 打印结果
        print("\n" + "=" * 80)
        print("📊 扩展前后指标对比（模型迭代扩展）")
        print("=" * 80)
        print(
            f"扩展前 | P: {avg_metrics['before_avg_precision']} | R: {avg_metrics['before_avg_recall']} | F1: {avg_metrics['before_avg_f1']} (±{avg_metrics['before_std_f1']})"
        )
        print(
            f"扩展后 | P: {avg_metrics['after_avg_precision']} | R: {avg_metrics['after_avg_recall']} | F1: {avg_metrics['after_avg_f1']} (±{avg_metrics['after_std_f1']})"
        )
        print(
            f"变化幅度 | P: {avg_metrics['precision_change']}% | R: {avg_metrics['recall_improvement']}% | F1: {avg_metrics['f1_improvement']}%"
        )
        print("=" * 80)

        return avg_metrics

    def run(self, dfname, seed) -> Dict:
        # 移除所有try-except，强制暴露错误
        # 打印基础信息（强制取值，无默认）
        print(f"\n{'='*70}\n📌 数据集：{dfname}  seed={seed}\n{'='*70}")
        print(
            f"核心参数：\n  normal_node_ratio: {self.params['normal_node_ratio']} | expand_hop: {self.params['expand_hop']}\n  "
            f"epoch: {self.params['epoch']} | seedNum: {self.params['seedNum']}"
        )
        print("-" * 70)

        # 1. 数据处理（强制传参，无默认）
        dp = dataProcess(
            dfname=self.params["dfname"],  # 强制使用params中的dfname
            normal_node_ratio=self.params["normal_node_ratio"],
            expand_hop=self.params["expand_hop"],
            min_community_size=self.params["min_community_size"],
        )

        print(
            f"数据规模：训练节点{len(dp.train_nodes)} | 训练黑客{len(dp.train_hacker)} | 测试节点{len(dp.test_nodes)} | 测试黑客{len(dp.test_hacker)}"
        )

        # 2. 构建训练图（无兜底，依赖dp的输出）
        train_g = Graph(
            dfnode=dp.train_nodes,
            dffeature=dp.train_feature,
            dfhacker=dp.train_hacker,
            dfedge=dp.train_edge,
        )

        # 3. 初始化模型和扩展器（强制取值，无默认）
        device = torch.device(self.params["device"])  # 强制指定device，无自动兜底
        model = Agent(
            input_size=train_g.embedsize,
            hidden_size=self.params["hidden_size"],  # 强制取值
        ).to(device)

        expander = Expander(
            graph=train_g,
            model=model,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=self.params["lr"]  # 强制指定学习率
            ),
            device=device,
            maxLen=self.params["maxLen"],
            gamma=self.params["gamma"],
            f1_base_weight=self.params["f1_base_weight"],
            p_bias=self.params["p_bias"],
            min_f1_threshold=self.params["min_f1_threshold"],
            len_penalty_coeff=self.params["len_penalty_coeff"],
        )

        # 4. 训练过程（强制取值，无默认）
        origin_seeds = dp.train_hacker["address"].tolist()
        epoch = self.params["epoch"]
        seedNum = self.params["seedNum"]
        print(f"\n🚀 开始训练：{epoch}轮 | 每轮采样{seedNum}种子")
        for i in range(epoch):
            seeds = random.sample(origin_seeds, k=seedNum)
            # 强制使用params中的maxLen，无默认
            true_coms = [
                train_g.sampleTrajectory(s, maxlen=self.params["maxLen"]) for s in seeds
            ]
            loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
            print(f"📝 Epoch {i} | loss: {loss:.4f}")

        # 5. 测试模型（无兜底）
        print(f"\n{'='*70}\n🧪 开始测试（真实社区+动态采样种子）\n{'='*70}")

        test_g = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge,
        )
        expander.graph = test_g

        test_metrics = self.eval_model(expander, test_g)

        # print(f"\n✅ 数据集 {dfname} 运行完成！")
        # print(
        #     f"   最终F1：{test_metrics['avg_f1']} | P：{test_metrics['avg_precision']} | R：{test_metrics['avg_recall']}"
        # )

        return test_metrics
