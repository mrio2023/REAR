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


class Starter:
    def __init__(self, params: Dict):
        """
        仅接收参数字典，不做任何本地赋值，全程透传（解耦核心）
        :param params: 包含所有配置的字典，key为参数名，value为参数值
        """
        self.params = params  # 仅保存字典引用，不拆解为实例属性

    def set_seed(self, seed: int):
        """固定所有随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"✅ 所有随机种子已固定为：{seed}")

    def eval_model(self, expander: Expander, test_g: Graph) -> Dict:
        """批量加速版模型评估，直接从self.params取参数"""
        expander.model.eval()
        # 预处理真实社区
        true_coms = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.community_seeds.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]

        # 1. 种子采样（直接从字典取maxTraLen）
        start_sample = time.time()
        test_seeds = {}
        for name_tag, addr_list in true_coms:
            # 直接用self.params['maxTraLen']，无本地属性
            sample_num = min(int(1 + len(addr_list) / self.params.get('maxTraLen', 50)), len(addr_list))
            test_seeds[name_tag] = random.sample(addr_list, k=sample_num)
        print(f"\n⏱️  测试种子采样耗时：{time.time()-start_sample:.2f} 秒")
        print(
            f"🔧 测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}"
        )
        print("-" * 60)

        # 2. 批量整理种子
        start_batch = time.time()
        all_seeds_flat = []
        community_map = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))
        print(f"⏱️  测试批量种子整理耗时：{time.time()-start_batch:.2f} 秒")

        # 3. 批量推理
        pred_coms_flat = []
        if all_seeds_flat:
            start_infer = time.time()
            with torch.no_grad():
                pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)
            infer_time = time.time() - start_infer
            print(f"⏱️  测试批量推理耗时（核心）：{infer_time:.2f} 秒")
            print(
                f"   推理种子数：{len(all_seeds_flat)}，单种子平均耗时：{infer_time/len(all_seeds_flat):.4f} 秒/种子"
            )

        # 4. 结果整理
        start_result = time.time()
        community_pred = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            valid_nodes = [n for n in pred_com if n != "Stp"]
            community_pred[community_map[idx]].update(valid_nodes)
        print(f"⏱️  测试结果整理耗时：{time.time()-start_result:.2f} 秒")

        # 5. 计算指标
        start_metric = time.time()
        metrics = {"precision": [], "recall": [], "f1": []}
        for name_tag, true_addr in true_coms:
            if name_tag not in community_pred:
                continue
            p, r, f1 = expander.eval_scores(list(community_pred[name_tag]), true_addr)
            metrics["precision"].append(p)
            metrics["recall"].append(r)
            metrics["f1"].append(f1)
        print(f"⏱️  测试指标计算耗时：{time.time()-start_metric:.2f} 秒")

        # 汇总指标
        avg_metrics = {
            "avg_precision": round(np.mean(metrics["precision"]), 4),
            "avg_recall": round(np.mean(metrics["recall"]), 4),
            "avg_f1": round(np.mean(metrics["f1"]), 4),
            "std_f1": round(np.std(metrics["f1"]), 4),
        }

        # 打印结果
        print(f"\n⏱️  测试阶段总耗时：{time.time()-start_sample:.2f} 秒")
        print("\n" + "=" * 60)
        print("📊 测试结果（批量加速版）")
        print("=" * 60)
        print(
            f"P: {avg_metrics['avg_precision']} | R: {avg_metrics['avg_recall']} | F1: {avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})"
        )
        print("=" * 60)

        return avg_metrics

    def run(self, dfname, seed) -> Dict:
        self.set_seed(seed)

        try:
            # 打印基础信息（直接从字典取参数）
            print(f"\n{'='*70}\n📌 数据集：{dfname}  seed={seed}\n{'='*70}")
            print(
                f"核心参数：\n  normal_node_ratio: {self.params.get('normal_node_ratio', 0.8)} | expand_hop: {self.params.get('expand_hop', 2)}\n  "
                f"epoch: {self.params.get('epoch', 30)} | seedNum: {self.params.get('seedNum', 100)}"
            )
            print("-" * 70)

            # 1. 数据处理：直接透传字典里的参数
            dp = dataProcess(
                dfname=self.params.get("dfname", dfname),  # 兼容传参优先级
                normal_node_ratio=self.params.get("normal_node_ratio", 0.8),
                expand_hop=self.params.get("expand_hop", 2),
                min_community_size=self.params.get("min_community_size", 10),
            )

            print(
                f"数据规模：训练节点{len(dp.train_nodes)} | 训练黑客{len(dp.train_hacker)} | 测试节点{len(dp.test_nodes)} | 测试黑客{len(dp.test_hacker)}"
            )

            # 2. 构建训练图：透传参数
            train_g = Graph(
                dfnode=dp.train_nodes,
                dffeature=dp.train_feature,
                dfhacker=dp.train_hacker,
                dfedge=dp.train_edge,
            )

            # 3. 初始化模型和扩展器：全程透传字典参数
            device = torch.device(self.params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            model = Agent(
                input_size=train_g.embedsize,
                hidden_size=self.params.get("hidden_size", 128)  # 从字典取hidden_size
            ).to(device)
            
            expander = Expander(
                graph=train_g,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=self.params.get("lr", 1e-4)),
                device=device,
                maxLen=self.params.get("maxLen", 20),
                gamma=self.params.get("gamma", 0.9),
                f1_base_weight=self.params.get("f1_base_weight", 0.5),
                p_bias=self.params.get("p_bias", 0.1),
                min_f1_threshold=self.params.get("min_f1_threshold", 0.1),
                len_penalty_coeff=self.params.get("len_penalty_coeff", 0.01),
            )

            # 4. 训练过程：透传epoch/seedNum
            origin_seeds = dp.train_hacker["address"].tolist()
            epoch = self.params.get("epoch", 30)
            seedNum = self.params.get("seedNum", 100)
            print(f"\n🚀 开始训练：{epoch}轮 | 每轮采样{seedNum}种子")
            for i in range(epoch):
                seeds = random.sample(origin_seeds, k=seedNum)
                true_coms = [train_g.sampleTrajectory(s) for s in seeds]
                loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
                print(f"📝 Epoch {i} | loss: {loss:.4f}")

            # 5. 测试模型：透传参数
            print(f"\n{'='*70}\n🧪 开始测试（真实社区+动态采样种子）\n{'='*70}")
          
            test_g = Graph(
                dfnode=dp.test_nodes,
                dffeature=dp.test_feature,
                dfhacker=dp.test_hacker,
                dfedge=dp.test_edge,
            )
            expander.graph = test_g

            test_metrics = self.eval_model(expander, test_g)

            print(f"\n✅ 数据集 {dfname} 运行完成！")
            print(
                f"   最终F1：{test_metrics['avg_f1']} | P：{test_metrics['avg_precision']} | R：{test_metrics['avg_recall']}"
            )

            return test_metrics

        except Exception as e:
            print(f"\n❌ 运行失败：{str(e)}")
            import traceback
            traceback.print_exc()
            return None