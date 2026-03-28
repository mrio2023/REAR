import random
import torch
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Set, Any

# 假设这些是你已有的模块
from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .tool import compute_single_metrics, safe_mean
from .dataProcess import DataLoader
from .tee import Tee
from .refiner import Refiner  # 导入 Refiner


class Starter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.params.setdefault("max_iter", 1)

    def eval_model(self, expander: Expander, test_g: Graph, refiner=None) -> None:
        """
        评估模型：计算原始预测指标，若提供 Refiner 则额外计算精炼后的指标。
        """
        expander.model.eval()

        # 1. 准备数据
        true_coms: List[Tuple[str, List]] = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.community_seeds.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]
        print(f"有 {len(true_coms)} 个社区")

        test_seeds: Dict[str, List] = {}
        for name_tag, addr_list in true_coms:
            test_seeds[name_tag] = random.sample(addr_list, k=1)

        print(
            f"测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}"
        )
        print("-" * 60)

        # 2. 初始推理
        all_seeds_flat = []
        community_map = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        pred_coms_flat = []
        if all_seeds_flat:
            with torch.no_grad():
                pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)

        # 原始预测（未精炼）
        community_pred_original: Dict[str, Set] = {
            tag: set() for tag in test_seeds.keys()
        }
        for idx, pred_com in enumerate(pred_coms_flat):
            if idx < len(community_map):
                valid_nodes = [n for n in pred_com if n != "Stp"]
                community_pred_original[community_map[idx]].update(valid_nodes)

        # 计算原始指标
        metrics_original = {"precision": [], "recall": [], "f1": [], "jaccard": []}
        for name_tag, true_addr in true_coms:
            pred_com = list(community_pred_original.get(name_tag, set()))
            compute_single_metrics(pred_com, true_addr, metrics_original)

        avg_p_orig = safe_mean(metrics_original["precision"])
        avg_r_orig = safe_mean(metrics_original["recall"])
        avg_f1_orig = safe_mean(metrics_original["f1"])
        avg_j_orig = safe_mean(metrics_original["jaccard"])

        print("\n" + "=" * 80)
        print("📊 原始评估指标（无精炼）")
        print("=" * 80)
        print(
            f"Precision: {avg_p_orig:.4f} | Recall: {avg_r_orig:.4f} | F1: {avg_f1_orig:.4f} | Jaccard: {avg_j_orig:.4f}"
        )

        # 3. 若提供 Refiner，则进行精炼并计算精炼后指标
        if refiner is not None:
            # 应用精炼
            community_pred_refined = {
                tag: set(community_pred_original[tag])
                for tag in community_pred_original.keys()
            }
            for tag in community_pred_refined.keys():
                comm_list = list(community_pred_refined[tag])
                refined = refiner.refine_community(comm_list, threshold=0.3)
                community_pred_refined[tag] = set(refined)
            print("🔧 已应用 Refiner 精炼（剔除噪声节点）")

            # 计算精炼后指标
            metrics_refined = {"precision": [], "recall": [], "f1": [], "jaccard": []}
            for name_tag, true_addr in true_coms:
                pred_com = list(community_pred_refined.get(name_tag, set()))
                compute_single_metrics(pred_com, true_addr, metrics_refined)

            avg_p_ref = safe_mean(metrics_refined["precision"])
            avg_r_ref = safe_mean(metrics_refined["recall"])
            avg_f1_ref = safe_mean(metrics_refined["f1"])
            avg_j_ref = safe_mean(metrics_refined["jaccard"])

            print("\n" + "=" * 80)
            print("📊 精炼后评估指标")
            print("=" * 80)
            print(
                f"Precision: {avg_p_ref:.4f} | Recall: {avg_r_ref:.4f} | F1: {avg_f1_ref:.4f} | Jaccard: {avg_j_ref:.4f}"
            )
            print("\n" + "=" * 80)
            print("📈 指标变化")
            print("=" * 80)
            print(
                f"Precision: {avg_p_orig:.4f} → {avg_p_ref:.4f} ({avg_p_ref - avg_p_orig:+.4f})"
            )
            print(
                f"Recall:    {avg_r_orig:.4f} → {avg_r_ref:.4f} ({avg_r_ref - avg_r_orig:+.4f})"
            )
            print(
                f"F1:        {avg_f1_orig:.4f} → {avg_f1_ref:.4f} ({avg_f1_ref - avg_f1_orig:+.4f})"
            )
            print(
                f"Jaccard:   {avg_j_orig:.4f} → {avg_j_ref:.4f} ({avg_j_ref - avg_j_orig:+.4f})"
            )
            print("=" * 80)

    def run(self, dfname: str, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.params["device"] == "cuda":
            torch.cuda.manual_seed(seed)

        log_dir = os.path.join("logs", dfname)
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{dfname}_seed{seed}_{timestamp}.log")

        f = None
        original_stdout = sys.stdout
        tee = None

        try:
            f = open(log_file, "w", encoding="utf-8")
            tee = Tee(f, original_stdout)
            sys.stdout = tee

            print(f" 数据集：{dfname}  seed={seed}")
            print(
                f"核心参数：min_community_size={self.params['min_community_size']}, epoch={self.params['epoch']}, seedNum={self.params['seedNum']}"
            )
            print("-" * 70)

            loader = DataLoader(
                dataset_name=dfname,
                train_ratio=0.8,
                min_com_size=self.params["min_community_size"],
                seed=seed,
            )

            global_adj = loader.graph["adj"]
            global_features = loader.nodefeats

            train_communities = {
                f"train_{idx}": comm for idx, comm in enumerate(loader.train_comms)
            }
            test_communities = {
                f"test_{idx}": comm for idx, comm in enumerate(loader.test_comms)
            }

            train_g = Graph(
                adj=global_adj, features=global_features, communities=train_communities
            )
            test_g = Graph(
                adj=global_adj, features=global_features, communities=test_communities
            )

            print(
                f"训练图：节点数 {train_g.n_nodes}，特征维度 {train_g.embedsize}，训练社区数 {len(train_communities)}"
            )
            print(
                f"测试图：节点数 {test_g.n_nodes}，特征维度 {test_g.embedsize}，测试社区数 {len(test_communities)}"
            )

            device = torch.device(self.params["device"])
            model = Agent(
                input_size=train_g.embedsize, hidden_size=self.params["hidden_size"]
            ).to(device)

            expander = Expander(
                graph=train_g,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=self.params["lr"]),
                device=device,
                maxLen=self.params["maxLen"],
                gamma=self.params["gamma"],
                f1_base_weight=self.params["f1_base_weight"],
                p_bias=self.params["p_bias"],
                r_bias=self.params["r_bias"],
                grad_norm=self.params["grad_norm"],
                target_f1=self.params["target_f1"],
                stop_reward_scale=self.params["stop_reward_scale"],
            )

            epoch = self.params["epoch"]
            seedNum = self.params["seedNum"]
            coms = list(train_g.communities.values())

            print(f"开始训练：{epoch}轮 | 每轮采样{seedNum}种子")
            for i in range(epoch):
                true_coms = random.sample(coms, k=seedNum)
                seeds = [random.choice(c) for c in true_coms]
                loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
                print(f"Epoch {i+1}/{epoch} | loss: {loss:.4f}")

            # -------------------- 新增：训练 Refiner --------------------
            print("\n开始训练 Refiner...")
            refiner = Refiner(train_g, expander)
            refiner.trainRefiner()
            # ---------------------------------------------------------

            # 测试：应用 Refiner
            print("\n开始测试（真实社区+动态采样种子）")
            expander.graph = test_g
            # 测试前切换图
            refiner.train_g = test_g
            self.eval_model(expander, test_g, refiner=refiner)

        except Exception as e:
            print(f"\n❌ 运行出错: {str(e)}", file=sys.stderr)
            raise
        finally:
            sys.stdout = original_stdout
            if tee:
                tee.flush()
            if f and not f.closed:
                f.close()
            print(f"\n📁 日志已保存至：{log_file}")
