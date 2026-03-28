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
from .tool import compute_single_metrics,aggregate_avg_metrics
from .dataProcess import DataLoader
from .tee import Tee



class Starter:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.params.setdefault("max_iter", 1)

    def eval_model(self, expander: Expander, test_g: Graph) -> Dict[str, float]:
        """
        评估模型：使用多种子扩展策略
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

        print(f"测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}")
        print("-" * 60)

        # 2. 初始推理 (Before)
        all_seeds_flat = []
        community_map = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        pred_coms_flat = []
        if all_seeds_flat:
            with torch.no_grad():
                pred_coms_flat, _ = expander.sample_bs_trajectories(all_seeds_flat)

        community_pred_before: Dict[str, Set] = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            if idx < len(community_map):
                valid_nodes = [n for n in pred_com if n != "Stp"]
                community_pred_before[community_map[idx]].update(valid_nodes)

        # 3. 多种子迭代扩展 (After)
        community_pred_after: Dict[str, Set] = {}
        max_iter = self.params.get("max_iter", 1)

        for name_tag in test_seeds.keys():
            current_com = set(community_pred_before[name_tag])
            if not current_com:
                community_pred_after[name_tag] = set()
                continue

            node_coverage = {}
            for iter_idx in range(max_iter):
                # 计算社区嵌入
                embeddings = []
                for node in current_com:
                    emb = test_g.singleNodeEmbed(node)
                    embeddings.append(emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb)
                com_emb = np.mean(embeddings, axis=0)
                norm_com = np.linalg.norm(com_emb)

                # 计算相似度
                sim_scores = []
                for node in current_com:
                    emb = test_g.singleNodeEmbed(node)
                    emb = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb
                    norm_emb = np.linalg.norm(emb)
                    sim = np.dot(com_emb, emb) / (norm_com * norm_emb) if norm_emb != 0 else 0.0
                    sim_scores.append((node, sim))

                # 选种
                sim_scores.sort(key=lambda x: x[1], reverse=True)
                k = min(max(1, int(len(current_com) * 0.5)), 10)
                topk_nodes = [node for node, _ in sim_scores[:k]]

                # 扩展
                all_new_nodes = set()
                for seed_node in topk_nodes:
                    with torch.no_grad():
                        preds, _ = expander.sample_bs_trajectories([seed_node])
                    for p in preds:
                        new_in_traj = [n for n in p if n != "Stp"]
                        for n in new_in_traj:
                            if n not in current_com:
                                node_coverage[n] = node_coverage.get(n, 0) + 1
                        all_new_nodes.update(new_in_traj)

                new_nodes = all_new_nodes - current_com
                if not new_nodes:
                    print(f"社区 {name_tag} 迭代 {iter_idx+1} 轮后无新增节点，提前停止")
                    break
                
                current_com.update(new_nodes)
                print(f"社区 {name_tag} 迭代 {iter_idx+1} 轮，新增 {len(new_nodes)} 个节点，当前大小 {len(current_com)}")

            if node_coverage:
                high_risk_nodes = {n: cnt for n, cnt in node_coverage.items() if cnt >= 2}
                if high_risk_nodes:
                    print(f"社区 {name_tag} 中高危险点数量：{len(high_risk_nodes)}")

            community_pred_after[name_tag] = current_com

        # ================== 计算指标 (调用解耦后的函数) ==================
        metrics_before = {"precision": [], "recall": [], "f1": [], "jaccard": []}
        metrics_after = {"precision": [], "recall": [], "f1": [], "jaccard": []}

        for name_tag, true_addr in true_coms:
            # 扩展前
            pred_before = list(community_pred_before.get(name_tag, set()))
            compute_single_metrics(pred_before, true_addr, metrics_before)

            # 扩展后
            pred_after = list(community_pred_after.get(name_tag, set()))
            compute_single_metrics(pred_after, true_addr, metrics_after)

        avg_metrics = aggregate_avg_metrics(metrics_before, metrics_after)

        # 打印结果
        print("\n" + "=" * 80)
        print("📊 扩展前后指标对比")
        print("=" * 80)
        print(f"扩展前 | P: {avg_metrics['before_avg_precision']} | R: {avg_metrics['before_avg_recall']} | F1: {avg_metrics['before_avg_f1']}")
        print(f"扩展后 | P: {avg_metrics['after_avg_precision']} | R: {avg_metrics['after_avg_recall']} | F1: {avg_metrics['after_avg_f1']}")
        print("=" * 80)

        return avg_metrics

    def run(self, dfname: str, seed: int) -> Dict[str, float]:
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
        test_metrics = {}

        try:
            f = open(log_file, "w", encoding="utf-8")
            tee = Tee(f, original_stdout)
            sys.stdout = tee

            print(f" 数据集：{dfname}  seed={seed}")
            print(f"核心参数：min_community_size={self.params['min_community_size']}, epoch={self.params['epoch']}, seedNum={self.params['seedNum']}")
            print("-" * 70)

            loader = DataLoader(
                dataset_name=dfname,
                train_ratio=0.8,
                min_com_size=self.params["min_community_size"],
                seed=seed,
            )

            global_adj = loader.graph["adj"]
            global_features = loader.nodefeats

            train_communities = {f"train_{idx}": comm for idx, comm in enumerate(loader.train_comms)}
            test_communities = {f"test_{idx}": comm for idx, comm in enumerate(loader.test_comms)}

            train_g = Graph(adj=global_adj, features=global_features, communities=train_communities)
            test_g = Graph(adj=global_adj, features=global_features, communities=test_communities)

            print(f"训练图：节点数 {train_g.n_nodes}，特征维度 {train_g.embedsize}，训练社区数 {len(train_communities)}")
            print(f"测试图：节点数 {test_g.n_nodes}，特征维度 {test_g.embedsize}，测试社区数 {len(test_communities)}")

            device = torch.device(self.params["device"])
            model = Agent(input_size=train_g.embedsize, hidden_size=self.params["hidden_size"]).to(device)

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
                target_f1=self.params["target_f1"], # 修复了参数名不一致
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

            print(f"开始测试（真实社区+动态采样种子）")
            expander.graph = test_g
            test_metrics = self.eval_model(expander, test_g)

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

        return test_metrics