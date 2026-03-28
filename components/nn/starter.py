import random
import torch
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Set, Any
from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .tool import eval_scores
from .dataProcess import PreprocessedDataLoader


class Tee:
    """将输出同时写入文件和终端"""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, message: str) -> None:
        try:
            self.file.write(message)
            self.stream.write(message)
        except Exception as e:
            print(f"写入日志时出错: {e}", file=self.stream)

    def flush(self) -> None:
        self.file.flush()
        self.stream.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()


class Starter:
    def __init__(self, params: Dict[str, Any]):
        """
        初始化Starter
        Args:
            params: 包含所有配置参数的字典，必须包含以下键：
                - dfname, min_community_size,
                  device, hidden_size, lr, maxLen, gamma, f1_base_weight,
                  p_bias, r_bias, repeat_penalty_coeff, entropy_coeff,
                  grad_norm, target_recall, stop_reward_scale, epoch, seedNum
                - 可选: max_iter (默认2)
        """
        required_params = [
            "dfname", "min_community_size",
            "device", "hidden_size", "lr", "maxLen", "gamma", "f1_base_weight",
            "p_bias", "r_bias", "repeat_penalty_coeff", "entropy_coeff",
            "grad_norm", "target_recall", "stop_reward_scale", "epoch", "seedNum"
        ]
        for param in required_params:
            if param not in params:
                raise ValueError(f"缺少必要参数: {param}")

        self.params = params
        self.params.setdefault("max_iter", 1)

    def eval_model(self, expander: Expander, test_g: Graph) -> Dict[str, float]:
        """
        评估模型：修改迭代扩展逻辑，每次迭代选择社区内与社区嵌入最相似的节点作为种子扩展一次。
        """
        expander.model.eval()

        true_coms: List[Tuple[str, List]] = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.community_seeds.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]
        print(f"有 {len(true_coms)} 个社区")

        test_seeds: Dict[str, List] = {}
        for name_tag, addr_list in true_coms:
            sample_num = 1
            test_seeds[name_tag] = random.sample(addr_list, k=sample_num)

        print(f"测试：真实社区 {len(true_coms)}，总种子 {sum(len(v) for v in test_seeds.values())}")
        print("-" * 60)

        all_seeds_flat: List = []
        community_map: List[str] = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        pred_coms_flat: List[List] = []
        if all_seeds_flat:
            with torch.no_grad():
                pred_coms_flat, _, _ = expander.sample_bs_trajectories(all_seeds_flat)
            print(f"推理种子数：{len(all_seeds_flat)}")

        community_pred_before: Dict[str, Set] = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            if idx >= len(community_map):
                continue
            valid_nodes = [n for n in pred_com if n != "Stp"]
            community_pred_before[community_map[idx]].update(valid_nodes)

        # ================== 迭代扩展：每次选社区内最相似节点作为种子 ==================
        community_pred_after: Dict[str, Set] = {}
        max_iter = self.params.get("max_iter", 1)

        for name_tag in test_seeds.keys():
            current_com = set(community_pred_before[name_tag])
            if not current_com:
                community_pred_after[name_tag] = set()
                continue

            for iter_idx in range(max_iter):
                embeddings = []
                for node in current_com:
                    emb = test_g.singleNodeEmbed(node)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    embeddings.append(emb)
                com_emb = np.mean(embeddings, axis=0)
                norm_com = np.linalg.norm(com_emb)

                best_node = None
                best_sim = -1.0
                for node in current_com:
                    emb = test_g.singleNodeEmbed(node)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    norm_emb = np.linalg.norm(emb)
                    if norm_emb == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(com_emb, emb) / (norm_com * norm_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_node = node

                if best_node is None:
                    break

                with torch.no_grad():
                    preds, _, _ = expander.sample_bs_trajectories([best_node])
                new_nodes = set()
                for p in preds:
                    new_nodes.update([n for n in p if n != "Stp"])

                if new_nodes.issubset(current_com):
                    print(f"社区 {name_tag} 迭代 {iter_idx+1} 轮后无新增节点，提前停止")
                    break
                current_com.update(new_nodes)

            community_pred_after[name_tag] = current_com

        # ================== 计算指标：新增 Jaccard ==================
        metrics_before = {"precision": [], "recall": [], "f1": [], "jaccard": []}
        metrics_after = {"precision": [], "recall": [], "f1": [], "jaccard": []}

        for name_tag, true_addr in true_coms:
            # 扩展前
            pred_before = list(community_pred_before.get(name_tag, set()))
            p_b, r_b, f1_b = eval_scores(pred_before, true_addr)
            j_b = len(set(pred_before) & set(true_addr)) / len(set(pred_before) | set(true_addr)) if len(set(pred_before) | set(true_addr)) > 0 else 0.0

            metrics_before["precision"].append(p_b)
            metrics_before["recall"].append(r_b)
            metrics_before["f1"].append(f1_b)
            metrics_before["jaccard"].append(j_b)

            # 扩展后
            pred_after = list(community_pred_after.get(name_tag, set()))
            p_a, r_a, f1_a = eval_scores(pred_after, true_addr)
            j_a = len(set(pred_after) & set(true_addr)) / len(set(pred_after) | set(true_addr)) if len(set(pred_after) | set(true_addr)) > 0 else 0.0

            metrics_after["precision"].append(p_a)
            metrics_after["recall"].append(r_a)
            metrics_after["f1"].append(f1_a)
            metrics_after["jaccard"].append(j_a)

        def safe_mean(values: List[float]) -> float:
            return round(np.mean(values) if values else 0.0, 4)

        def safe_std(values: List[float]) -> float:
            return round(np.std(values) if len(values) > 1 else 0.0, 4)

        def safe_percent_change(new_val: float, old_val: float) -> float:
            if old_val == 0:
                return 100.0 if new_val > 0 else 0.0
            return round((new_val - old_val) / old_val * 100, 2)

        # 平均值
        before_avg_precision = safe_mean(metrics_before["precision"])
        before_avg_recall = safe_mean(metrics_before["recall"])
        before_avg_f1 = safe_mean(metrics_before["f1"])
        before_avg_jaccard = safe_mean(metrics_before["jaccard"])

        after_avg_precision = safe_mean(metrics_after["precision"])
        after_avg_recall = safe_mean(metrics_after["recall"])
        after_avg_f1 = safe_mean(metrics_after["f1"])
        after_avg_jaccard = safe_mean(metrics_after["jaccard"])

        # 标准差
        before_std_f1 = safe_std(metrics_before["f1"])
        after_std_f1 = safe_std(metrics_after["f1"])
        before_std_jaccard = safe_std(metrics_before["jaccard"])
        after_std_jaccard = safe_std(metrics_after["jaccard"])

        # 变化率
        f1_imp = safe_percent_change(after_avg_f1, before_avg_f1)
        recall_imp = safe_percent_change(after_avg_recall, before_avg_recall)
        precision_change = safe_percent_change(after_avg_precision, before_avg_precision)
        jaccard_change = safe_percent_change(after_avg_jaccard, before_avg_jaccard)

        avg_metrics = {
            "before_avg_precision": before_avg_precision,
            "before_avg_recall": before_avg_recall,
            "before_avg_f1": before_avg_f1,
            "before_avg_jaccard": before_avg_jaccard,
            "before_std_f1": before_std_f1,
            "before_std_jaccard": before_std_jaccard,

            "after_avg_precision": after_avg_precision,
            "after_avg_recall": after_avg_recall,
            "after_avg_f1": after_avg_f1,
            "after_avg_jaccard": after_avg_jaccard,
            "after_std_f1": after_std_f1,
            "after_std_jaccard": after_std_jaccard,

            "f1_improvement": f1_imp,
            "recall_improvement": recall_imp,
            "precision_change": precision_change,
            "jaccard_change": jaccard_change,
        }

        # ================== 打印输出（带 Jaccard） ==================
        print("\n" + "=" * 80)
        print("📊 扩展前后指标对比（模型迭代扩展）")
        print("=" * 80)
        print(f"扩展前 | P: {before_avg_precision} | R: {before_avg_recall} | F1: {before_avg_f1} (±{before_std_f1}) | Jaccard: {before_avg_jaccard} (±{before_std_jaccard})")
        print(f"扩展后 | P: {after_avg_precision} | R: {after_avg_recall} | F1: {after_avg_f1} (±{after_std_f1}) | Jaccard: {after_avg_jaccard} (±{after_std_jaccard})")
        print(f"变化幅度 | P: {precision_change}% | R: {recall_imp}% | F1: {f1_imp}% | Jaccard: {jaccard_change}%")
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

            print(f"\n{'='*70}\n📌 数据集：{dfname}  seed={seed}\n{'='*70}")
            print(f"核心参数：min_community_size={self.params['min_community_size']}, epoch={self.params['epoch']}, seedNum={self.params['seedNum']}")
            print("-" * 70)

            loader = PreprocessedDataLoader(
                dataset_name=dfname,
                train_ratio=0.8,
                min_com_size=self.params["min_community_size"],
                seed=seed
            )

            global_adj = loader.graph['adj']
            global_features = loader.nodefeats
            all_nodes = set(global_adj.keys())

            train_communities = {}
            for idx, comm in enumerate(loader.train_comms):
                train_communities[f"train_{idx}"] = comm

            test_communities = {}
            for idx, comm in enumerate(loader.test_comms):
                test_communities[f"test_{idx}"] = comm

            train_g = Graph(
                adj=global_adj,
                features=global_features,
                communities=train_communities,
                node_list=list(all_nodes)
            )

            test_g = Graph(
                adj=global_adj,
                features=global_features,
                communities=test_communities,
                node_list=list(all_nodes)
            )

            print(f"训练图：节点数 {train_g.n_nodes}，特征维度 {train_g.embedsize}，训练社区数 {len(train_communities)}")
            print(f"测试图：节点数 {test_g.n_nodes}，特征维度 {test_g.embedsize}，测试社区数 {len(test_communities)}")

            device = torch.device(self.params["device"])
            model = Agent(
                input_size=train_g.embedsize,
                hidden_size=self.params["hidden_size"],
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
                repeat_penalty_coeff=self.params["repeat_penalty_coeff"],
                entropy_coeff=self.params["entropy_coeff"],
                grad_norm=self.params["grad_norm"],
                target_recall=self.params["target_recall"],
                stop_reward_scale=self.params["stop_reward_scale"],
            )

          
            epoch = self.params["epoch"]
            seedNum = self.params["seedNum"]
            coms=list(train_g.communities.values())


            print(f"\n🚀 开始训练：{epoch}轮 | 每轮采样{seedNum}种子")
            for i in range(epoch):
                true_coms=random.sample(coms,k=seedNum)
                seeds=[]
                for c in true_coms:
                    seeds.append(random.choice(c))
                
                loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
                print(f"📝 Epoch {i+1}/{epoch} | loss: {loss:.4f}")

            print(f"\n{'='*70}\n🧪 开始测试（真实社区+动态采样种子）\n{'='*70}")
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