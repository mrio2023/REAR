import random
import torch
import numpy as np
import os
import sys
import time
from typing import Dict, List, Tuple, Set, Any
from .dataProcess import dataProcess
from .graph import Graph
from .Agent import Agent
from .expander import Expander
from .tool import eval_scores, eval_f1

class Tee:
    """将输出同时写入文件和终端（代码保持不变）"""
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
                - dfname, normal_node_ratio, expand_hop, min_community_size,
                  device, hidden_size, lr, maxLen, gamma, f1_base_weight,
                  p_bias, min_f1_threshold, len_penalty_coeff, epoch, seedNum
                - 可选: max_iter (默认2), target_recall (默认0.8), stop_reward_scale (默认2.0),
                        entropy_coeff (默认0.01), grad_norm (默认1.0), r_bias (默认1),
                        repeat_penalty_coeff (默认0.5)
        """
        required_params = [
            "dfname", "normal_node_ratio", "expand_hop", "min_community_size",
            "device", "hidden_size", "lr", "maxLen", "gamma", "f1_base_weight",
            "p_bias", "min_f1_threshold", "len_penalty_coeff", "epoch", "seedNum"
        ]
        for param in required_params:
            if param not in params:
                raise ValueError(f"缺少必要参数: {param}")
        
        self.params = params
        # 设置默认值
        self.params.setdefault("max_iter", 2)
        self.params.setdefault("target_recall", 0.8)
        self.params.setdefault("stop_reward_scale", 2.0)
        self.params.setdefault("entropy_coeff", 0.01)
        self.params.setdefault("grad_norm", 1.0)
        self.params.setdefault("r_bias", 1)
        self.params.setdefault("repeat_penalty_coeff", 0.5)

    def eval_model(self, expander: Expander, test_g: Graph) -> Dict[str, float]:
        """评估模型性能（已适配 sample_bs_trajectories 返回三个值）"""
        expander.model.eval()

        true_coms: List[Tuple[str, List]] = [
            (tag, list(addr_set))
            for tag, addr_set in test_g.community_seeds.items()
            if isinstance(addr_set, (set, list)) and len(addr_set) > 0
        ]
        print(f"有 {len(true_coms)} 个社区")

        test_seeds: Dict[str, List] = {}
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

        all_seeds_flat: List = []
        community_map: List[str] = []
        for name_tag, seeds in test_seeds.items():
            all_seeds_flat.extend(seeds)
            community_map.extend([name_tag] * len(seeds))

        pred_coms_flat: List[List] = []
        if all_seeds_flat:
            with torch.no_grad():
                # 修改：接收三个返回值，忽略后两个
                pred_coms_flat, _, _ = expander.sample_bs_trajectories(all_seeds_flat)
            print(f"   推理种子数：{len(all_seeds_flat)}")

        community_pred_before: Dict[str, Set] = {tag: set() for tag in test_seeds.keys()}
        for idx, pred_com in enumerate(pred_coms_flat):
            if idx >= len(community_map):
                continue
            valid_nodes = [n for n in pred_com if n != "Stp"]
            community_pred_before[community_map[idx]].update(valid_nodes)

        # 迭代扩展
        community_pred_after: Dict[str, Set] = {}
        max_iter = self.params["max_iter"]
        
        for name_tag in test_seeds.keys():
            current_com = set(community_pred_before[name_tag])
            if not current_com:
                community_pred_after[name_tag] = set()
                continue

            for iter_idx in range(max_iter):
                seeds = list(current_com)
                if not seeds:
                    break
                    
                with torch.no_grad():
                    # 修改：接收三个返回值，忽略后两个
                    preds, _, _ = expander.sample_bs_trajectories(seeds)
                
                new_nodes = set()
                for p in preds:
                    new_nodes.update([n for n in p if n != "Stp"])
                
                if new_nodes.issubset(current_com):
                    print(f"   社区 {name_tag} 迭代 {iter_idx+1} 轮后无新增节点，提前停止")
                    break
                
                current_com.update(new_nodes)

            community_pred_after[name_tag] = current_com

        # 计算指标（后续代码不变）
        metrics_before = {"precision": [], "recall": [], "f1": []}
        metrics_after = {"precision": [], "recall": [], "f1": []}
        for name_tag, true_addr in true_coms:
            pred_before = list(community_pred_before.get(name_tag, set()))
            p_b, r_b, f1_b = eval_scores(pred_before, true_addr)
            metrics_before["precision"].append(p_b)
            metrics_before["recall"].append(r_b)
            metrics_before["f1"].append(f1_b)

            pred_after = list(community_pred_after.get(name_tag, set()))
            p_a, r_a, f1_a = eval_scores(pred_after, true_addr)
            metrics_after["precision"].append(p_a)
            metrics_after["recall"].append(r_a)
            metrics_after["f1"].append(f1_a)

        def safe_mean(values: List[float]) -> float:
            return round(np.mean(values) if values else 0.0, 4)
        
        def safe_std(values: List[float]) -> float:
            return round(np.std(values) if len(values) > 1 else 0.0, 4)
        
        def safe_percent_change(new_val: float, old_val: float) -> float:
            if old_val == 0:
                return 100.0 if new_val > 0 else 0.0
            return round((new_val - old_val) / old_val * 100, 2)

        before_avg_f1 = safe_mean(metrics_before["f1"])
        before_avg_recall = safe_mean(metrics_before["recall"])
        before_avg_precision = safe_mean(metrics_before["precision"])
        
        after_avg_f1 = safe_mean(metrics_after["f1"])
        after_avg_recall = safe_mean(metrics_after["recall"])
        after_avg_precision = safe_mean(metrics_after["precision"])

        avg_metrics = {
            "before_avg_precision": before_avg_precision,
            "before_avg_recall": before_avg_recall,
            "before_avg_f1": before_avg_f1,
            "before_std_f1": safe_std(metrics_before["f1"]),
            "after_avg_precision": after_avg_precision,
            "after_avg_recall": after_avg_recall,
            "after_avg_f1": after_avg_f1,
            "after_std_f1": safe_std(metrics_after["f1"]),
            "f1_improvement": safe_percent_change(after_avg_f1, before_avg_f1),
            "recall_improvement": safe_percent_change(after_avg_recall, before_avg_recall),
            "precision_change": safe_percent_change(after_avg_precision, before_avg_precision),
        }

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

    def run(self, dfname: str, seed: int) -> Dict[str, float]:
        """执行完整的训练和测试流程（已适配Expander新参数）"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available() and self.params["device"] == "cuda":
            torch.cuda.manual_seed(seed)

        log_dir = "logs"
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
            print(
                f"核心参数：\n  normal_node_ratio: {self.params['normal_node_ratio']} | expand_hop: {self.params['expand_hop']}\n  "
                f"epoch: {self.params['epoch']} | seedNum: {self.params['seedNum']}"
            )
            print("-" * 70)

            dp = dataProcess(
                dfname=self.params["dfname"],
                normal_node_ratio=self.params["normal_node_ratio"],
                expand_hop=self.params["expand_hop"],
                min_community_size=self.params["min_community_size"],
            )

            print(
                f"数据规模：训练节点{len(dp.train_nodes)} | 训练黑客{len(dp.train_hacker)} | 测试节点{len(dp.test_nodes)} | 测试黑客{len(dp.test_hacker)}"
            )

            train_g = Graph(
                dfnode=dp.train_nodes,
                dffeature=dp.train_feature,
                dfhacker=dp.train_hacker,
                dfedge=dp.train_edge,
            )

            device = torch.device(self.params["device"])
            model = Agent(
                input_size=train_g.embedsize,
                hidden_size=self.params["hidden_size"],
            ).to(device)

            # 创建Expander时传入所有参数（包括新增的）
            expander = Expander(
                graph=train_g,
                model=model,
                optimizer=torch.optim.Adam(model.parameters(), lr=self.params["lr"]),
                device=device,
                maxLen=self.params["maxLen"],
                gamma=self.params["gamma"],
                f1_base_weight=self.params["f1_base_weight"],
                p_bias=self.params["p_bias"],
                min_f1_threshold=self.params["min_f1_threshold"],
                len_penalty_coeff=self.params["len_penalty_coeff"],
                r_bias=self.params["r_bias"],
                repeat_penalty_coeff=self.params["repeat_penalty_coeff"],
                entropy_coeff=self.params["entropy_coeff"],
                grad_norm=self.params["grad_norm"],
                target_recall=self.params["target_recall"],
                stop_reward_scale=self.params["stop_reward_scale"],
            )

            origin_seeds = dp.train_hacker["address"].tolist()
            epoch = self.params["epoch"]
            seedNum = self.params["seedNum"]
            
            if len(origin_seeds) < seedNum:
                raise ValueError(f"训练种子数量 {len(origin_seeds)} 小于每轮采样数 {seedNum}")
            
            print(f"\n🚀 开始训练：{epoch}轮 | 每轮采样{seedNum}种子")
            for i in range(epoch):
                seeds = random.sample(origin_seeds, k=seedNum)
                true_coms = [
                    train_g.sampleTrajectory(s, maxlen=self.params["maxLen"]) for s in seeds
                ]
                loss = expander.trainReward(seeds=seeds, true_coms=true_coms)
                print(f"📝 Epoch {i+1}/{epoch} | loss: {loss:.4f}")

            print(f"\n{'='*70}\n🧪 开始测试（真实社区+动态采样种子）\n{'='*70}")
            test_g = Graph(
                dfnode=dp.test_nodes,
                dffeature=dp.test_feature,
                dfhacker=dp.test_hacker,
                dfedge=dp.test_edge,
            )
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