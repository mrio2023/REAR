from typing import Union, Optional, List, Set
import numpy as np
import torch
import torch.nn.functional as F
from .tool import eval_f1, eval_scores, pruning


class Expander:
    def __init__(
        self,
        graph,
        model,
        maxLen: int,
        optimizer,
        device: Optional[torch.device],
        # 基础RL参数
        gamma: float,
        # F1奖励核心参数（外部传入，方便调试）
        f1_base_weight: float,  # F1基础权重
        p_bias: float,  # 精度倾斜系数
        len_penalty_coeff: float,  # 长度惩罚系数
        min_f1_threshold: float,  # 最小F1阈值
        r_bias: float = 1,  # 召回倾斜系数（新增）
        repeat_penalty_coeff: float = 0.5,  # 重复选点惩罚系数（新增）
    ):
        
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.device = device or torch.device("cpu")

        # 核心F1奖励参数
        self.f1_base_weight = f1_base_weight
        self.p_bias = p_bias
        self.r_bias = r_bias  # 新增：召回偏向系数
        self.min_f1_threshold = min_f1_threshold
        self.len_penalty_coeff = len_penalty_coeff
        self.repeat_penalty_coeff = repeat_penalty_coeff  # 新增：重复惩罚系数
        print("self.r_bias",r_bias)
        print("maxlen:",maxLen)
    def sample_actions(self, logits):
        """贪心选择概率最高的动作 + 统计停止概率"""
        actions, log_probs = [], []
        stop_count = 0  # 统计停止次数
        # total_count = len(logits)  # 总选点次数

        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(-1e9, device=self.device, requires_grad=True))
                # stop_count += 1
                continue

            dist = torch.distributions.Categorical(logits=batch_logits)
            action = torch.argmax(dist.probs)
            log_prob = dist.log_prob(action)

            log_prob = log_prob.squeeze() if log_prob.dim() > 0 else log_prob
            log_prob.requires_grad_(True)

            action_item = int(action.item()) if isinstance(action, torch.Tensor) else action
            if action_item == "Stp" or action_item >= len(dist.probs):
                actions.append("Stp")
                stop_count += 1
            else:
                actions.append(action_item)
            log_probs.append(log_prob)

        # # 计算并打印停止概率
        # stop_prob = stop_count / total_count if total_count > 0 else 0.0
        # print(f"提前停止概率: {stop_prob:.4f}")  # 比如0.2 → 20%概率停止
        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """准备模型输入：为空邻居填充假样本，保证维度一致"""
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        # 获取嵌入维度（从第一个有效节点的嵌入中提取）
        embed_dim = self.graph.embedsize

        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))

            if len(unique_neigh) <= 0:
                fake_embed = torch.zeros(
                    (1, embed_dim), dtype=torch.float32, device=self.device
                )
                t_tra_vector.append(fake_embed)
                t_seed_vector.append(fake_embed)
                t_tra_vector.append(tra_vector[i].unsqueeze(0))
                t_seed_vector.append(seed_vector[i].unsqueeze(0))

                choices.append("Stp")
                indptr.append((offset, offset + 2, offset + 1))
                offset += 2

            neigh_embed = self.graph.nodesEmbed(unique_neigh)

            pruned_nodes = pruning(
                community_pooled_embed=tra_vector[i],
                neigh_node_embed_list=neigh_embed,
                neigh_nodes=unique_neigh,
            )

            idx_map = {node: j for j, node in enumerate(unique_neigh)}
            keep_idx = [idx_map[n] for n in pruned_nodes]

            if isinstance(neigh_embed, list):
                neigh_embed = [neigh_embed[j] for j in keep_idx]
            else:
                neigh_embed = neigh_embed[keep_idx]

            unique_neigh = pruned_nodes
            # print(len(unique_neigh))
            choices.append(unique_neigh)
            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = torch.tensor(
                    (
                        np.stack(neigh_embed)
                        if isinstance(neigh_embed, list)
                        else neigh_embed
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                neigh_embed = neigh_embed.to(self.device)

            if self.model.training:
                neigh_embed.requires_grad_(True)

            # 1. 记录初始长度
            beforeLen = len(t_tra_vector)

            # 2. 逐个添加所有邻居的嵌入
            for n in neigh_embed:
                v = n.unsqueeze(0)
                t_tra_vector.append(v)
                t_seed_vector.append(v)

            # 3. 添加当前节点的嵌入
            t_tra_vector.append(tra_vector[i].unsqueeze(0))
            t_seed_vector.append(seed_vector[i].unsqueeze(0))

            afterLen = len(t_tra_vector)
            increase = afterLen - beforeLen
            target_increase = len(unique_neigh) + 1

            if increase != target_increase:
                print("===== 嵌入添加数量异常 =====")
                print(f"节点ID: {tra} | 索引: {i}")
                print(f"邻居数量（unique_neigh）: {len(unique_neigh)}")
                print(f"预期新增元素数（邻居+当前节点）: {target_increase}")
                print(f"实际新增元素数: {increase}")
                print(f"初始长度: {beforeLen} | 最终长度: {afterLen}")
                print(f"差值（预期-实际）: {target_increase - increase}")
                raise ValueError(f"节点{tra}嵌入添加数量异常")

            # 更新indptr和offset（非空邻居场景）
            indptr.append(
                (offset, offset + 1 + len(unique_neigh), offset + len(unique_neigh))
            )
            offset += len(unique_neigh) + 1

        # 边界处理：如果所有节点都是空邻居，返回空张量（保持维度一致）
        if not t_seed_vector or not t_tra_vector:
            seed_embed = torch.empty(
                (0, embed_dim), dtype=torch.float32, device=self.device
            )
            tra_embed = torch.empty(
                (0, embed_dim), dtype=torch.float32, device=self.device
            )
        else:
            seed_embed = torch.cat(t_seed_vector, dim=0)
            tra_embed = torch.cat(t_tra_vector, dim=0)

        return (
            seed_embed,
            tra_embed,
            np.array(indptr),
            choices,
        )

    def add_node(self, new_node, tra_nodes, index):
        """添加节点到轨迹，更新done状态"""
        if (
            new_node in (None, "Stp", -1)
            or len(tra_nodes[index]) >= self.maxLen
            or (new_node != "Stp" and new_node in tra_nodes[index])
        ):
            self.done[index] = True
            return None
        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        embed = (
            torch.tensor(embed, dtype=torch.float32, device=self.device)
            if not isinstance(embed, torch.Tensor)
            else embed.to(self.device)
        )
        if self.model.training:
            embed.requires_grad_(True)
        return embed

    def all_done(self):
        """判断是否所有轨迹都完成"""
        return all(self.done) if self.done else False

    def vecpool(self, v1, v2, k):
        """向量池化更新"""
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        seed_vector = self.graph.nodesEmbed(seeds)
        seed_vector = (
            torch.tensor(
                np.stack(seed_vector) if isinstance(seed_vector, list) else seed_vector,
                dtype=torch.float32,
                device=self.device,
            )
            if not isinstance(seed_vector, torch.Tensor)
            else seed_vector.to(self.device)
        )
        if self.model.training:
            seed_vector.requires_grad_(True)

        tra_vector, tra_nodes, tra_logps = (
            seed_vector.clone(),
            [[s] for s in seeds],
            [[] for _ in range(len(seeds))],
        )
        self.done = [False] * len(seeds)
        step = 0

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break
            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]

      
            walk_active_tra_nodes = active_tra_nodes.copy()  
        
       
            *model_inputs, batch_candidates = self.prepare_inputs(
                active_tra_vector, active_seed_vector, walk_active_tra_nodes
            )
            batch_logits = self.model(*model_inputs)
            actions, logps = self.sample_actions(batch_logits)

            for j, orig_idx in enumerate(active_indices):
                ac, logp = actions[j], logps[j]
                if batch_candidates[j] == "Stp":
                    self.add_node("Stp", tra_nodes, orig_idx)
                    tra_logps[orig_idx].append(logp)
                    continue
                if ac == "Stp" or ac >= len(batch_candidates[j]):
                    self.add_node("Stp", tra_nodes, orig_idx)
                    tra_logps[orig_idx].append(logp)
                else:
                    selected_node = batch_candidates[j][ac]
                    if selected_node in tra_nodes[orig_idx]:
                        self.add_node("Stp", tra_nodes, orig_idx)
                        tra_logps[orig_idx].append(logp)
                        continue
                    newvec = self.add_node(selected_node, tra_nodes, orig_idx)
                    if newvec is not None:
                        tra_vector[orig_idx] = self.vecpool(
                            tra_vector[orig_idx], newvec, len(tra_nodes[orig_idx])
                        )
                        tra_logps[orig_idx].append(logp)
            step += 1
        return tra_nodes, tra_logps

    def trainReward(self, seeds: List[int], true_coms):
        """核心训练逻辑：基于batch全局指标调整最终loss（参数仅作用于batch级）"""
        self.model.train()

        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        p_list, r_list, f1_list = [], [], []
        pred_len_list = []
        early_stop_count = 0  # 新增：统计提前终止的样本数
        max_len = self.maxLen  # 取预设的最大扩展长度
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            p, r, f1 = eval_scores(pred_com_clean, true_com)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            pred_len_list.append(len(pred_com_clean))
            
            # 新增：判断是否提前终止（长度<maxLen 且 包含Stp）
            has_stp = "Stp" in pred_com
            is_early_stop = len(pred_com_clean) < max_len and has_stp
            if is_early_stop:
                early_stop_count += 1

        batch_recall = np.mean(r_list)
        batch_precision = np.mean(p_list)
        batch_f1 = np.mean(f1_list)
        avg_ext_len = np.mean(pred_len_list)
        early_stop_prob = early_stop_count / bs if bs > 0 else 0.0  # 提前终止概率

        # 打印新增提前终止概率
        print(f"Batch Metrics: P={batch_precision:.4f}, R={batch_recall:.4f}, F1={batch_f1:.4f}, AvgExtLen={avg_ext_len:.2f}, EarlyStopProb={early_stop_prob:.4f}")

        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com, step_rewards = [com[0]], []
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)
            repeat_count = 0

            for node in com[1:]:
                if node in temp_com and node != "Stp":
                    repeat_count += 1
                    repeat_penalty = -self.repeat_penalty_coeff * repeat_count
                    step_rewards.append(repeat_penalty)
                    continue

                if node == "Stp":
                    step_rewards.append(-0.1)
                    continue

                pre_f1 = eval_f1(temp_com, true_com_set)
                temp_com.append(node)
                curr_f1 = eval_f1(temp_com, true_com_set)

                if curr_f1 > pre_f1 and curr_f1 > self.min_f1_threshold:
                    base_reward = curr_f1 * self.f1_base_weight
                else:
                    base_reward = -(1 - curr_f1) * self.f1_base_weight

                curr_pred_len = len(temp_com)
                if curr_pred_len > true_com_len:
                    base_reward *= self.len_penalty_coeff ** (curr_pred_len - true_com_len)

                base_reward = np.clip(base_reward, -1.0, 1.0)
                step_rewards.append(base_reward)

            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            else:
                discounted = [0.0]
            rewards_list.append(discounted)

        max_len_pad = max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps)) if (rewards_list and logps) else 0
        rewards_padded = np.zeros((bs, max_len_pad))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, : len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) and lp_list[j] is not None 
                else torch.tensor(-1e9, device=self.device, requires_grad=True)
                for j in range(max_len_pad)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        mask = (torch.arange(max_len_pad, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)).float()

        recall_low = 0.4
        recall_mid = 0.7
        recall_high = 0.9

        if batch_recall < recall_low:
            loss_weight = 3.0
            recall_bonus = 0.0
        elif batch_recall < recall_mid:
            loss_weight = 2.0
            recall_bonus = 0.1
        elif batch_recall < recall_high:
            loss_weight = 1.5
            recall_bonus = 0.2
        else:
            loss_weight = 1.0
            recall_bonus = 0.3

        base_loss = -(rewards.detach() * logps * mask).sum()
        final_loss = (base_loss * loss_weight) - (recall_bonus * bs)

        print(f"Loss Adjust | Base Loss: {base_loss.item():.4f}, Weight: {loss_weight:.1f}, Bonus: {recall_bonus:.1f}, Final Loss: {final_loss.item():.4f}")

        self.optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        param_mean = torch.mean(torch.stack([p.data.mean() for p in self.model.parameters() if p.requires_grad]))
        grad_mean = torch.mean(torch.stack([p.grad.mean() if p.grad is not None else torch.tensor(0.0, device=self.device) for p in self.model.parameters() if p.requires_grad]))
        print(f"\nGrad Check | Norm: {grad_norm:.4f} | Param Mean: {param_mean:.4f} | Grad Mean: {grad_mean:.4f}")
        print(f"Final Loss: {final_loss.item():.4f}")

        return final_loss.item()