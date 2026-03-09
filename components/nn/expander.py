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
        f1_base_weight: float,  # F1基础权重（现用于缩放增量）
        p_bias: float,  # 精度倾斜系数（暂未使用，可扩展）
        len_penalty_coeff: float,  # 长度惩罚系数
        min_f1_threshold: float,  # 最小F1阈值（暂未使用）
        r_bias: float = 1,  # 召回倾斜系数（暂未使用）
        repeat_penalty_coeff: float = 0.5,  # 重复选点惩罚系数
        # 新增参数
        entropy_coeff: float = 0.01,  # 熵正则系数
        grad_norm: float = 10.0,  # 梯度裁剪阈值
    ):

        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.device = device or torch.device("cpu")

        self.f1_base_weight = f1_base_weight
        self.p_bias = p_bias
        self.r_bias = r_bias
        self.min_f1_threshold = min_f1_threshold
        self.len_penalty_coeff = len_penalty_coeff
        self.repeat_penalty_coeff = repeat_penalty_coeff
        self.entropy_coeff = entropy_coeff
        self.grad_norm = grad_norm

        print(f"self.r_bias {r_bias}")
        print(f"maxlen: {maxLen}")
        print(f"entropy_coeff: {entropy_coeff}, grad_norm: {grad_norm}")

    def sample_actions(self, logits, training=True):
        """
        根据模式采样动作：
        - 训练模式：按概率采样（探索）
        - 评估模式：贪心选择
        返回：动作列表、对数概率列表、熵列表（仅训练模式有效）
        """
        actions, log_probs, entropies = [], [], []
        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0:
                # 无候选时默认停止，但实际不会走到这里（prepare_inputs已保证有候选）
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
                continue

            dist = torch.distributions.Categorical(logits=batch_logits)
            if training:
                action = dist.sample()
            else:
                action = torch.argmax(dist.probs)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()  # 用于正则项

            # 动作可能是整数或张量，统一转为整数或"Stp"
            if action == len(dist.probs) - 1:  # 最后一个logit对应停止
                actions.append("Stp")
            else:
                actions.append(action.item())
            log_probs.append(log_prob)
            entropies.append(entropy)

        return actions, log_probs, entropies

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """
        准备模型输入：
        - 始终保证至少有一个候选节点（若无邻居，则填充一个虚拟节点，但后续会特殊处理）
        - 返回：种子嵌入、轨迹节点嵌入、indptr、候选节点列表
        """
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        embed_dim = self.graph.embedsize

        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))

            # --- 空邻居处理：填充一个虚拟节点，但后续在奖励计算中会忽略 ---
            if len(unique_neigh) == 0:
                fake_embed = torch.zeros((1, embed_dim), dtype=torch.float32, device=self.device)
                t_tra_vector.append(fake_embed)           # 虚拟邻居（候选）
                t_seed_vector.append(fake_embed)
                t_tra_vector.append(tra_vector[i].unsqueeze(0))  # 当前节点（用于计算停止特征）
                t_seed_vector.append(seed_vector[i].unsqueeze(0))

                choices.append("Stp")  # 标记为无真实候选
                indptr.append((offset, offset + 2, offset + 1))  # 候选只有虚拟节点
                offset += 2
                continue

            # --- 正常邻居处理 ---
            neigh_embed = self.graph.nodesEmbed(unique_neigh)

            # 剪枝
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

            # 转为张量
            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = torch.tensor(
                    np.stack(neigh_embed) if isinstance(neigh_embed, list) else neigh_embed,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                neigh_embed = neigh_embed.to(self.device)

            # 添加邻居嵌入
            for n in neigh_embed:
                t_tra_vector.append(n.unsqueeze(0))
                t_seed_vector.append(n.unsqueeze(0))
            # 添加当前节点嵌入
            t_tra_vector.append(tra_vector[i].unsqueeze(0))
            t_seed_vector.append(seed_vector[i].unsqueeze(0))

            choices.append(unique_neigh)
            indptr.append((offset, offset + 1 + len(unique_neigh), offset + len(unique_neigh)))
            offset += len(unique_neigh) + 1

        # 构建最终张量
        if not t_seed_vector:
            seed_embed = torch.empty((0, embed_dim), device=self.device)
            tra_embed = torch.empty((0, embed_dim), device=self.device)
        else:
            seed_embed = torch.cat(t_seed_vector, dim=0)
            tra_embed = torch.cat(t_tra_vector, dim=0)

        return seed_embed, tra_embed, np.array(indptr), choices

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
        # 注意：不设置requires_grad，因为graph嵌入可能不参与训练
        return embed

    def all_done(self):
        return all(self.done) if hasattr(self, 'done') else False

    def vecpool(self, v1, v2, k):
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        """采样batch轨迹，返回节点序列和对数概率列表（每个step的log_prob）"""
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

        tra_vector = seed_vector.clone()
        tra_nodes = [[s] for s in seeds]
        tra_logps = [[] for _ in range(len(seeds))]
        self.done = [False] * len(seeds)
        step = 0

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break

            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]

            *model_inputs, batch_candidates = self.prepare_inputs(
                active_tra_vector, active_seed_vector, active_tra_nodes
            )
            batch_logits = self.model(*model_inputs)

            # 训练模式下采样（探索），否则贪心（但此方法仅训练时调用，故使用训练模式）
            actions, logps, entropies = self.sample_actions(batch_logits, training=True)

            for j, orig_idx in enumerate(active_indices):
                ac = actions[j]
                logp = logps[j]

                # --- 关键修复：当候选为"Stp"（空邻居）时，重新计算停止动作的log_prob ---
                if batch_candidates[j] == "Stp":
                    # 从batch_logits中提取停止动作的log_prob（最后一个logit）
                    stop_log_prob = F.log_softmax(batch_logits[j], dim=-1)[-1]
                    self.add_node("Stp", tra_nodes, orig_idx)
                    tra_logps[orig_idx].append(stop_log_prob)
                    continue

                # 正常情况：有真实候选
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
        self.model.train()

        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 统计batch指标
        p_list, r_list, f1_list, pred_len_list = [], [], [], []
        len_true_sum = 0
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            len_true_sum += len(true_com)
            p, r, f1 = eval_scores(pred_com_clean, true_com)
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            pred_len_list.append(len(pred_com_clean))

        print(f"真实社区平均长度 {len_true_sum/len(true_coms):.3f}")
        batch_recall = np.mean(r_list)
        batch_precision = np.mean(p_list)
        batch_f1 = np.mean(f1_list)
        avg_ext_len = np.mean(pred_len_list)
        early_stop_prob = sum(self.done) / len(self.done) if len(self.done) > 0 else 0.0
        print(f"提前终止的概率 {early_stop_prob:.3f}")
        print(f"Batch Metrics: P={batch_precision:.4f}, R={batch_recall:.4f}, F1={batch_f1:.4f}, AvgExtLen={avg_ext_len:.2f}")

        # ---------- 改进的奖励计算 ----------
        target_recall = 0.8  # 可调整为动态值
        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com = [com[0]]
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)
            repeat_count = 0
            step_rewards = []

            for node in com[1:]:
                # 重复节点惩罚
                if node in temp_com and node != "Stp":
                    repeat_count += 1
                    step_rewards.append(-self.repeat_penalty_coeff * repeat_count)
                    continue

                if node == "Stp":
                    # 停止奖励：基于当前召回与目标召回的差距
                    curr_r = len(set(temp_com) & true_com_set) / true_com_len
                    stop_reward = (curr_r - target_recall) * 5  # 缩放因子可调
                    step_rewards.append(stop_reward)
                    continue

                # 添加节点前的p, r
                pre_intersect = len(set(temp_com) & true_com_set)
                pre_p = pre_intersect / len(set(temp_com)) if temp_com else 0.0
                pre_r = pre_intersect / true_com_len

                temp_com.append(node)

                curr_intersect = len(set(temp_com) & true_com_set)
                curr_p = curr_intersect / len(set(temp_com))
                curr_r = curr_intersect / true_com_len

                recall_inc = curr_r - pre_r
                precision_inc = curr_p - pre_p

                # 加权奖励
                base_reward = (recall_inc * self.r_bias + precision_inc * self.p_bias) * self.f1_base_weight

                # 长度惩罚
                if len(temp_com) > true_com_len:
                    penalty = self.len_penalty_coeff ** (len(temp_com) - true_com_len)
                    base_reward *= penalty

                step_rewards.append(base_reward)

            # 计算折扣回报
            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            else:
                discounted = [0.0]
            rewards_list.append(discounted)

        # 填充到相同长度
        max_len_pad = max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps))
        rewards_padded = np.zeros((bs, max_len_pad))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, :len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        # 可选：去掉标准化，或保留但使用更稳健的baseline
        mask = (torch.arange(max_len_pad, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)).float()
        # 这里暂时去掉标准化，直接使用原始奖励
        # 如果要去掉，注释下面三行
        # valid_rewards = rewards[mask.bool()]
        # if valid_rewards.numel() > 1:
        #     rewards = (rewards - valid_rewards.mean()) / (valid_rewards.std() + 1e-8)

        # 构建log_probs张量
        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) else torch.tensor(0.0, device=self.device)
                for j in range(max_len_pad)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # 策略梯度损失
        pg_loss = -(rewards.detach() * logps * mask).sum()
        loss = pg_loss  # 可加入熵正则

        print(f"Loss Adjust | Base Loss: {pg_loss.item():.4f}")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_val = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
        self.optimizer.step()

        # 监控梯度
        param_mean = torch.mean(
            torch.stack([p.data.mean() for p in self.model.parameters() if p.requires_grad])
        )
        grad_mean = torch.mean(
            torch.stack([
                p.grad.mean() if p.grad is not None else torch.tensor(0.0, device=self.device)
                for p in self.model.parameters() if p.requires_grad
            ])
        )
        print(f"\nGrad Check | Norm: {grad_norm_val:.4f} | Param Mean: {param_mean:.4f} | Grad Mean: {grad_mean:.4f}")

        return loss.item()