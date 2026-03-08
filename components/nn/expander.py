from typing import Union, Optional, List, Set
import numpy as np
import torch
import torch.nn.functional as F


class Expander:
    def __init__(
        self,
        graph,
        model,
        maxLen: int,
        optimizer,
        device: Optional[torch.device] = None,
        # 基础RL参数
        gamma: float = 0.99,
        # F1奖励核心参数（外部传入，方便调试）
        f1_base_weight: float = 1.0,  # F1基础权重
        p_bias: float = 0.2,  # 精度倾斜系数
        min_f1_threshold: float = 0.1,  # 最小F1阈值
        len_penalty_coeff: float = 0,  # 长度惩罚系数
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
        self.min_f1_threshold = min_f1_threshold
        self.len_penalty_coeff = len_penalty_coeff

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """计算P/R/F1（核心评估指标）"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        f1 = 2 * p * r / (p + r + 1e-9)
        return round(p, 4), round(r, 4), round(f1, 4)

    def eval_f1(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """单独计算F1（奖励核心）"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        return 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0

    def sample_actions(self, logits):
        """贪心选择概率最高的动作"""
        actions, log_probs = [], []
        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(
                    torch.tensor(-1e9, device=self.device, requires_grad=True)
                )
                continue

            # 贪心选argmax
            dist = torch.distributions.Categorical(logits=batch_logits)
            action = torch.argmax(dist.probs)
            log_prob = dist.log_prob(action)

            log_prob = log_prob.squeeze() if log_prob.dim() > 0 else log_prob
            log_prob.requires_grad_(True)

            action_item = int(action.item()) if isinstance(action, torch.Tensor) else action
            actions.append(action_item if action_item != "Stp" else "Stp")
            log_probs.append(log_prob)

        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """准备模型输入：为空邻居填充假样本，保证维度一致"""
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        # 获取嵌入维度（从第一个有效节点的嵌入中提取）
        embed_dim = tra_vector[0].shape[-1] if tra_vector else 64  # 兜底维度，可根据实际调整
        
        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))  

            # ========== 核心修改：空邻居处理逻辑 ==========
            if len(unique_neigh) <= 0:
                # 1. 填充假嵌入（全0张量，维度和正常嵌入一致）
                fake_embed = torch.zeros((1, embed_dim), dtype=torch.float32, device=self.device)
                t_tra_vector.append(fake_embed)
                t_seed_vector.append(fake_embed)
                # 添加当前节点的真实嵌入（保证offset递增逻辑一致）
                t_tra_vector.append(tra_vector[i].unsqueeze(0))
                t_seed_vector.append(seed_vector[i].unsqueeze(0))
                
                # 2. 标记choices为Stp，indptr正常记录偏移（保证长度一致）
                choices.append("Stp")
                indptr.append((offset, offset + 2, offset + 1))  # 假邻居+当前节点，共2个元素
                offset += 2  # 偏移量同步增加
                continue
            # ========== 空邻居处理结束 ==========

            # 非空邻居的正常逻辑
            choices.append(unique_neigh)
            neigh_embed = self.graph.nodesEmbed(unique_neigh)

            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = torch.tensor(
                    np.stack(neigh_embed) if isinstance(neigh_embed, list) else neigh_embed,
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
            seed_embed = torch.empty((0, embed_dim), dtype=torch.float32, device=self.device)
            tra_embed = torch.empty((0, embed_dim), dtype=torch.float32, device=self.device)
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
        """批量采样轨迹"""
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

            *model_inputs, batch_candidates = self.prepare_inputs(
                active_tra_vector, active_seed_vector, active_tra_nodes
            )
            batch_logits = self.model(*model_inputs)
            actions, logps = self.sample_actions(batch_logits)

            for j, orig_idx in enumerate(active_indices):
                ac, logp = actions[j], logps[j]
                # ========== 增强空邻居的采样跳过逻辑 ==========
                if batch_candidates[j] == "Stp":
                    self.add_node("Stp", tra_nodes, orig_idx)
                    tra_logps[orig_idx].append(logp)
                    continue
                # ========== 原有逻辑 ==========
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
        """核心训练逻辑：修复执行顺序+验证梯度流向"""
        self.model.train()
     
        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 1. 计算P/R/F1指标（仅打印）
        avg_metrics = {}
        p_list, r_list, f1_list = [], [], []
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            p, r, f1 = self.eval_scores(pred_com_clean, true_com)
            p_list.append(p), r_list.append(r), f1_list.append(f1)

        avg_metrics = {
            "avg_precision": round(np.mean(p_list), 4),
            "avg_recall": round(np.mean(r_list), 4),
            "avg_f1": round(np.mean(f1_list), 4),
        }
        print(
            f"Batch Metrics: P={avg_metrics['avg_precision']}, R={avg_metrics['avg_recall']}, F1={avg_metrics['avg_f1']}"
        )

        # 2. F1奖励计算（转torch张量，保留数值）
        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com, step_rewards = [com[0]], []
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)

            for node in com[1:]:
                if node == "Stp" or node in temp_com:
                    continue

                # 计算选节点前后的F1（转torch张量）
                pre_f1 = self.eval_f1(temp_com, true_com_set)
                temp_com.append(node)
                curr_f1 = self.eval_f1(temp_com, true_com_set)
                curr_p, curr_r, _ = self.eval_scores(temp_com, true_com_set)

                # 精度倾斜+增量奖励+长度惩罚
                if curr_p < curr_r:
                    biased_f1 = curr_f1 * (1 + self.p_bias)
                else:
                    biased_f1 = curr_f1

                if curr_f1 > pre_f1 and curr_f1 > self.min_f1_threshold:
                    base_reward = biased_f1 * self.f1_base_weight
                else:
                    base_reward = 0.0

                curr_pred_len = len(temp_com)
                if curr_pred_len > true_com_len:
                    base_reward *= self.len_penalty_coeff ** (
                        curr_pred_len - true_com_len
                    )

                base_reward = np.clip(base_reward, 0.0, self.f1_base_weight)
                step_rewards.append(base_reward)

            # 折扣奖励
            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            rewards_list.append(discounted if discounted else [0.0])

        # 3. 奖励填充（转torch张量）
        max_len = (
            max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps))
            if (rewards_list and logps)
            else 0
        )
        rewards_padded = np.zeros((bs, max_len))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, : len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        # 4. logps填充
        logps_padded = []
        for lp_list in logps:
            padded = [
                (
                    lp_list[j]
                    if j < len(lp_list) and lp_list[j] is not None
                    else torch.tensor(-1e9, device=self.device, requires_grad=True)
                )
                for j in range(max_len)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # 5. Mask计算
        mask = (
            torch.arange(max_len, device=self.device).expand(bs, -1)
            < (lengths - 1).unsqueeze(1)
        ).float()

        # 梯度计算与更新
        rewards_detach = rewards.detach()
        self.optimizer.zero_grad(set_to_none=True)
        loss = -(rewards_detach * logps * mask).sum()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        # 打印关键指标
        param_mean = torch.mean(
            torch.stack(
                [p.data.mean() for p in self.model.parameters() if p.requires_grad]
            )
        )
        grad_mean = torch.mean(
            torch.stack(
                [
                    (
                        p.grad.mean()
                        if p.grad is not None
                        else torch.tensor(0.0, device=self.device)
                    )
                    for p in self.model.parameters()
                    if p.requires_grad
                ]
            )
        )
        print(
            f"\nGrad Check | Total Grad Norm: {grad_norm:.4f} | Param Mean: {param_mean:.4f} | Grad Mean: {grad_mean:.4f}"
        )
        print(f"loss: {loss.item():.4f}")
        return loss.item()