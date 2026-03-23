from typing import Optional, List
import numpy as np
import torch
from .tool import eval_scores, pruning


class Expander:
    def __init__(
        self,
        graph,
        model,
        maxLen: int,
        optimizer,
        device: Optional[torch.device],
        gamma: float,
        f1_base_weight: float,
        p_bias: float,
        r_bias: float,
        repeat_penalty_coeff: float,
        entropy_coeff: float,
        grad_norm: float,
        target_recall: float,
        stop_reward_scale: float,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.device = device

        self.f1_base_weight = f1_base_weight
        self.p_bias = p_bias
        self.r_bias = r_bias
        self.repeat_penalty_coeff = repeat_penalty_coeff
        self.entropy_coeff = entropy_coeff
        self.grad_norm = grad_norm
        self.target_recall = target_recall
        self.stop_reward_scale = stop_reward_scale

        print(f"初始化 Expander: maxLen={maxLen}, gamma={gamma}, f1_base_weight={f1_base_weight},device={self.device}")
        print(f"p_bias={p_bias}, r_bias={r_bias}, repeat_penalty={repeat_penalty_coeff}")
        print(f"entropy_coeff={entropy_coeff}, grad_norm={grad_norm}")
        print(f"target_recall={target_recall}, stop_reward_scale={stop_reward_scale}")

    def sample_actions(self, logits, training=True):
        """
        根据模式采样动作：
        - 训练模式：按概率采样（探索）
        - 评估模式：贪心选择
        参数：
            logits: 列表，每个元素是一个一维张量 [num_actions]
            training: 是否训练模式
        返回：
            actions: 列表，每个元素是采样到的动作（整数或"Stp"）
            log_probs: 列表，每个元素是动作的对数概率张量
            entropies: 列表，每个元素是动作分布的熵张量
        """
        actions, log_probs, entropies = [], [], []

        for i, batch_logits in enumerate(logits):
            if batch_logits is None or batch_logits.numel() == 0:
                print(f"[WARN] Empty logits at batch index {i}，视为停止")
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device))
                entropies.append(torch.tensor(0.0, device=self.device))
                continue

            # 防御性处理 NaN 和极端值
            if torch.isnan(batch_logits).any():
                print(f"[ERROR] NaN detected in logits at batch index {i}")
                batch_logits = torch.nan_to_num(batch_logits, nan=0.0)
            batch_logits = torch.clamp(batch_logits, min=-20, max=20)

            dist = torch.distributions.Categorical(logits=batch_logits)

            if training:
                action = dist.sample()
            else:
                action = torch.argmax(dist.probs)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # 假设停止动作是最后一个动作
            if action == len(dist.probs) - 1:
                actions.append("Stp")
            else:
                actions.append(action.item())

            log_probs.append(log_prob)
            entropies.append(entropy)

        return actions, log_probs, entropies

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """
        准备模型输入：
        - 若无邻居，则候选集为空，仅保留停止动作
        - 返回：种子嵌入、轨迹节点嵌入、indptr、候选节点列表
        """
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        embed_dim = self.graph.embedsize

        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))

            # --- 空邻居处理：无候选，仅停止动作 ---
            if len(unique_neigh) == 0:
                t_tra_vector.append(tra_vector[i].unsqueeze(0))
                t_seed_vector.append(seed_vector[i].unsqueeze(0))
                choices.append([])
                indptr.append((offset, offset + 1, offset))
                offset += 1
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
            if torch.isnan(seed_embed).any() or torch.isnan(tra_embed).any():
                print("[ERROR] NaN in input embeddings!")
                # 在 prepare_inputs 的最后，return 之前添加：
        if torch.isnan(seed_embed).any() or torch.isnan(tra_embed).any():
            print("[ERROR] NaN in seed_embed or tra_embed")
            # 可选：打印出有问题的索引
            nan_seed_idx = torch.isnan(seed_embed).any(dim=1).nonzero().squeeze()
            print(f"NaN in seed_embed at indices: {nan_seed_idx}")
            # 可考虑用零替换
            seed_embed = torch.nan_to_num(seed_embed, nan=0.0)
            tra_embed = torch.nan_to_num(tra_embed, nan=0.0)

        return seed_embed, tra_embed, np.array(indptr), choices

    def add_node(self, new_node, tra_nodes, tra_sets, index):
        """添加节点到轨迹，更新done状态，返回新节点嵌入（如果添加成功）"""
        if (
            new_node in (None, "Stp", -1)
            or len(tra_nodes[index]) >= self.maxLen
            or (new_node != "Stp" and new_node in tra_sets[index])
        ):
            # 重复节点不标记 done，允许继续，但返回 None
            if new_node != "Stp" and new_node in tra_sets[index]:
                return None
            self.done[index] = True
            return None
        tra_nodes[index].append(new_node)
        tra_sets[index].add(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        embed = (
            torch.tensor(embed, dtype=torch.float32, device=self.device)
            if not isinstance(embed, torch.Tensor)
            else embed.to(self.device)
        )
        return embed

    def all_done(self):
        return all(self.done) if hasattr(self, 'done') else False

    def vecpool(self, v1, v2, k):
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        """采样batch轨迹，返回节点序列、对数概率列表和熵列表（每个step的log_prob和entropy）"""
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
        tra_sets = [{s} for s in seeds]
        tra_logps = [[] for _ in range(len(seeds))]
        tra_entropies = [[] for _ in range(len(seeds))]
        self.done = [False] * len(seeds)
        step = 0

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break

            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]
            active_tra_sets = [tra_sets[i] for i in active_indices]

            *model_inputs, batch_candidates = self.prepare_inputs(
                active_tra_vector, active_seed_vector, active_tra_nodes
            )
            batch_logits = self.model(*model_inputs)

            actions, logps, entropies = self.sample_actions(batch_logits, training=False)

            for j, orig_idx in enumerate(active_indices):
                ac = actions[j]
                logp = logps[j]
                entropy = entropies[j]

                # 处理空邻居情况：候选列表为空，动作必然为停止
                if batch_candidates[j] == []:
                    self.add_node("Stp", tra_nodes, tra_sets, orig_idx)
                    tra_logps[orig_idx].append(logp)
                    tra_entropies[orig_idx].append(entropy)
                    continue

                if ac == "Stp" or ac >= len(batch_candidates[j]):
                    self.add_node("Stp", tra_nodes, tra_sets, orig_idx)
                    tra_logps[orig_idx].append(logp)
                    tra_entropies[orig_idx].append(entropy)
                else:
                    selected_node = batch_candidates[j][ac]
                    newvec = self.add_node(selected_node, tra_nodes, tra_sets, orig_idx)
                    if newvec is not None:
                        tra_vector[orig_idx] = self.vecpool(
                            tra_vector[orig_idx], newvec, len(tra_nodes[orig_idx])
                        )
                    # 无论是否重复，都记录logp和entropy
                    tra_logps[orig_idx].append(logp)
                    tra_entropies[orig_idx].append(entropy)
            step += 1

        return tra_nodes, tra_logps, tra_entropies

    def trainReward(self, seeds: List[int], true_coms):
        self.model.train()

        selected_nodes, logps, entropies_list = self.sample_bs_trajectories(seeds)
                # 在 sample_bs_trajectories 之后
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in param: {name}")
        # 检查种子嵌入
        seed_embed = self.graph.nodesEmbed(seeds)
        if np.isnan(seed_embed).any():
            print("NaN in seed_embed")
        # 在 prepare_inputs 中检查 tra_vector 和 seed_vector
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 统计batch指标（仅用于监控）
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

        # ========== 改进的奖励计算 ==========
        rewards_list = []
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com = [com[0]]
            temp_set = {com[0]}
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)
            repeat_count = 0
            step_rewards = []

            for node in com[1:]:
                # 重复节点惩罚（不停止，只给负奖励）
                if node != "Stp" and node in temp_set:
                    repeat_count += 1
                    step_rewards.append(-self.repeat_penalty_coeff * repeat_count)
                    continue  # 不加入节点，但继续循环

                if node == "Stp":
                    # 停止奖励：基于当前召回与目标召回的差距
                    curr_r = len(temp_set & true_com_set) / true_com_len
                    stop_reward = (curr_r - self.target_recall) * self.stop_reward_scale
                    step_rewards.append(stop_reward)
                    continue

                # 添加节点前的p, r
                pre_intersect = len(temp_set & true_com_set)
                pre_p = pre_intersect / len(temp_set) if temp_set else 0.0
                pre_r = pre_intersect / true_com_len

                temp_com.append(node)
                temp_set.add(node)

                curr_intersect = len(temp_set & true_com_set)
                curr_p = curr_intersect / len(temp_set)
                curr_r = curr_intersect / true_com_len

                recall_inc = curr_r - pre_r
                precision_inc = curr_p - pre_p

                # 加权奖励
                base_reward = (recall_inc * self.r_bias + precision_inc * self.p_bias) * self.f1_base_weight

                # # 长度惩罚（线性更温和）
                # if len(temp_set) > true_com_len:
                #     penalty = max(1.0 - 0.1 * (len(temp_set) - true_com_len), 0.0)
                #     base_reward *= penalty

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

        # 构建mask（有效步数 = 轨迹长度-1，因为种子无动作）
        mask = (torch.arange(max_len_pad, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)).float()

        # 构建log_probs张量
        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) else torch.tensor(0.0, device=self.device)
                for j in range(max_len_pad)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # 构建熵张量（用于正则）
        entropies_padded = []
        for e_list in entropies_list:
            padded = [
                e_list[j] if j < len(e_list) else torch.tensor(0.0, device=self.device)
                for j in range(max_len_pad)
            ]
            entropies_padded.append(torch.stack(padded))
        entropies = torch.stack(entropies_padded)

        # 策略梯度损失
        pg_loss = -(rewards.detach() * logps * mask).sum()
        # 熵正则（鼓励探索）
        avg_entropy = (entropies * mask).sum() / mask.sum()
        loss = pg_loss - self.entropy_coeff * avg_entropy

        print(f"Loss | PG: {pg_loss.item():.4f}, Entropy: {avg_entropy.item():.4f}, Total: {loss.item():.4f}")

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