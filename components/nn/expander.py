from typing import Union, Optional, List, Set
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
class Expander:
    def __init__(
        self,
        graph,
        model,
        maxLen: int,  
        optimizer,
        device: Optional[torch.device] = None,
        gamma: float = 0.99,
        reward_weight_abs: float = 0.7,
        reward_weight_delta: float = 0.3,
        len_penalty_base: float = 0.99,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.device = device or torch.device("cpu")
        self.reward_weight_abs = reward_weight_abs
        self.reward_weight_delta = reward_weight_delta
        self.len_penalty_base = len_penalty_base

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        f1 = 2 * p * r / (p + r + 1e-9)
        return round(p, 4), round(r, 4), round(f1, 4)

    def eval_f1(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        return 2 * p * r / (p + r + 1e-9) if (p + r) > 0 else 0.0

    def sample_actions(self, logits):
        actions, log_probs = [], []
        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(-1e9, device=self.device, requires_grad=True))
                continue
            
            # 统一logits维度
            if batch_logits.dim() > 1:
                batch_logits = batch_logits.squeeze()
                batch_logits = batch_logits.mean(dim=-1) if batch_logits.dim() > 1 else batch_logits
            
            # ========== 核心修改：贪心选择概率最高的动作 ==========
            try:
                # 方法1：直接从logits选argmax（等价于选概率最高的动作）
                dist = torch.distributions.Categorical(logits=batch_logits)
                action = torch.argmax(dist.probs)  # 选概率最高的动作
                log_prob = dist.log_prob(action)   # 计算该动作的对数概率
            except (ValueError, RuntimeError) as e:
                # 兜底策略：从log_softmax选argmax（和原逻辑一致）
                log_probs_dist = F.log_softmax(batch_logits, dim=-1)
                action = torch.argmax(log_probs_dist)
                log_prob = log_probs_dist[action]
            
            # 处理维度和梯度
            log_prob = log_prob.squeeze() if log_prob.dim() > 0 else log_prob
            log_prob.requires_grad_(True)
            
            # 动作值转换
            action_item = int(action.item()) if isinstance(action, torch.Tensor) else action
            actions.append(action_item if action_item != "Stp" else "Stp")
            log_probs.append(log_prob)
        
        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        t_tra_vector, t_seed_vector, indptr, choices = [], [], [], []
        offset = 0
        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))
            choices.append(unique_neigh)
            neigh_embed = self.graph.nodesEmbed(unique_neigh)
            
            if not isinstance(neigh_embed, torch.Tensor):
                if isinstance(neigh_embed, list) and len(neigh_embed) == 0:
                    neigh_embed = torch.empty(0, tra_vector[i].size(-1), dtype=torch.float32, device=self.device)
                else:
                    neigh_embed = torch.tensor(np.stack(neigh_embed) if isinstance(neigh_embed, list) else neigh_embed, 
                                               dtype=torch.float32, device=self.device)
            else:
                neigh_embed = neigh_embed.to(self.device)
            if self.model.training:
                neigh_embed.requires_grad_(True)

            t_tra_vector.extend([neigh_embed, tra_vector[i].unsqueeze(0)])
            t_seed_vector.extend([neigh_embed, seed_vector[i].unsqueeze(0)])
            indptr.append((offset, offset + 1 + len(unique_neigh), offset + len(unique_neigh)))
            offset += len(unique_neigh) + 1

        return torch.cat(t_seed_vector, dim=0), torch.cat(t_tra_vector, dim=0), np.array(indptr), choices

    def add_node(self, new_node, tra_nodes, index):
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))
        if (new_node in (None, "Stp", -1) or len(tra_nodes[index]) >= self.maxLen or 
            (new_node != "Stp" and new_node in tra_nodes[index])):
            self.done[index] = True
            return None
        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        embed = torch.tensor(embed, dtype=torch.float32, device=self.device) if not isinstance(embed, torch.Tensor) else embed.to(self.device)
        if self.model.training:
            embed.requires_grad_(True)
        return embed

    def all_done(self):
        return all(self.done) if self.done else False

    def vecpool(self, v1, v2, k):
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        seed_vector = self.graph.nodesEmbed(seeds)
        seed_vector = torch.tensor(np.stack(seed_vector) if isinstance(seed_vector, list) else seed_vector, 
                                   dtype=torch.float32, device=self.device) if not isinstance(seed_vector, torch.Tensor) else seed_vector.to(self.device)
        if self.model.training:
            seed_vector.requires_grad_(True)

        tra_vector, tra_nodes, tra_logps = seed_vector.clone(), [[s] for s in seeds], [[] for _ in range(len(seeds))]
        self.done = [False] * len(seeds)
        step = 0

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break
            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]

            *model_inputs, batch_candidates = self.prepare_inputs(active_tra_vector, active_seed_vector, active_tra_nodes)
            batch_logits = self.model(*model_inputs)
            actions, logps = self.sample_actions(batch_logits)

            for j, orig_idx in enumerate(active_indices):
                ac, logp = actions[j], logps[j]
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
                        tra_vector[orig_idx] = self.vecpool(tra_vector[orig_idx], newvec, len(tra_nodes[orig_idx]))
                        tra_logps[orig_idx].append(logp)
            step += 1
        return tra_nodes, tra_logps

    def eval_auc_pr(self, selected_nodes: List[int], true_comm: Set[int], all_candidate_nodes: List[int] = None):
        """
        计算当前选择序列的AUC-PR（Average Precision）
        :param selected_nodes: 已选择的节点序列（按选择顺序）
        :param true_comm: 真实社区节点集合
        :param all_candidate_nodes: 所有候选节点（可选，默认用已选节点+真实社区节点）
        :return: AP值
        """
        # 1. 确定候选节点池（避免空值，兼容小样本）
        if all_candidate_nodes is None:
            all_candidate_nodes = list(set(selected_nodes) | true_comm)
        if not all_candidate_nodes:
            return 0.0
        
        # 2. 为候选节点生成「得分」和「真实标签」
        # 得分：选择顺序越靠前，得分越高（反向索引，如第1个选的得分为len(selected_nodes)，第2个为len-1...）
        node_score = {node: 0.0 for node in all_candidate_nodes}
        for idx, node in enumerate(selected_nodes):
            if node in node_score:
                # 选得越早，得分越高（保证得分唯一性，避免平局）
                node_score[node] = len(selected_nodes) - idx + 1e-6 * (len(selected_nodes) - idx)
        
        # 真实标签：1=属于真实社区，0=不属于
        y_true = np.array([1 if node in true_comm else 0 for node in all_candidate_nodes])
        y_score = np.array([node_score[node] for node in all_candidate_nodes])
        
        # 3. 计算AP（AUC-PR），处理全0/全1的边界情况
        try:
            # 过滤掉得分相同的重复值（避免sklearn计算报错）
            unique_indices = np.unique(y_score, return_index=True)[1]
            y_true_unique = y_true[unique_indices]
            y_score_unique = y_score[unique_indices]
            ap = average_precision_score(y_true_unique, y_score_unique)
        except ValueError:
            # 无正样本/负样本时，AP=0
            ap = 0.0
        # 限制AP范围，避免极端值
        ap = np.clip(ap, 0.0, 1.0)
        return ap

   
    def trainReward(self, seeds: List[int], true_coms):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        bs = len(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 计算指标（保留F1用于对比，新增AUC-PR）
        avg_metrics = {}
        p_list, r_list, f1_list, ap_list = [], [], [], []
        for pred_com, true_com in zip(selected_nodes, true_coms):
            pred_com_clean = [n for n in pred_com if n != "Stp"]
            p, r, f1 = self.eval_scores(pred_com_clean, true_com)
            ap = self.eval_auc_pr(pred_com_clean, set(true_com))  # 新增AUC-PR计算
            p_list.append(p), r_list.append(r), f1_list.append(f1), ap_list.append(ap)
        
        avg_metrics = {
            "avg_precision": round(np.mean(p_list),4), 
            "avg_recall": round(np.mean(r_list),4), 
            "avg_f1": round(np.mean(f1_list),4),
            "avg_auc_pr": round(np.mean(ap_list),4)  # 新增AUC-PR指标
        }
        # 打印新增AUC-PR指标
        print(f"Batch Metrics: P={avg_metrics['avg_precision']}, R={avg_metrics['avg_recall']}, F1={avg_metrics['avg_f1']}, AUC-PR={avg_metrics['avg_auc_pr']}")

        # 核心奖励计算（替换为AUC-PR，添加稳定性调优）
        rewards_list = []
        # 奖励放大系数（适配AUC-PR数值偏小的问题）
        reward_scale = 1.0
        for idx, (com, true_com) in enumerate(zip(selected_nodes, true_coms)):
            temp_com, step_rewards = [com[0]], []
            true_com_set = set(true_com)
            true_com_len = len(true_com_set)
            
            for node in com[1:]:
                if node == 'Stp':
                    continue
                # 计算新增节点前的AUC-PR
                pre_ap = self.eval_auc_pr(temp_com, true_com_set)
                temp_com.append(node)
                # 计算新增节点后的AUC-PR
                after_ap = self.eval_auc_pr(temp_com, true_com_set)
                # 增量AUC-PR（奖励选择能提升AUC-PR的节点）
                delta_ap = after_ap - pre_ap

                # 基础奖励：保留原权重结构，添加放大系数
                base_reward = (self.reward_weight_abs * after_ap + self.reward_weight_delta * delta_ap) * reward_scale
                
                # 1. 长度惩罚：对超过真实社区长度的选择做惩罚
                current_pred_len = len(temp_com)
                if current_pred_len > true_com_len:
                    base_reward *= (self.len_penalty_base ** (current_pred_len - true_com_len))
                
                # 2. 奖励裁剪：限制极端值，保证训练稳定
                base_reward = np.clip(base_reward, 0.0, reward_scale)  # 最小0，最大放大后的上限
                
                # 3. 奖励平滑：避免单步奖励波动过大
                if step_rewards:
                    base_reward = 0.2 * base_reward + 0.8 * np.mean(step_rewards)
                
                step_rewards.append(base_reward)

            # 折扣奖励（和原逻辑一致）
            discounted = []
            if step_rewards:
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + self.gamma * cum
                    discounted.insert(0, cum)
            # 空奖励兜底
            rewards_list.append(discounted if discounted else [0.0])

        # 填充对齐长度（兼容logp的维度）
        max_len = max(max(len(r) for r in rewards_list), max(len(lp) for lp in logps)) if (rewards_list and logps) else 0
        rewards_padded = np.zeros((bs, max_len))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, :len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        # logps填充（和原逻辑一致，保证梯度可导）
        logps_padded = []
        for lp_list in logps:
            padded = [
                lp_list[j] if j < len(lp_list) and lp_list[j] is not None 
                else torch.tensor(-1e9, device=self.device, requires_grad=True) 
                for j in range(max_len)
            ]
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # Mask与Loss计算（和原逻辑完全一致）
        mask = (torch.arange(max_len, device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)).float()
        loss = -(rewards * logps * mask).sum()

        # 梯度裁剪：防止梯度爆炸（新增，适配AUC-PR奖励放大）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        loss.backward()
        self.optimizer.step()

        return loss.item()