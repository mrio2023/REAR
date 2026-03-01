from typing import Union, Optional, List, Set
import pandas as pd
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
        gamma: float = 0.99,
        min_reward_threshold: float = 0.001,  # 【新增】最小有效奖励阈值
        invalid_penalty: float = 0.01,  # 【新增】无效节点惩罚值
        reward_weight_abs: float = 0.7,  # 【新增】F2绝对值权重
        reward_weight_delta: float = 0.3,  # 【新增】F2增量权重
        seedNum:int=50,
        epoch:int=30,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.done = []
        self.device = device or torch.device("cpu")
        # 新增超参数
        self.min_reward_threshold = min_reward_threshold
        self.invalid_penalty = invalid_penalty
        self.reward_weight_abs = reward_weight_abs
        self.reward_weight_delta = reward_weight_delta
        self.seedNum=seedNum
        self.epoch=epoch

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """计算社区检测评估指标（F1, P, R, J）"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect) + 1e-9)
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    # ========== 新增方法：F-beta 分数 ==========
    def eval_fbeta(self, pred_comm: Union[List, Set], true_comm: Union[List, Set], beta: float = 2.0):
        """计算 F-beta 分数，beta>1 更重视召回率"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        if p + r == 0:
            return 0.0
        fbeta = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)
        return round(fbeta, 4)
    # ==========================================

    def sample_actions(self, logits):
        """采样动作，返回动作和对数概率"""
        actions = []
        log_probs = []

        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0 or batch_logits.size(0) == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                continue

            if batch_logits.dim() > 1:
                batch_logits = batch_logits.squeeze()
                if batch_logits.dim() > 1:
                    batch_logits = batch_logits.mean(dim=-1)

            try:
                dist = torch.distributions.Categorical(logits=batch_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            except Exception:
                log_probs_dist = F.log_softmax(batch_logits, dim=-1)
                action = torch.argmax(log_probs_dist)
                log_prob = log_probs_dist[action]

            if log_prob.dim() > 0:
                log_prob = log_prob.squeeze()
            log_prob.requires_grad_(True)

            actions.append(int(action.item()))
            log_probs.append(log_prob)

        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        """准备模型输入"""
        t_tra_vector = []
        t_seed_vector = []
        indptr = []
        choices = []
        offset = 0

        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            unique_neigh = list(set(neigh) - set(tra))
            choices.append(unique_neigh)

            neigh_embed = self.graph.nodesEmbed(unique_neigh)
            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = torch.tensor(neigh_embed, dtype=torch.float32, device=self.device)
            else:
                neigh_embed = neigh_embed.to(self.device)

            if self.model.training:
                neigh_embed.requires_grad_(True)

            t_tra_vector.append(neigh_embed)
            t_tra_vector.append(tra_vector[i].unsqueeze(0))
            t_seed_vector.append(neigh_embed)
            t_seed_vector.append(seed_vector[i].unsqueeze(0))

            indptr.append((offset, offset + 1 + len(unique_neigh), offset + len(unique_neigh)))
            offset += len(unique_neigh) + 1

        t_tra_vector = torch.cat(t_tra_vector, dim=0)
        t_seed_vector = torch.cat(t_seed_vector, dim=0)
        indptr = np.array(indptr)

        return t_seed_vector, t_tra_vector, indptr, choices

    def add_node(self, new_node, tra_nodes, index):
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))

        if (new_node in (None, "Stp", -1) or
                len(tra_nodes[index]) >= self.maxLen or
                (new_node != "Stp" and new_node in tra_nodes[index])):
            self.done[index] = True
            return None

        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        if not isinstance(embed, torch.Tensor):
            embed = torch.tensor(embed, dtype=torch.float32, device=self.device)

        if self.model.training:
            embed.requires_grad_(True)
        return embed

    def all_done(self):
        return all(self.done) if self.done else False

    def vecpool(self, v1, v2, k):
        return (v1 * (k - 1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        seed_vector = self.graph.nodesEmbed(seeds)
        if not isinstance(seed_vector, torch.Tensor):
            seed_vector = torch.tensor(seed_vector, dtype=torch.float32, device=self.device)
        else:
            seed_vector = seed_vector.to(self.device)

        if self.model.training:
            seed_vector.requires_grad_(True)

        tra_vector = seed_vector.clone()
        tra_nodes = [[s] for s in seeds]
        tra_logps = [[] for _ in range(len(seeds))]
        step = 0
        self.done = [False] * len(seeds)

        while step < self.maxLen and not self.all_done():
            active_indices = [i for i, d in enumerate(self.done) if not d]
            if not active_indices:
                break

            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]

            *model_inputs, batch_candidates = self.prepare_inputs(
                tra_nodes=active_tra_nodes,
                tra_vector=active_tra_vector,
                seed_vector=active_seed_vector
            )

            batch_logits = self.model(*model_inputs)
            actions, logps = self.sample_actions(batch_logits)

            for j, orig_idx in enumerate(active_indices):
                ac = actions[j]
                logp = logps[j]

                if isinstance(ac, str) and ac == "Stp":
                    self.add_node("Stp", tra_nodes, orig_idx)
                    if logp is not None:
                        tra_logps[orig_idx].append(logp)
                else:
                    cand_list = batch_candidates[j]
                    if ac >= len(cand_list):
                        self.add_node("Stp", tra_nodes, orig_idx)
                        tra_logps[orig_idx].append(logp)
                    else:
                        selected_node = cand_list[ac]
                        if selected_node in tra_nodes[orig_idx]:
                            self.add_node("Stp", tra_nodes, orig_idx)
                            tra_logps[orig_idx].append(logp)
                            continue
                        newvec = self.add_node(selected_node, tra_nodes, orig_idx)
                        if newvec is not None:
                            current_len = len(tra_nodes[orig_idx])
                            with torch.no_grad():
                                tra_vector[orig_idx] = self.vecpool(
                                    tra_vector[orig_idx],
                                    newvec,
                                    current_len
                                )
                            tra_logps[orig_idx].append(logp)

            step += 1

        return tra_nodes, tra_logps

    def trainReward(self, seeds: List[int], true_coms):
        """
        通过奖励更新参数（移除isweak逻辑）
        @param seeds: 一个batch的节点
        @param true_coms: 节点对应的真实社区（即sample_coms）
        @return: loss值 + 本次batch的平均相似度指标
        """
        bs = len(seeds)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # ===================== 计算与sample_com的平均相似度 =====================
        batch_metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "jaccard": [],
            "f2": []
        }
        
        for idx, (pred_com, sample_com) in enumerate(zip(selected_nodes, true_coms)):
            pred_com_clean = [node for node in pred_com if node != "Stp"]
            p, r, f1, j = self.eval_scores(pred_com_clean, sample_com)
            f2 = self.eval_fbeta(pred_com_clean, sample_com, beta=2.0)
            
            batch_metrics["precision"].append(p)
            batch_metrics["recall"].append(r)
            batch_metrics["f1"].append(f1)
            batch_metrics["jaccard"].append(j)
            batch_metrics["f2"].append(f2)
        
        avg_metrics = {
            "avg_precision": round(np.mean(batch_metrics["precision"]), 4),
            "avg_recall": round(np.mean(batch_metrics["recall"]), 4),
            "avg_f1": round(np.mean(batch_metrics["f1"]), 4),
            "avg_jaccard": round(np.mean(batch_metrics["jaccard"]), 4),
            "avg_f2": round(np.mean(batch_metrics["f2"]), 4)
        }
        
        print(f"Batch Metrics (Pred vs SampleCom): "
              f"P={avg_metrics['avg_precision']}, R={avg_metrics['avg_recall']}, "
              f"F1={avg_metrics['avg_f1']}, Jaccard={avg_metrics['avg_jaccard']}, F2={avg_metrics['avg_f2']}")

        # 计算每个样本每一步的折扣奖励（移除isweak相关逻辑）
        rewards_list = []
        gamma = self.gamma  # 【修改】使用类属性的gamma，保持一致性
        for idx, com in enumerate(selected_nodes):
            true_com = true_coms[idx]
            temp_com = [com[0]]
            step_rewards = []
            
            for step_idx, node in enumerate(com[1:]):
                if node == 'Stp':
                    continue
                # 计算加节点前后的F2分数
                pre_f2 = self.eval_fbeta(temp_com, true_com, beta=2.0)
                temp_com.append(node)
                after_f2 = self.eval_fbeta(temp_com, true_com, beta=2.0)
                
                # 【修改】重构奖励计算：绝对值+增量，兼顾"当前F2高度"和"增量"
                delta_f2 = after_f2 - pre_f2
                reward = self.reward_weight_abs * after_f2 + self.reward_weight_delta * delta_f2
                
                # 【修改】无效节点惩罚：增量小于阈值则扣奖励
                if delta_f2 < self.min_reward_threshold:
                    reward -= self.invalid_penalty
                
                step_rewards.append(reward)

            # 计算折扣奖励（保留原有逻辑）
            if step_rewards:
                discounted = []
                cum = 0.0
                for r in reversed(step_rewards):
                    cum = r + gamma * cum
                    discounted.insert(0, cum)
                rewards_list.append(discounted)
            else:
                rewards_list.append([0.0])

        # 填充到相同长度（保留原有逻辑）
        max_len = max(len(r) for r in rewards_list) if rewards_list else 0
        max_logp_len = max(len(lp) for lp in logps) if logps else 0
        max_len = max(max_len, max_logp_len)

        rewards_padded = np.zeros((bs, max_len))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, :len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)

        logps_padded = []
        for i, lp_list in enumerate(logps):
            padded = []
            for j in range(max_len):
                if j < len(lp_list) and lp_list[j] is not None:
                    padded.append(lp_list[j])
                else:
                    padded.append(torch.tensor(0.0, device=self.device, requires_grad=True))
            logps_padded.append(torch.stack(padded))
        logps = torch.stack(logps_padded)

        # mask（保留原有逻辑）
        mask = torch.arange(rewards.size(1), device=self.device).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()

        # 计算损失（保留原有逻辑）
        loss = -(rewards * logps * mask).sum()
        loss.backward()

        # 极简打印核心参数和梯度（仅6行）
        print("reward:",reward)
        print("logps:",logps)
        print("MASK:",mask)
        # 打印关键参数梯度
        for n,p in self.model.named_parameters():
            if p.grad is not None:
                print(f"{n[:10]} 梯度范数: {torch.norm(p.grad).item():.6f}", end=" | ")
        print("\n" + "-"*50)

        self.optimizer.step()

        return loss.item(), avg_metrics




# from dataProcess import dataProcess

# def test_expander():
#     """
#     测试Expander类核心功能（复用真实Graph+elliptic数据集）：
#     1. 加载elliptic数据集并初始化Graph
#     2. 初始化Expander（真实Graph+Agent+优化器）
#     3. 测试trainReward方法（移除isweak后）
#     4. 验证损失计算和指标输出
#     """
#     try:
#         # ===================== 1. 加载真实数据并初始化Graph =====================
#         # 初始化数据处理类（加载elliptic数据集）
#         dp = dataProcess("elliptic")
#         print("✅ dataProcess初始化成功")
        
#         # 初始化真实Graph类（使用训练集数据）
#         g = Graph(
#             dfnode=dp.train_nodes,
#             dffeature=dp.train_feature,
#             dfhacker=dp.train_hacker,
#             dfedge=dp.train_edge
#         )
#         print("✅ Graph类初始化成功（基于elliptic训练集）")

#         # ===================== 2. 初始化Expander组件 =====================
#         device = torch.device("cpu")  # 可改为"cuda"（如有GPU）
#         # 初始化Agent模型（输入维度匹配Graph的embedsize）
#         model = Agent(input_size=g.embedsize,hidden_size=128).to(device)
#         # 初始化优化器
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
#         # 初始化Expander
#         expander = Expander(
#             graph=g,
#             model=model,
#             optimizer=optimizer,
#             device=device,
#             maxLen=10,  # 缩小测试长度，加快运行
#             gamma=0.99,
#             min_reward_threshold=0.001,
#             invalid_penalty=0.01
#         )
#         print("✅ Expander初始化成功")

#         # ===================== 3. 构造测试数据 =====================
#         # 从训练集黑客节点中选2个作为测试种子（避免空数据）
#         if len(dp.train_hacker) < 2:
#             raise ValueError("训练集黑客节点数量不足，无法测试")
        
#         seeds = dp.train_hacker["address"].iloc[:2].tolist()  # 取前2个种子节点
#         print(f"✅ 构造测试种子节点：{seeds}")
        
#         # 构造真实社区（从Graph的community_seeds中获取对应社区）
#         true_coms = []
#         for seed in seeds:
#             # 获取种子节点的真实社区标签
#             node_hacker_row = dp.train_hacker[dp.train_hacker["address"] == seed]
#             if not node_hacker_row.empty and not pd.isna(node_hacker_row["name_tag"].iloc[0]):
#                 tag = node_hacker_row["name_tag"].iloc[0]
#                 # 从Graph中获取该标签对应的真实社区节点
#                 true_com = list(g.community_seeds.get(tag, set()))[:10]  # 取前10个节点
#             else:
#                 true_com = [seed]  # 兜底：仅包含自身
#             true_coms.append(true_com)
#         print(f"✅ 构造真实社区完成，社区大小：{[len(c) for c in true_coms]}")

#         # ===================== 4. 测试trainReward方法 =====================
#         loss, metrics = expander.trainReward(seeds=seeds, true_coms=true_coms)
#         print("\n===================== 测试结果 =====================")
#         print(f"✅ trainReward执行成功")
#         print(f"   本次训练损失值：{loss:.4f}")
#         print(f"   平均精度(P)：{metrics['avg_precision']}")
#         print(f"   平均召回(R)：{metrics['avg_recall']}")
#         print(f"   平均F1分数：{metrics['avg_f1']}")
#         print(f"   平均F2分数：{metrics['avg_f2']}")
#         print(f"   平均Jaccard系数：{metrics['avg_jaccard']}")

#     except Exception as e:
#         print(f"\n❌ 测试失败：{e}")
#         import traceback
#         traceback.print_exc()

# # 执行测试
# if __name__ == "__main__":
#     test_expander()