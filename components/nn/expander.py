
from typing import Union, Optional, List, Set
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F



from codes.components.nn.Agent import Agent


class Expander:
    def __init__(
        self,
        graph,
        model: Agent,
        optimizer,
        device: Optional[torch.device] = None,
        k: int = 3,
        alpha: float = 0.85,
        gamma: float = 0.99,
        maxLen: int = 16,
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.maxLen = maxLen
        self.done = []
        self.device = device or torch.device("cpu")

        # 缓存每个社区的全邻居（所有社区节点的出边+入边邻居）
        self.community_full_neighbors = self._cache_community_full_neighbors()

    def _cache_community_full_neighbors(self):
        """缓存每个社区的全邻居（所有社区节点的出边+入边邻居）"""
        full_neighbors = {}
        for tag in self.graph.allNameTags:
            com_nodes = self.graph.community_seeds.get(tag, set())
            if not com_nodes:
                full_neighbors[tag] = []
                continue
            com_neighbors = self.graph.getNodesNeigh(list(com_nodes))
            full_neighbors[tag] = com_neighbors
        return full_neighbors

    def get_node_community(self, node: str):
        """获取节点所属的社区标签"""
        node_com_row = self.graph.df_hacker[self.graph.df_hacker["address"] == node]
        if node_com_row.empty or pd.isna(node_com_row["name_tag"].iloc[0]):
            return None
        return node_com_row["name_tag"].iloc[0]

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """计算社区检测评估指标"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect) + 1e-9)
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    def sample_actions(self, logits):
        """采样动作，返回动作和对数概率（适配Agent修复后的logits）"""
        actions = []
        log_probs = []

        for batch_idx, batch_logits in enumerate(logits):
            if batch_logits is None or batch_logits.numel() == 0 or batch_logits.size(0) == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                continue

            if batch_logits.dim() > 1:
                batch_logits = batch_logits.squeeze()

            if batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                continue

            # 在采样环节做log_softmax，保留完整梯度链路
            log_probs_dist = F.log_softmax(batch_logits, dim=-1)

            try:
                # 使用原始logits构造分布（推荐）
                dist = torch.distributions.Categorical(logits=batch_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            except Exception as e:
                # 降级方案：使用log_softmax结果
                action = torch.argmax(log_probs_dist)
                log_prob = log_probs_dist[action]

            if log_prob.dim() > 0:
                log_prob = log_prob.squeeze()

            # 强制保留梯度
            log_prob.requires_grad_(True)

            actions.append(int(action.item()))
            log_probs.append(log_prob)
            
        return actions, log_probs

    def prepare_inputs(self, tra_vector, seed_vector, tra_nodes):
        # tra_vector 和 seed_vector 现在都是 tensor
        neigh = []
        t_tra_vector = []
        t_seed_vector = []
        indptr = []
        choice = []
        offset = 0
        
        for i, tra in enumerate(tra_nodes):
            neigh = self.graph.getNodesNeigh(tra)
            choice.append(neigh)
            neigh_embed = self.graph.nodesEmbed(neigh)
            
            # 确保 neigh_embed 是 tensor（统一处理numpy/tensor）
            if not isinstance(neigh_embed, torch.Tensor):
                neigh_embed = np.array(neigh_embed)
                neigh_embed = torch.tensor(neigh_embed, dtype=torch.float32, device=self.device)
            else:
                neigh_embed = neigh_embed.to(self.device)
            
            # 训练阶段强制开启梯度
            if self.model.training:
                neigh_embed.requires_grad_(True)

            # 直接拼接 tensor，不用 numpy
            t_tra_vector.append(neigh_embed)
            t_tra_vector.append(tra_vector[i].unsqueeze(0))
            
            t_seed_vector.append(neigh_embed)
            t_seed_vector.append(seed_vector[i].unsqueeze(0))
            
            indptr.append((offset, offset + 1 + len(neigh), offset + len(neigh)))
            offset += len(neigh) + 1
        
        # 直接 cat tensor
        t_tra_vector = torch.cat(t_tra_vector, dim=0)
        t_seed_vector = torch.cat(t_seed_vector, dim=0)
        indptr = np.array(indptr)
        
        return t_seed_vector, t_tra_vector, indptr, choice

    def add_node(self, new_node, tra_nodes, index):
        """添加节点到轨迹"""
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))

        if new_node in (None, "Stp", -1) or len(tra_nodes[index]) >= self.maxLen:
            self.done[index] = True
            return None

        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        
        # 将numpy数组转为tensor
        if not isinstance(embed, torch.Tensor):
            embed = torch.tensor(embed, dtype=torch.float32, device=self.device)
        
        # 训练阶段强制开启梯度
        if self.model.training:
            embed.requires_grad_(True)
        return embed

    def all_done(self):
        """检查是否全部完成"""
        return all(self.done) if self.done else False

    def calc_rewards(self, tra_nodes, true_comms=None):
        """计算奖励（新增：有真实标签时，超长部分单独扣-0.1惩罚，保留多维列表格式）"""
        rewards = []
        # 可自定义：超长惩罚力度（每超1个节点扣0.1）、超长阈值（超过真实长度即罚）
        OVER_LENGTH_PENALTY = -0.1  # 每个超长节点的惩罚值

        for i, tra in enumerate(tra_nodes):
            if true_comms and i < len(true_comms):
                # 场景1：有有效真实标签 - F1不变，超长部分单独扣罚
                true_comm = true_comms[i]
                true_len = len(true_comm)
                pred_len = len(tra)
                
                # 1. 计算原始F1（和原逻辑完全一致，不修改）
                p, r, f1, j = self.eval_scores(tra, true_comm)
                
                # 2. 计算超长惩罚：仅当预测长度 > 真实长度时，按超出节点数扣罚
                over_penalty = 0.0
                if pred_len > true_len:
                    over_len = pred_len - true_len  # 超出的节点数量
                    over_penalty = over_len * OVER_LENGTH_PENALTY  # 总惩罚
                    # 兜底：确保奖励不会扣成负数（可选，按需删除）
                    over_penalty = max(over_penalty, -f1)
                
                # 3. 最终奖励 = 原始F1 + 超长惩罚，保留多维列表格式 [最终奖励]
                final_reward = f1 + over_penalty
                rewards.append([final_reward])
            
            else:
                # 场景2：无有效真实标签/索引越界 - 保留原逻辑，仅优化格式
                reward_val = -0.1
                if not true_comms:
                    reward_val = len(tra) / self.maxLen
                    # 可选：无标签时也给超长惩罚（比如超过maxLen时扣罚）
                    if len(tra) > self.maxLen:
                        over_len = len(tra) - self.maxLen
                        reward_val += over_len * OVER_LENGTH_PENALTY
                        reward_val = max(reward_val, 0.0)  # 最低奖励为0
                # 保留原多维列表格式
                rewards.append([reward_val])

        return rewards
    def vecpool(self, v1, v2, k):
        # 直接返回新 tensor，不使用 in-place 操作（避免梯度污染）
        return (v1 * (k-1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        seed_vector = self.graph.nodesEmbed(seeds)
        
        # 统一seed_vector为tensor+强制梯度
        if not isinstance(seed_vector, torch.Tensor):
            seed_vector = np.array(seed_vector)
            seed_vector = torch.tensor(seed_vector, dtype=torch.float32, device=self.device)
        else:
            seed_vector = seed_vector.to(self.device)
        
        # 训练阶段强制开启梯度
        if self.model.training:
            seed_vector.requires_grad_(True)
        
        tra_vector = seed_vector.clone()  # 克隆保留梯度链路

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

            # 模型前向传播
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
                        # print(f"Stp触发：样本{orig_idx} 索引{ac}≥候选数{len(cand_list)}")
                        self.add_node("Stp", tra_nodes, orig_idx)
                        tra_logps[orig_idx].append(logp)
                    else:
                        selected_node = cand_list[ac]
                        newvec = self.add_node(selected_node, tra_nodes, orig_idx)
                        
                        if newvec is not None:
                            current_len = len(tra_nodes[orig_idx])
                            # 仅pooling时禁用梯度，不影响主链路
                            with torch.no_grad():
                                tra_vector[orig_idx] = self.vecpool(
                                    tra_vector[orig_idx],
                                    newvec,
                                    current_len
                                )
                            tra_logps[orig_idx].append(logp)
            
            step += 1
        
        return tra_nodes, tra_logps

    def trainReward(self, seeds: List[int], true_coms,isweak):
        '''
        通过奖励更新参数
        @param seeds: 一个batch的节点
        @param true_coms: 节点对应的真实社区
        '''
        

        bs = len(seeds)
        self.model.train()
        # 高效梯度清零，避免梯度残留
        self.optimizer.zero_grad(set_to_none=True)

        # 采样轨迹（训练模式下，输入已开启梯度追踪）
        selected_nodes, logps = self.sample_bs_trajectories(seeds)
        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 计算奖励
        rewards_list = []

       
        for index in range(len(selected_nodes)):

            com = selected_nodes[index]
            true_com = true_coms[index]
            isweakflags=isweak[index]
            # print(isweakflags)

            r, gamma = [], 0.99
            temp_com = [com[0]]
            for step_idx, node in enumerate(com[1:]):
                if node != 'Stp':
                    _, _, pre_cost, _ = self.eval_scores(temp_com, true_com)
                    temp_com.append(node)
                    _, _, after_cost, _ = self.eval_scores(temp_com, true_com)
                    step_reward = after_cost - pre_cost
                    if step_idx < len(isweakflags) and isweakflags[step_idx]:
                        step_reward *= 0.5

                        # print("weak_index:",step_idx)


                    r.append(step_reward)
            if len(r) == 0:
                reward = [0.0]
            else:
                reward = [np.sum(r[i] * (gamma ** np.array(range(i, len(r))))) for i in range(len(r))]
            rewards_list.append(reward)

        # 填充rewards和logps到相同长度
        max_len = max(len(r) for r in rewards_list) if rewards_list else 0
        max_logp_len = max(len(lp) for lp in logps) if logps else 0
        max_len = max(max_len, max_logp_len)
        
        # 奖励填充：保持原有逻辑
        rewards_padded = np.zeros((bs, max_len))
        for i, r in enumerate(rewards_list):
            rewards_padded[i, :len(r)] = r
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)
        
        # print("rewards",rewards)
        # print("logps",logps)

        # Logps填充：避免原地操作，先创建列表再拼接
        logps_list = []
        for i, lp_list in enumerate(logps):
            # 对每个样本的logps进行填充
            padded_lp = []
            for j in range(max_len):
                if j < len(lp_list) and lp_list[j] is not None:
                    padded_lp.append(lp_list[j])
                else:
                    # 填充0，且保持梯度特性
                    padded_lp.append(torch.tensor(0.0, device=self.device, requires_grad=True))
            # 拼接成单个tensor
            logps_tensor = torch.stack(padded_lp)
            logps_list.append(logps_tensor)
        # 拼接所有样本的logps
        logps = torch.stack(logps_list)
        
        # 生成mask
        mask = torch.arange(rewards.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()

        # print("rewards:",reward)
        # print("logs:",logps)
        # print("masks:",mask)

        # 计算损失
        loss_core = -(rewards * logps * mask).sum()
        policy_loss = loss_core * 1.0  # 损失缩放系数
        loss = policy_loss

        # 反向传播
        loss.backward()
        # # #看梯度
        print({name: param.grad.mean().item() if param.grad is not None else None for name, param in self.model.named_parameters()})
        # 优化器更新
        self.optimizer.step()

        return policy_loss.item()

# from random import random
# from torch import optim
# from ellipticGraph import ellipticGraph
# from ellipticDataProcess import ellipticDataProcess
# def test(epochs=3):
#     d = ellipticDataProcess(dfname="elliptic2")
#     g = ellipticGraph(dffeature=d.train_feature, dfhacker=d.train_hacker, dfnode=d.train_nodes)
    
#     model = Agent(input_size=g.embedsize, hidden_size=128)
#     # 优化器配置
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     e = Expander(graph=g, optimizer=optimizer, model=model)
    
#     history = {'loss': [], 'f1': []}
    
#     for epoch in range(epochs):
#         # 训练
#         seeds = random.sample(g.df_hacker["address"].values.tolist(), k=3)
#         isweak=[]
#         truecom =[]
#         for s in seeds:
#             c,w=g.sampleTrajectory(s)
#             truecom.append(c)
#             isweak.append(w)

#         print(truecom,isweak)
#         print(truecom)
#         loss = e.trainReward(seeds=seeds, true_coms=truecom,isweak=isweak)
#         history['loss'].append(loss)
        
      
    

    
#     return history

# if __name__ == "__main__":
#     test()