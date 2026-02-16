import copy
from typing import Union, Optional, List, Set
import numpy as np
import torch
import torch.nn.functional as F
from dataProcess import DataProcess
import random
from tool import Tool
from graph import Graph
from gnn import GNN
from Agent import Agent


class Expander:
    def __init__(
        self,
        graph: Graph,
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
        self.conv = GNN(graph, k, alpha)
        self.gamma = gamma
        self.maxLen = maxLen
        self.done = []
        self.device = device or torch.device("cpu")

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """计算社区检测评估指标"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if pred_comm else 0.0
        r = len(intersect) / len(true_comm) if true_comm else 0.0
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect) + 1e-9)
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    def sample_actions(self, logits):
        """采样动作，返回动作和对数概率"""
        actions = []
        log_probs = []
        
        for batch_logits in logits:
            # 处理空张量
            if batch_logits is None or batch_logits.numel() == 0 or batch_logits.size(0) == 0:
                actions.append("Stp")
                log_probs.append(None)
                continue

            # 确保logits是1D
            if batch_logits.dim() == 2:
                if batch_logits.size(0) == 1:
                    batch_logits = batch_logits.squeeze(0)
                elif batch_logits.size(0) > 1:
                    batch_logits = batch_logits[0]
            
            # 再次检查是否有效
            if batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(None)
                continue
            
            # 计算概率分布
            log_probs_dist = F.log_softmax(batch_logits, dim=-1)
            probs = torch.exp(log_probs_dist).detach().cpu().numpy()
            probs = probs / (probs.sum() + 1e-8)  # 确保和为1，避免除零
            
            # 采样
            if len(probs) == 1:
                action = 0
            else:
                try:
                    action = np.random.choice(len(probs), p=probs)
                except ValueError:
                    action = np.argmax(probs)  # 降级处理
            
            # 确保log_prob是标量
            log_prob = log_probs_dist[action]
            if log_prob.dim() > 0:
                log_prob = log_prob.squeeze()
            
            actions.append(int(action))
            log_probs.append(log_prob.cpu())
        
        return actions, log_probs

    def prepare_inputs(self, tra_nodes):
        """准备模型输入"""
        choices = []
        indptr = []
        offset = 0
        
        for tra in tra_nodes:
            if not tra:
                indptr.append([offset, offset, offset])
                continue
            
            cur = tra[-1]
            data = self.graph.parentGraph.adjmap.get(cur, [])
            candidates = [d["to"] for d in data] or [-1]  # 无邻居时用-1
            
            # 记录指针：先候选节点，后轨迹节点
            start = offset
            cand_start = offset + len(candidates)
            end = offset + len(candidates) + len(tra)
            indptr.append([start, end, cand_start])
            
            choices.extend(candidates + tra)
            offset = end
        
        return choices, torch.tensor(indptr, device=self.device)

    def add_node(self, new_node, tra_nodes, index):
        """添加节点到轨迹"""
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))
        
        # 终止条件
        if new_node in (None, "Stp") or len(tra_nodes[index]) >= self.maxLen:
            self.done[index] = True
            return None
        
        tra_nodes[index].append(new_node)
        return self.graph.parentGraph.singleNodeEmbed(new_node)

    def all_done(self):
        """检查是否全部完成"""
        return bool(self.done) and all(self.done)

    def calc_rewards(self, tra_nodes, true_comms=None):
        """计算奖励"""
        rewards = []
        
        for i, tra in enumerate(tra_nodes):
            if true_comms and i < len(true_comms):
                _, _, f1, _ = self.eval_scores(tra, true_comms[i])
                rewards.append([f1])
            else:
                rewards.append([len(tra) / self.maxLen])
        
        return rewards

    def train_rl(self, seeds: list, true_comms: List[Set[int]] = None, episodes: int = 30):
        """
        REINFORCE算法训练
        """
        # 初始化
        tra_nodes = [[n] for n in seeds]
        seed_embeds = self.graph.nodesEmbed(seeds)
        tra_embeds = seed_embeds.copy()
        self.done = [False] * len(seeds)
        
        all_log_probs = []
        all_rewards = []
        
        # 轨迹扩展
        step = 0
        while not self.all_done() and step < self.maxLen:
            # 准备输入
            choices, indptr = self.prepare_inputs(tra_nodes)
            
            # 转换为tensor
            seed_tensor = torch.tensor(np.array(seed_embeds), device=self.device, dtype=torch.float32)
            tra_tensor = torch.tensor(np.array(tra_embeds), device=self.device, dtype=torch.float32)
            
            # 前向传播
            logits = self.model(seed_tensor, tra_tensor, indptr)
            
            # 采样动作
            actions, step_log_probs = self.sample_actions(logits[:len(tra_nodes)])
            
            # 收集对数概率
            for i, (act, log_prob) in enumerate(zip(actions, step_log_probs)):
                if not self.done[i] and log_prob is not None:
                    all_log_probs.append(log_prob)
            
            # 执行动作
            new_embeds = []
            for i, action in enumerate(actions):
                if self.done[i]:
                    new_embeds.append(None)
                    continue
                
                # 获取新节点
                if action == "Stp":
                    new_node = "Stp"
                else:
                    new_node = choices[action] if action < len(choices) else "Stp"
                
                # 添加节点
                node_embed = self.add_node(new_node, tra_nodes, i)
                new_embeds.append(node_embed)
            
            # 更新轨迹嵌入
            for i, embed in enumerate(new_embeds):
                if embed is not None and i < len(tra_embeds):
                    tra_len = len(tra_nodes[i])
                    tra_embeds[i] = (tra_embeds[i] * (tra_len - 1) + embed) / tra_len
            
            # 记录奖励
            all_rewards.append(self.calc_rewards(tra_nodes, true_comms))
            step += 1
        
        # 如果没有有效动作，提前返回
        if not all_log_probs:
            return {
                "loss": 0, 
                "expanded_communities": tra_nodes, 
                "steps": step, 
                "final_rewards": []
            }
        
        # 计算折扣奖励 - 保持每个轨迹的奖励分开
        discounted_per_trajectory = []  # 二维列表
        
        for tra_idx in range(len(tra_nodes)):
            tra_rew = [r[tra_idx][0] for r in all_rewards if tra_idx < len(r)]
            
            disc = []
            cum = 0
            for r in reversed(tra_rew):
                cum = r + self.gamma * cum
                disc.append(cum)
            disc.reverse()
            discounted_per_trajectory.append(disc)
        
        # 将所有奖励展平用于损失计算
        discounted_flat = []
        for tra_disc in discounted_per_trajectory:
            discounted_flat.extend(tra_disc)
        
        # 对齐长度
        min_len = min(len(all_log_probs), len(discounted_flat))
        
        if min_len == 0:
            return {
                "loss": 0, 
                "expanded_communities": tra_nodes, 
                "steps": step, 
                "final_rewards": []
            }
        
        # 计算损失
        log_probs_tensor = torch.stack(all_log_probs[:min_len])
        rewards_tensor = torch.tensor(discounted_flat[:min_len], device=self.device, dtype=torch.float32)
        
        # 标准化奖励
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        loss = -(log_probs_tensor * rewards_tensor).sum()
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 计算每个轨迹的最终奖励总和
        final_rewards = [sum(disc) for disc in discounted_per_trajectory]
        
        return {
            "loss": loss.item(),
            "expanded_communities": tra_nodes,
            "steps": step,
            "final_rewards": final_rewards
        }


# ===================== 测试代码 =====================
if __name__ == "__main__":
    # 加载数据
    d = DataProcess(dfname="PlusTokenPonzi")
    g = Graph(dffeature=d.train_feature, dfhacker=d.train_hacker, dfnode=d.train_nodes)
    g = Tool(graph=g).initKego()
    
    # 初始化模型
    agent = Agent(hidden_size=128, input_size=84)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    expander = Expander(graph=g, model=agent, optimizer=optimizer, device=torch.device("cpu"))
    
    # 准备种子和真实社区
    hackers = g.df_hacker["address"].values.tolist()
    seeds = random.sample(hackers, min(3, len(hackers)))
    
    # 构建社区映射
    com_map = {}
    for name, df in g.df_hacker.groupby("name_tag"):
        addresses = set(df["address"].values)
        for addr in addresses:
            com_map[addr] = addresses
    
    true_comms = [com_map.get(s, set()) for s in seeds]
    
    # 训练
    result = expander.train_rl(seeds=seeds, true_comms=true_comms, episodes=30)
    
    # 输出结果
    print(f"损失: {result['loss']:.4f}")
    print(f"步数: {result['steps']}")
    for i, (seed, comm) in enumerate(zip(seeds, result["expanded_communities"])):
        print(f"\n种子{i}: {seed}")
        print(f"  扩展社区 ({len(comm)}个节点)")
        if true_comms[i]:
            p, r, f1, j = expander.eval_scores(comm, true_comms[i])
            print(f"  评估: P={p}, R={r}, F1={f1}, J={j}")