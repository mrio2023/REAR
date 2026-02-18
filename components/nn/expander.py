import copy
import time
from typing import Union, Optional, List, Set
import pandas as pd
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
        """采样动作，返回动作和对数概率"""
        actions = []
        log_probs = []

        for batch_idx, batch_logits in enumerate(logits):
            if batch_logits is None or batch_logits.numel() == 0 or batch_logits.size(0) == 0:
                actions.append("Stp")
                log_probs.append(None)
                continue

            if batch_logits.dim() == 2:
                if batch_logits.size(0) == 1:
                    batch_logits = batch_logits.squeeze(0)
                elif batch_logits.size(0) > 1:
                    batch_logits = batch_logits[0]

            if batch_logits.numel() == 0:
                actions.append("Stp")
                log_probs.append(None)
                continue

            log_probs_dist = F.log_softmax(batch_logits, dim=-1)
            probs = torch.exp(log_probs_dist).detach().cpu().numpy()
            probs = probs / (probs.sum() + 1e-8)

            if len(probs) == 1:
                action = 0
            else:
                try:
                    action = np.random.choice(len(probs), p=probs)
                except ValueError:
                    action = np.argmax(probs)

            log_prob = log_probs_dist[action]
            if log_prob.dim() > 0:
                log_prob = log_prob.squeeze()

            actions.append(int(action))
            log_probs.append(log_prob.item())

        return actions, log_probs

    def prepare_inputs(self, tra_vector,seed_vector,tra_nodes):
        neigh=[]
        t_tra_vector=[]
        t_seed_vector=[]
        indptr=[]
        choice=[]#候选人
        offset = 0
        for i,tra in enumerate(tra_nodes):
            neigh=self.graph.getNodesNeigh(tra)
            choice.append(neigh)
            neigh_embed=self.graph.nodesEmbed(neigh)
         
            temp=neigh_embed+[tra_vector[i]]
            t_tra_vector+=temp
            temp=neigh_embed+[seed_vector[i]]
            t_seed_vector+=temp
            indptr.append((offset, offset +1+len(neigh), offset + len(neigh)))  
            offset += len(neigh)+1
        
      
        t_seed_vector=np.array(t_seed_vector)
        t_tra_vector=np.array(t_tra_vector)
        vals_seed = torch.from_numpy(t_seed_vector).to(self.device)
        vals_node = torch.from_numpy(t_tra_vector).to(self.device)
        indptr = np.array(indptr)
        return vals_seed, vals_node, indptr, choice

     

    def add_node(self, new_node, tra_nodes, index):
        """添加节点到轨迹"""
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))

        if new_node in (None, "Stp", -1) or len(tra_nodes[index]) >= self.maxLen:
            self.done[index] = True
            return None

        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)
        return embed

    def all_done(self):
        """检查是否全部完成"""
        return all(self.done) if self.done else False

    def calc_rewards(self, tra_nodes, true_comms=None):
        """计算奖励"""
        rewards = []

        for i, tra in enumerate(tra_nodes):
            if true_comms and i < len(true_comms):
                p, r, f1, j = self.eval_scores(tra, true_comms[i])
                rewards.append([f1])
            else:
                reward_val = len(tra) / self.maxLen
                rewards.append([reward_val])

        return rewards
    def vecpool(self,v1, v2, k):
        # print(v1)
        # print(v2)
        if not isinstance(v1, np.ndarray):
            v1 = np.array(v1)
        if not isinstance(v2, np.ndarray):
            v2 = np.array(v2)
        

        if len(v1) != len(v2):
            print(f"警告：向量维度不一致 v1:{len(v1)} v2:{len(v2)}")
            # 可以截断或补零
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]
        
        return (v1*(k-1) + v2) / k

    def sample_bs_trajectories(self, seeds):
        seed_vector = self.graph.nodesEmbed(seeds)
        tra_nodes = [[s] for s in seeds]
        tra_logps = [[] for _ in range(len(seeds))]  # 改为空列表，不要用[1]
        tra_vector = copy.deepcopy(seed_vector)
        step = 0
        self.done = [False] * len(seeds)  # 初始化 done 状态
        
        while step < self.maxLen and not self.all_done():
            # 只处理未完成的轨迹
            active_indices = [i for i, d in enumerate(self.done) if not d]
            
            if not active_indices:
                break
                
            active_tra_nodes = [tra_nodes[i] for i in active_indices]
            active_tra_vector = [tra_vector[i] for i in active_indices]
            active_seed_vector = [seed_vector[i] for i in active_indices]
            
            # 准备输入
            *model_inputs, batch_candidates = self.prepare_inputs(
                tra_nodes=active_tra_nodes,
                tra_vector=active_tra_vector,
                seed_vector=active_seed_vector
            )
            
            # 模型前向传播
            batch_logits = self.model(*model_inputs)
            actions, logps = self.sample_actions(batch_logits)
            
            # 更新每个活跃轨迹
            for j, orig_idx in enumerate(active_indices):
                ac = actions[j]
                logp = logps[j]
                
                # 处理不同类型的动作
                if isinstance(ac, str) and ac == "Stp":  # 停止动作
                    self.add_node("Stp", tra_nodes, orig_idx)
                    tra_logps[orig_idx].append(logp if logp is not None else 0)
                    
                else:  # 整数动作，选择节点
                    cand_list = batch_candidates[j]
                    if ac >= len(cand_list):  # 动作超出候选范围，也视为停止
                        self.add_node("Stp", tra_nodes, orig_idx)
                        tra_logps[orig_idx].append(logp)
                    else:  # 正常选择节点
                        selected_node = cand_list[ac]
                        newvec = self.add_node(selected_node, tra_nodes, orig_idx)
                        
                        if newvec is not None:  # 成功添加节点
                            # 更新轨迹向量（平均池化）
                            current_len = len(tra_nodes[orig_idx])
                            tra_vector[orig_idx] = self.vecpool(
                                tra_vector[orig_idx],
                                newvec,
                                current_len
                            )
                            tra_logps[orig_idx].append(logp)
            
            step += 1
        
        return tra_nodes, tra_logps

    def trainReward(self, seeds: List[int], true_coms):
        '''
        通过奖励更新参数
        @param seeds: 一个batch的节点
        @param true_coms: 节点对应的真是社区
        '''
        bs = len(seeds)
        self.model.train()
        self.optimizer.zero_grad()
        selected_nodes, logps = self.sample_bs_trajectories(seeds)

        lengths = torch.LongTensor([len(x) for x in selected_nodes]).to(self.device)

        # 计算奖励
        rewards_list = []
        for index in range(len(selected_nodes)):
            com = selected_nodes[index]
            true_com = true_coms[index]
            r, gamma = [], 0.99
            temp_com = [com[0]]
            for node in com[1:]:
                if node != 'EOS':
                    _, _, pre_cost, _ = self.eval_scores(temp_com, true_com)
                    temp_com.append(node)
                    _, _, after_cost, _ = self.eval_scores(temp_com, true_com)
                    r.append(after_cost - pre_cost)
            reward = [np.sum(r[i] * (gamma ** np.array(range(i, len(r))))) for i in range(len(r))]
            rewards_list.append(reward)
        
        # 填充rewards和logps到相同长度
        max_len = max(len(r) for r in rewards_list) if rewards_list else 0
        max_logp_len = max(len(lp) for lp in logps) if logps else 0
        max_len = max(max_len, max_logp_len)
        
        rewards_padded = np.zeros((bs, max_len))
        logps_padded = np.zeros((bs, max_len))
        
        for i, r in enumerate(rewards_list):
            rewards_padded[i, :len(r)] = r
        for i, lp in enumerate(logps):
            logps_padded[i, :len(lp)] = lp
        
        rewards = torch.from_numpy(rewards_padded).float().to(self.device)
        logps = torch.tensor(logps_padded, device=self.device, dtype=torch.float32, requires_grad=True)
        
        mask = torch.arange(rewards.size(1), device=self.device,
                            dtype=torch.int64).expand(bs, -1) < (lengths - 1).unsqueeze(1)
        mask = mask.float()

       
        policy_loss = -(rewards * logps * mask).sum()
        
        loss = policy_loss
        # print("rewards:", rewards)
        # print("logps:", logps)
        # print("mask:",mask)
        # print("loss",loss)
        loss.backward()

        ###test    
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.6f}")
        
        # 检查参数变化
        old_params = [p.data.clone() for p in self.model.parameters()]
        
        self.optimizer.step()
        
        # 检查参数是否更新
        for i, (old, p) in enumerate(zip(old_params, self.model.parameters())):
            diff = (old - p.data).abs().sum().item()
            print(f"Layer {i} change: {diff:.6f}")

   


        

     

from dataProcess import DataProcess
from torch import optim
import random
import numpy as np

def test(epochs=30):
    d = DataProcess(dfname="PlusTokenPonzi")
    g = Graph(dffeature=d.train_feature, dfhacker=d.train_hacker, dfnode=d.train_nodes)
    
    model = Agent(input_size=88, hidden_size=128)
    e = Expander(graph=g, optimizer=optim.Adam(model.parameters(), lr=0.001), model=model)
    
    history = {'loss': [], 'f1': []}
    
    for epoch in range(epochs):
        # 训练
        seeds = random.sample(g.df_hacker["address"].values.tolist(), k=3)
        truecom = [g.sampleTrajectory(s) for s in seeds]
        e.trainReward(seeds=seeds, true_coms=truecom)
        
        # 评估
        e.model.eval()
        tra_nodes, _ = e.sample_bs_trajectories(seeds)
        f1s = [e.eval_scores(tra, truecom[i])[2] for i, tra in enumerate(tra_nodes)]
        avg_f1 = np.mean(f1s)
        
        history['f1'].append(avg_f1)
        print(f"Epoch {epoch+1}: F1={avg_f1:.4f}")
    
    print(f"\nBest F1: {max(history['f1']):.4f}")
    return history

if __name__ == "__main__":
    test()