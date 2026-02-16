import copy
import time
from typing import Union, Optional, List, Set
import  pandas as pd
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
        debug: bool = True,  # 新增调试开关
    ):
        self.graph = graph
        self.model = model
        self.optimizer = optimizer
        self.conv = GNN(graph, k, alpha)
        self.gamma = gamma
        self.maxLen = maxLen
        self.done = []
        self.device = device or torch.device("cpu")
        self.debug = debug  # 调试模式开关
        # 记录模型初始参数（用于检查是否更新）
        self.init_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # ========== 新增：缓存社区全邻居 ==========
        self.community_full_neighbors = self._cache_community_full_neighbors()

    def _cache_community_full_neighbors(self):
        """缓存每个社区的全邻居（所有社区节点的出边+入边邻居）"""
        full_neighbors = {}
        for tag in self.graph.allNameTags:
            # 获取该社区所有种子节点
            com_nodes = self.graph.community_seeds.get(tag, set())
            if not com_nodes:
                full_neighbors[tag] = []
                continue
            # 获取该社区所有节点的双向邻居（出边+入边）
            com_neighbors = self.graph.getNodesNeigh(list(com_nodes))
            full_neighbors[tag] = com_neighbors
        self.log_debug(f"社区全邻居缓存完成，覆盖 {len(full_neighbors)} 个社区")
        return full_neighbors

    def get_node_community(self, node: str):
        """获取节点所属的社区标签"""
        node_com_row = self.graph.df_hacker[self.graph.df_hacker["address"] == node]
        if node_com_row.empty or pd.isna(node_com_row["name_tag"].iloc[0]):
            return None
        return node_com_row["name_tag"].iloc[0]

    def log_debug(self, msg: str):
        """统一的调试日志输出"""
        if self.debug:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[DEBUG {timestamp}] {msg}")

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
            # 调试：输出logits基本信息
            self.log_debug(f"Batch {batch_idx} logits - shape: {batch_logits.shape if batch_logits is not None else 'None'}, numel: {batch_logits.numel() if batch_logits is not None else 0}")
            
            # 处理空张量
            if batch_logits is None or batch_logits.numel() == 0 or batch_logits.size(0) == 0:
                self.log_debug(f"Batch {batch_idx} - 空logits，选择Stp动作")
                actions.append("Stp")
                log_probs.append(None)
                continue

            # 确保logits是1D
            if batch_logits.dim() == 2:
                self.log_debug(f"Batch {batch_idx} - 转换2D logits到1D，原始shape: {batch_logits.shape}")
                if batch_logits.size(0) == 1:
                    batch_logits = batch_logits.squeeze(0)
                elif batch_logits.size(0) > 1:
                    batch_logits = batch_logits[0]
            
            # 再次检查是否有效
            if batch_logits.numel() == 0:
                self.log_debug(f"Batch {batch_idx} - 处理后仍为空，选择Stp动作")
                actions.append("Stp")
                log_probs.append(None)
                continue
            
            # 计算概率分布
            log_probs_dist = F.log_softmax(batch_logits, dim=-1)
            probs = torch.exp(log_probs_dist).detach().cpu().numpy()
            probs = probs / (probs.sum() + 1e-8)  # 确保和为1，避免除零
            self.log_debug(f"Batch {batch_idx} - 概率分布: {probs[:5]}... (sum: {probs.sum():.6f})")
            
            # 采样
            if len(probs) == 1:
                action = 0
                self.log_debug(f"Batch {batch_idx} - 仅1个候选，选择动作0")
            else:
                try:
                    action = np.random.choice(len(probs), p=probs)
                    self.log_debug(f"Batch {batch_idx} - 随机采样动作: {action}")
                except ValueError as e:
                    self.log_debug(f"Batch {batch_idx} - 采样失败({e})，降级选择最大概率动作")
                    action = np.argmax(probs)  # 降级处理
            
            # 确保log_prob是标量
            log_prob = log_probs_dist[action]
            if log_prob.dim() > 0:
                log_prob = log_prob.squeeze()
            
            actions.append(int(action))
            log_probs.append(log_prob.cpu())
            self.log_debug(f"Batch {batch_idx} - 最终动作: {action}, log_prob: {log_prob.item() if log_prob is not None else 'None'}")
        
        return actions, log_probs

    def prepare_inputs(self, tra_nodes):
        """准备模型输入 - 核心改动：改用全社区邻居作为候选节点"""
        choices = []
        indptr = []
        offset = 0
        
        for tra_idx, tra in enumerate(tra_nodes):
            if not tra:
                indptr.append([offset, offset, offset])
                self.log_debug(f"轨迹{tra_idx} - 空轨迹，indptr: [0,0,0]")
                continue
            
            cur = tra[-1]
            # ========== 核心改动1：获取全社区邻居作为候选 ==========
            # 步骤1：获取当前节点所属社区
            node_com = self.get_node_community(cur)
            if node_com:
                # 步骤2：获取该社区的全邻居（所有社区节点的出边+入边邻居）
                candidates = self.community_full_neighbors.get(node_com, [])
                self.log_debug(f"轨迹{tra_idx} - 当前节点{cur}属于社区{node_com}，全社区邻居数: {len(candidates)}")
            else:
                # 兜底：获取当前节点的双向邻居（出边+入边）
                candidates = self.graph.getSingleNodeNeighbor(cur)
                self.log_debug(f"轨迹{tra_idx} - 当前节点{cur}无社区标签，使用自身双向邻居数: {len(candidates)}")
            
            # 兜底：无候选时用-1
            if not candidates:
                candidates = [-1]
            
            # 记录指针：先候选节点，后轨迹节点
            start = offset
            cand_start = offset + len(candidates)
            end = offset + len(candidates) + len(tra)
            indptr.append([start, end, cand_start])
            
            choices.extend(candidates + tra)
            offset = end
            
            self.log_debug(f"轨迹{tra_idx} - 当前节点: {cur}, 候选节点数: {len(candidates)}, 轨迹长度: {len(tra)}, indptr: [{start}, {end}, {cand_start}]")
        
        self.log_debug(f"总候选列表长度: {len(choices)}, indptr数量: {len(indptr)}")
        return choices, torch.tensor(indptr, device=self.device)

    def add_node(self, new_node, tra_nodes, index):
        """添加节点到轨迹"""
        if len(self.done) <= index:
            self.done.extend([False] * (index + 1 - len(self.done)))
        
        # 终止条件
        if new_node in (None, "Stp", -1) or len(tra_nodes[index]) >= self.maxLen:
            self.done[index] = True
            self.log_debug(f"轨迹{index} - 终止扩展，原因: {'新节点为Stp/None/-1' if new_node in (None, 'Stp', -1) else f'达到最大长度{self.maxLen}'}")
            return None
        
        tra_nodes[index].append(new_node)
        embed = self.graph.singleNodeEmbed(new_node)  # 修复：改用self.graph而非parentGraph
        self.log_debug(f"轨迹{index} - 添加节点{new_node}，新轨迹长度: {len(tra_nodes[index])}")
        return embed

    def all_done(self):
        """检查是否全部完成"""
        done_status = all(self.done) if self.done else False
        self.log_debug(f"所有轨迹完成状态: {done_status} (当前done列表: {self.done})")
        return done_status

    def calc_rewards(self, tra_nodes, true_comms=None):
        """计算奖励"""
        rewards = []
        
        for i, tra in enumerate(tra_nodes):
            if true_comms and i < len(true_comms):
                p, r, f1, j = self.eval_scores(tra, true_comms[i])
                rewards.append([f1])
                self.log_debug(f"轨迹{i} - 奖励计算（有真实标签）: P={p}, R={r}, F1={f1}, 奖励值: {f1}")
            else:
                reward_val = len(tra) / self.maxLen
                rewards.append([reward_val])
                self.log_debug(f"轨迹{i} - 奖励计算（无真实标签）: 轨迹长度{len(tra)}/{self.maxLen}, 奖励值: {reward_val}")
        
        return rewards

    def check_gradient_flow(self):
        """检查模型梯度是否有效传播"""
        self.log_debug("=== 梯度检查 ===")
        has_gradient = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                has_gradient = True
                self.log_debug(f"参数 {name} - 梯度范数: {grad_norm:.6f}, 梯度是否全零: {torch.all(param.grad == 0).item()}")
            else:
                self.log_debug(f"参数 {name} - 无梯度")
        
        if not has_gradient:
            self.log_debug("⚠️ 警告：所有参数都没有梯度！")
        return has_gradient

    def check_parameter_update(self):
        """检查模型参数是否更新"""
        self.log_debug("=== 参数更新检查 ===")
        params_updated = False
        for name, param in self.model.named_parameters():
            init_param = self.init_params[name]
            if not torch.allclose(param, init_param):
                params_updated = True
                diff_norm = torch.norm(param - init_param).item()
                self.log_debug(f"参数 {name} - 已更新，差异范数: {diff_norm:.6f}")
            else:
                self.log_debug(f"参数 {name} - 未更新")
        
        if not params_updated:
            self.log_debug("⚠️ 警告：模型参数未更新！")
        return params_updated

    def train_rl(self, seeds: list, true_comms: List[Set[int]] = None, episodes: int = 30):
        """
        REINFORCE算法训练
        """
        self.log_debug(f"=== 开始RL训练 === 种子节点: {seeds}, 真实社区数: {len(true_comms) if true_comms else 0}, 最大步数: {self.maxLen}")
        
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
            self.log_debug(f"\n=== 第 {step} 步扩展 ===")
            
            # 准备输入
            choices, indptr = self.prepare_inputs(tra_nodes)
            
            # 转换为tensor
            seed_tensor = torch.tensor(np.array(seed_embeds), device=self.device, dtype=torch.float32)
            tra_tensor = torch.tensor(np.array(tra_embeds), device=self.device, dtype=torch.float32)
            self.log_debug(f"模型输入 - seed_tensor shape: {seed_tensor.shape}, tra_tensor shape: {tra_tensor.shape}, indptr shape: {indptr.shape}")
            
            # 前向传播
            self.model.train()  # 确保模型在训练模式
            logits = self.model(seed_tensor, tra_tensor, indptr)
            self.log_debug(f"模型输出logits数量: {len(logits)}, 第一个logits shape: {logits[0].shape if logits else 'None'}")
            
            # 采样动作
            actions, step_log_probs = self.sample_actions(logits[:len(tra_nodes)])
            
            # 收集对数概率
            valid_log_probs_count = 0
            for i, (act, log_prob) in enumerate(zip(actions, step_log_probs)):
                if not self.done[i] and log_prob is not None:
                    all_log_probs.append(log_prob)
                    valid_log_probs_count += 1
            self.log_debug(f"第{step}步 - 有效对数概率数量: {valid_log_probs_count}")
            
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
            updated_embeds_count = 0
            for i, embed in enumerate(new_embeds):
                if embed is not None and i < len(tra_embeds):
                    tra_len = len(tra_nodes[i])
                    tra_embeds[i] = (tra_embeds[i] * (tra_len - 1) + embed) / tra_len
                    updated_embeds_count += 1
            self.log_debug(f"第{step}步 - 更新嵌入数量: {updated_embeds_count}")
            
            # 记录奖励
            step_rewards = self.calc_rewards(tra_nodes, true_comms)
            all_rewards.append(step_rewards)
            self.log_debug(f"第{step}步 - 奖励列表: {[r[0] for r in step_rewards]}")
            
            step += 1
        
        # 如果没有有效动作，提前返回
        if not all_log_probs:
            self.log_debug("⚠️ 没有收集到有效对数概率，跳过优化步骤")
            return {
                "loss": 0, 
                "expanded_communities": tra_nodes, 
                "steps": step, 
                "final_rewards": []
            }
        
        self.log_debug(f"\n=== 开始优化 ===")
        self.log_debug(f"总有效对数概率数量: {len(all_log_probs)}, 总奖励步数: {len(all_rewards)}")
        
        # 计算折扣奖励 - 保持每个轨迹的奖励分开
        discounted_per_trajectory = []  # 二维列表
        
        for tra_idx in range(len(tra_nodes)):
            tra_rew = [r[tra_idx][0] for r in all_rewards if tra_idx < len(r)]
            self.log_debug(f"轨迹{tra_idx} - 原始奖励序列: {tra_rew}")
            
            disc = []
            cum = 0
            for r in reversed(tra_rew):
                cum = r + self.gamma * cum
                disc.append(cum)
            disc.reverse()
            discounted_per_trajectory.append(disc)
            self.log_debug(f"轨迹{tra_idx} - 折扣奖励序列: {disc}")
        
        # 将所有奖励展平用于损失计算
        discounted_flat = []
        for tra_disc in discounted_per_trajectory:
            discounted_flat.extend(tra_disc)
        
        # 对齐长度
        min_len = min(len(all_log_probs), len(discounted_flat))
        self.log_debug(f"对数概率数量: {len(all_log_probs)}, 折扣奖励数量: {len(discounted_flat)}, 对齐后长度: {min_len}")
        
        if min_len == 0:
            self.log_debug("⚠️ 对齐后长度为0，跳过优化步骤")
            return {
                "loss": 0, 
                "expanded_communities": tra_nodes, 
                "steps": step, 
                "final_rewards": []
            }
        
        # 计算损失
        log_probs_tensor = torch.stack(all_log_probs[:min_len])
        rewards_tensor = torch.tensor(discounted_flat[:min_len], device=self.device, dtype=torch.float32)
        self.log_debug(f"损失计算 - log_probs shape: {log_probs_tensor.shape}, rewards shape: {rewards_tensor.shape}")
        self.log_debug(f"原始奖励统计 - 均值: {rewards_tensor.mean().item():.6f}, 标准差: {rewards_tensor.std().item():.6f}")
        
        # 标准化奖励
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            self.log_debug(f"标准化后奖励统计 - 均值: {rewards_tensor.mean().item():.6f}, 标准差: {rewards_tensor.std().item():.6f}")
        
        loss = -(log_probs_tensor * rewards_tensor).sum()
        self.log_debug(f"计算得到损失值: {loss.item():.6f}")
        
        # 优化
        self.optimizer.zero_grad()
        self.log_debug("优化器梯度已清零")
        
        loss.backward()
        self.log_debug("损失反向传播完成")
        
        # 检查梯度流
        self.check_gradient_flow()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.log_debug(f"梯度裁剪后总范数: {grad_norm:.6f}")
        
        self.optimizer.step()
        self.log_debug("优化器参数更新完成")
        
        # 检查参数是否更新
        self.check_parameter_update()
        
        # 计算每个轨迹的最终奖励总和
        final_rewards = [sum(disc) for disc in discounted_per_trajectory]
        self.log_debug(f"\n=== 训练完成 ===")
        self.log_debug(f"总步数: {step}, 最终损失: {loss.item():.6f}, 各轨迹最终奖励: {final_rewards}")
        
        return {
            "loss": loss.item(),
            "expanded_communities": tra_nodes,
            "steps": step,
            "final_rewards": final_rewards
        }


# ===================== 测试代码 =====================
if __name__ == "__main__":
    # 1. 加载数据（确保数据路径/名称正确）
    d = DataProcess(dfname="PlusTokenPonzi")
    g = Graph(dffeature=d.train_feature, dfhacker=d.train_hacker, dfnode=d.train_nodes)
    # 注意：initKego()若依赖k-ego子图需注释，否则保留
    # g = Tool(graph=g).initKego()
    
    # 2. 初始化模型（input_size需匹配节点嵌入维度）
    agent = Agent(hidden_size=128, input_size=84)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    expander = Expander(
        graph=g, 
        model=agent, 
        optimizer=optimizer, 
        device=torch.device("cpu"),
        debug=True
    )
    
    # 3. 准备种子和真实社区
    hackers = g.df_hacker["address"].values.tolist()
    if len(hackers) == 0:
        print("警告：无黑客节点数据！")
        exit()
    seeds = random.sample(hackers, min(3, len(hackers)))
    
    # 生成真实社区轨迹（使用全社区邻居的sampleTrajectory）
    true_comms = []
    for s in seeds:
        traj = g.sampleTrajectory(node=s, min_community_ratio=0.6)
        true_comm = [n for n in traj if n != "stp"]
        true_comms.append(true_comm)
    print(f"生成真实社区: {true_comms}")
    
    # 4. 训练
    print("\n=== 开始测试训练 ===")
    try:
        result = expander.train_rl(
            seeds=seeds, 
            true_comms=true_comms, 
            episodes=30
        )
    except Exception as e:
        import traceback
        print(f"训练出错: {e}")
        traceback.print_exc()
        exit()
    
    # 5. 输出结果
    print(f"\n=== 最终结果 ===")
    print(f"损失: {result.get('loss', 0):.4f}")
    print(f"步数: {result.get('steps', 0)}")
    print(f"真实社区: {true_comms}")
    
    expanded_comms = result.get("expanded_communities", [])
    for i, (seed, comm) in enumerate(zip(seeds, expanded_comms)):
        print(f"\n种子{i}: {seed}")
        print(f"  扩展社区 ({len(comm)}个节点): {comm[:10]}..." if len(comm)>10 else f"  扩展社区 ({len(comm)}个节点): {comm}")
        if i < len(true_comms) and true_comms[i]:
            p, r, f1, j = expander.eval_scores(comm, true_comms[i])
            print(f"  评估: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, J={j:.4f}")
        else:
            print(f"  评估: 无真实社区数据")