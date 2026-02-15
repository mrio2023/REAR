import copy
from typing import Union, Optional, List, Set, Dict
import numpy as np
from scipy import sparse as sp
from sklearn.decomposition import TruncatedSVD

import torch
from torch import nn
import torch.nn.functional as F

# 测试引用
from dataProcess import DataProcess
import random

# 测试引用

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
        self.done = []  # 初始化done列表
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def eval_scores(self, pred_comm: Union[List, Set], true_comm: Union[List, Set]):
        """计算社区检测的评估指标：P/R/F1/Jaccard"""
        intersect = set(true_comm) & set(pred_comm)
        p = len(intersect) / len(pred_comm) if len(pred_comm) > 0 else 0.0
        r = len(intersect) / len(true_comm) if len(true_comm) > 0 else 0.0
        f = 2 * p * r / (p + r + 1e-9)
        j = len(intersect) / (len(pred_comm) + len(true_comm) - len(intersect) + 1e-9)
        return round(p, 4), round(r, 4), round(f, 4), round(j, 4)

    def tianchong(self, rewards, logps):
        """对齐rewards和logps的维度，填充0"""
        # 找到所有rewards中的最大长度
        max_len = max(len(r) for r in rewards) if rewards else 0

        # 每个reward填充到相同长度
        filled = []
        for r in rewards:
            # 在后面补0直到max_len
            padded = r + [0] * (max_len - len(r))
            filled.append(padded)

        return np.array(filled)

    def getLogits(self, seed_vector, tra_vector, indptr):
        """获取模型的logits输出"""
        model_inputs = [seed_vector, tra_vector, indptr]
        # 将输入移到指定设备
        model_inputs = [
            torch.cat(x).to(self.device) if isinstance(x, list) else x.to(self.device)
            for x in model_inputs
        ]
        return self.model(*model_inputs)

    def sampleActions(self, logits):
        """
        改造版：将logits转为list并完成采样，避免张量依赖
        Args:
            logits: Agent返回的logits列表（每个元素是torch.Tensor）
        Returns:
            list: 采样的动作索引（int）或"Stp"
        """
        res = []

        for batch_logits in logits:
            if batch_logits is None or batch_logits.numel() == 0:
                res.append("Stp")
                continue

            # 1. 将tensor转为numpy array，再转为list（核心改造）
            if batch_logits.dim() == 2 and batch_logits.size(0) == 1:
                batch_logits = batch_logits.squeeze(0)  # [num_candidates]
            logits_np = batch_logits.detach().cpu().numpy()  # 转到cpu并解图
            logits_list = logits_np.tolist()  # 转为纯list

            # 2. 计算softmax（原生Python实现）
            def softmax(x):
                e_x = np.exp(x - np.max(x))  # 防止数值溢出
                return e_x / e_x.sum(axis=0)
            
            probs = softmax(logits_list) + 1e-8  # 加小值避免0概率
            probs_list = probs.tolist()

            # 3. 采样（原生Python实现）
            if len(probs_list) == 1:
                action = 0
            else:
                # 按概率采样索引
                action = np.random.choice(len(probs_list), p=probs_list)
                action = int(action)  # 确保是int类型

            res.append(action)

        return res

    def prepareModelInputs(self, tra_nodes):
        """准备模型输入：构建选择空间和indptr"""
        choices = []  # 只存放候选节点（邻居）
        indptr = []  # 存放 [start, end, candidate_start] 三元组
        offset = 0

        for tra in tra_nodes:
            if not tra:
                indptr.append([offset, offset, offset])  # 空轨迹，无节点
                continue

            # 当前节点是轨迹的最后一个节点
            cur = tra[-1]

            # 获取当前节点的邻居作为候选节点
            data = self.graph.parentGraph.adjmap.get(cur, [])
            candidate_nodes = [d["to"] for d in data]

            if not candidate_nodes:
                candidate_nodes = [-1]  # 无邻居时添加终止标记

            # 涉及的节点：候选节点 + 当前轨迹
            involved_nodes = candidate_nodes + tra

            # 记录指针信息
            start = offset
            candidate_start = offset + len(candidate_nodes)
            end = offset + len(involved_nodes)

            indptr.append([start, end, candidate_start])

            # 将所有节点加入choices（按顺序：先候选节点，后轨迹节点）
            choices.extend(involved_nodes)
            offset = end

        indptr = torch.tensor(indptr, device=self.device)
        return choices, indptr

    def putNode(self, newNode, tra_nodes, index):
        """添加新节点到轨迹，或标记为完成"""
        # 初始化done列表（首次调用时）
        if len(self.done) <= index:
            self.done += [False] * (index + 1 - len(self.done))

        # 终止条件
        if (
            newNode is None
            or newNode == "Stp"
            or len(tra_nodes[index]) + 1 >= self.maxLen
        ):
            self.done[index] = True
            return None

        # 添加新节点到轨迹
        tra_nodes[index].append(newNode)
        # 返回新节点的嵌入
        return self.graph.parentGraph.singleNodeEmbed(newNode)

    def Alldone(self):
        """检查是否所有轨迹都已完成"""
        if len(self.done) <= 0:
            return False
        return all(self.done)

    def calculate_reward(
        self, tra_nodes: List[List[int]], true_communities: List[Set[int]] = None
    ):
        """计算每个轨迹的奖励（可根据实际需求调整）"""
        rewards = []
        for i, tra in enumerate(tra_nodes):
            if true_communities:
                # 如果有真实社区，使用评估指标作为奖励
                p, r, f1, j = self.eval_scores(tra, true_communities[i])
                reward = f1  # 使用F1作为奖励
            else:
                # 无真实社区时，使用社区内聚性作为奖励（简化版）
                # 实际需根据你的任务定义奖励函数
                reward = len(tra) / self.maxLen  # 临时奖励函数

            rewards.append([reward])
        return rewards
    def listtotensor(self,l:list):
        return torch.tensor(
            np.array(l),
            device=self.device,
            dtype=torch.float32,
        ) 
    def trainRL(
        self, seeds: list, true_communities: List[Set[int]] = None, episode: int = 30
    ):
        """
        强化学习训练主逻辑（REINFORCE算法）
        Args:
            seeds: 社区种子节点列表
            true_communities: 真实社区标签（用于计算奖励）
            episode: 训练轮数
        """
        # 初始化轨迹和嵌入
        tra_nodes = [[n] for n in seeds]  # 每个种子对应一个轨迹
        seed_vectors = torch.tensor(
            self.graph.nodesEmbed(seeds),
        )  # 种子节点嵌入
        tra_vectors = copy.deepcopy(seed_vectors)  # 轨迹嵌入
        self.done = [False] * len(seeds)  # 初始化done状态

        # 存储训练数据
        all_log_probs = []  # 所有动作的对数概率
        all_rewards = []  # 所有轨迹的奖励

        # 轨迹扩展循环
        step = 0
        while not self.Alldone() and step < self.maxLen:
            # 1. 准备模型输入
            choices, indptr = self.prepareModelInputs(tra_nodes)
            # 2.将torch和list分开，混在一起会报错 
            t_seed_vectors=self.listtotensor(seed_vectors)
            t_tra_vectores=self.listtotensor(tra_vectors)


            logits = self.getLogits(t_seed_vectors,t_tra_vectores,indptr)

            # 3. 采样动作
            actions = self.sampleActions(
                logits[: len(tra_nodes)]
            )  # 每个轨迹采样一个动作

            # 4. 执行动作（添加节点）
            # 4. 执行动作（添加节点）
            new_node_vectors = []
            print(actions)
            for i, action in enumerate(actions):

                if self.done[i]:
                    new_node_vectors.append(None)
                    continue

                # 判断action是否是字符串"Stp"
                if action == "Stp":
                    new_node = "Stp"
                else:
                    # action是整数索引
                    new_node = choices[action] if action < len(choices) else "Stp"

                # 添加节点并获取嵌入
                node_embed = self.putNode(new_node, tra_nodes, i)
                new_node_vectors.append(node_embed)

                # 添加节点并获取嵌入
                node_embed = self.putNode(new_node, tra_nodes, i)
                new_node_vectors.append(node_embed)

            # 5. 更新轨迹嵌入
            for i, embed in enumerate(new_node_vectors):
                if embed is not None:
                    idx = min(i, len(tra_vectors) - 1)  # 确保索引不越界
                    tra_vectors[idx] = (tra_vectors[idx] * len(tra_nodes[idx]) + embed) / (len(tra_nodes[idx]) + 1)
            # 6. 计算当前步骤奖励
            step_rewards = self.calculate_reward(tra_nodes, true_communities)
            all_rewards.append(step_rewards)

            step += 1

        # 7. 计算折扣奖励
        discounted_rewards = []
        for tra_idx in range(len(tra_nodes)):
            tra_rewards = [r[tra_idx][0] for r in all_rewards if tra_idx < len(r)]
            # 反向计算折扣奖励
            disc_reward = 0
            tra_disc_rewards = []
            for r in reversed(tra_rewards):
                disc_reward = r + self.gamma * disc_reward
                tra_disc_rewards.append(disc_reward)
            tra_disc_rewards.reverse()
            discounted_rewards.append(tra_disc_rewards)

        # 8. 对齐奖励和对数概率维度
        filled_rewards = self.tianchong(discounted_rewards, all_log_probs)
        filled_rewards = torch.tensor(
            filled_rewards, device=self.device, dtype=torch.float32
        )

        # 9. 计算损失
        log_probs_tensor = torch.stack(all_log_probs)  # [total_steps]
        # 将filled_rewards展平成1D
        rewards_flat = filled_rewards.flatten()
        # 确保长度匹配（取较短的那个）
        min_len = min(len(log_probs_tensor), len(rewards_flat))
        loss = -(log_probs_tensor[:min_len] * rewards_flat[:min_len]).sum()

        # 10. 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 11. 返回训练结果
        return {
            "loss": loss.item(),
            "expanded_communities": tra_nodes,
            "steps": step,
            "final_rewards": [sum(r) for r in discounted_rewards],
        }


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # ===================== 1. 数据加载与预处理 =====================
    # 加载PlusTokenPonzi数据集
    d = DataProcess(dfname="PlusTokenPonzi")

    # 解包数据（确保DataProcess类正确生成这些属性）
    train_nodes = d.train_nodes
    test_nodes = d.test_nodes
    train_feature = d.train_feature
    test_feature = d.test_feature
    train_hacker = d.train_hacker
    test_hacker = d.test_hacker

    # 构建图对象（基于训练集数据）
    g = Graph(dffeature=train_feature, dfhacker=train_hacker, dfnode=train_nodes)
    t = Tool(graph=g)
    g = t.initKego()

    # ===================== 2. 模型与扩展器初始化 =====================
    # 初始化Agent模型和优化器
    agent = Agent(hidden_size=128, input_size=84)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)

    # 创建Expander实例（补充必要的参数，如args/device等）
    # 先创建空的args对象（适配Expander的初始化参数）
    args = type("Args", (), {"lr": 0.001, "device": "cpu"})()
    expander = Expander(
        graph=g,
        model=agent,
        optimizer=optimizer,
        device=torch.device("cpu"),  # 如需GPU改为 'cuda'
        k=3,
        alpha=0.85,
        gamma=0.99,
        maxLen=16,
    )

    # ===================== 3. 准备种子节点和真实社区 =====================
    # 从黑客地址中随机采样3个作为种子节点
    hacker_addresses = g.df_hacker["address"].values.tolist()
    seeds = (
        random.sample(hacker_addresses, k=3)
        if len(hacker_addresses) >= 3
        else hacker_addresses
    )

    # 构建真实社区标签（按name_tag分组，每个种子对应其所属的真实社区）
    coms = g.df_hacker.groupby("name_tag")
    # 构建{地址: 所属社区}的映射
    address_to_community = {}
    true_communities_dict = {}
    for comm_name, comm_df in coms:
        # 提取该社区的所有地址
        comm_addresses = set(comm_df["address"].values.tolist())
        true_communities_dict[comm_name] = comm_addresses
        # 为每个地址绑定社区
        for addr in comm_addresses:
            address_to_community[addr] = comm_addresses

    # 为每个种子节点匹配对应的真实社区
    true_communities = []
    for seed in seeds:
        # 如果种子在社区映射中，取对应社区；否则取空集合
        true_comm = address_to_community.get(seed, set())
        true_communities.append(true_comm)

    # ===================== 4. 强化学习训练 =====================
    print(f"开始训练，种子节点: {seeds}")
    print(f"对应真实社区大小: {[len(c) for c in true_communities]}")

    # 执行RL训练
    result = expander.trainRL(
        seeds=seeds, true_communities=true_communities, episode=30  # 训练轮数
    )

    # ===================== 5. 训练结果打印 =====================
    print("\n===== 训练结果 =====")
    print(f"训练损失: {result['loss']:.4f}")
    print(f"扩展后的社区:")
    for i, (seed, comm) in enumerate(zip(seeds, result["expanded_communities"])):
        print(f"  种子{seed}: {comm} (长度: {len(comm)})")
    print(f"训练步数: {result['steps']}")
    print(f"最终奖励: {[round(r, 4) for r in result['final_rewards']]}")

    # 可选：计算扩展社区的评估指标
    print("\n===== 评估指标 (P/R/F1/Jaccard) =====")
    for i, (pred_comm, true_comm) in enumerate(
        zip(result["expanded_communities"], true_communities)
    ):
        p, r, f1, j = expander.eval_scores(pred_comm, true_comm)
        print(f"  种子{i+1}: P={p}, R={r}, F1={f1}, Jaccard={j}")
