import random
import pandas as pd
import numpy as np


class Graph:
    def __init__(self, dfnode, dffeature, dfhacker, dfedge):
        self.df_nodes = dfnode
        self.df_feature = dffeature
        self.df_edge = dfedge
        self.df_hacker = dfhacker

        self.adjmap = self.calAdjMap(self.df_edge)  # from → to（出边）
        self.adjtomap = self.calAdjToMap(self.df_edge)  # to → from（入边）
        self.allNameTags = sorted(self.df_hacker["name_tag"].dropna().unique())
        self.tag2idx = {tag: i for i, tag in enumerate(self.allNameTags)}
        self.n_nodes = len(self.df_nodes)
        self.community_seeds = self._cache_community_seeds()
        self.embedsize = self.initEmbedSize()

    def initEmbedSize(self):

        return self.df_feature.shape[1] - 1

    def get_1hop_subgraph(self, node: str):
        """获取节点的一阶子图（节点自身 + 所有直接邻居）"""
        neighbors = self.getSingleNodeNeighbor(node)
        subgraph_nodes = set([node] + neighbors)
        return subgraph_nodes

    def getNodesNeigh(self, nodes: list):
        """获取多个节点的所有邻居（出边+入边，去重）"""
        res = set()
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))
        return list(res)

    def getSingleNodeNeighbor(self, node: str):
        """获取单个节点的所有邻居（出边+入边）"""
        fromdata = self.adjmap.get(node, [])
        todata = self.adjtomap.get(node, [])
        neigh = [d["to"] for d in fromdata] + [d["from"] for d in todata]
        return list(set(neigh))

    def _cache_community_seeds(self):
        """缓存每个社区的种子节点"""
        community_seeds = {}
        for tag in self.allNameTags:
            seeds = set(
                self.df_hacker[self.df_hacker["name_tag"] == tag]["address"].tolist()
            )
            community_seeds[tag] = seeds
        return community_seeds

    def sampleTrajectory(self, node: str, traj_length: int):
        """
        采样轨迹（仅在同社区节点内部随机游走）
        异常场景（无社区/无有效节点/无候选邻居）直接返回当前轨迹，允许长度小于指定长度

        Args:
            node: 起始节点
            traj_length: 期望的轨迹长度（最大长度）
        Returns:
            tra: 采样后的轨迹（长度可能小于traj_length）
        """
        # 初始化轨迹，至少包含起始节点
        tra = [node]

        # 1. 无社区标签：直接返回仅包含起始节点的轨迹
        node_hacker_row = self.df_hacker[self.df_hacker["address"] == node]
        if node_hacker_row.empty or pd.isna(node_hacker_row["name_tag"].iloc[0]):
            return tra

        # 2. 获取社区节点，无有效社区节点：直接返回当前轨迹
        name_tag = node_hacker_row["name_tag"].iloc[0]
        community_nodes = self.community_seeds.get(name_tag, set())
        # 过滤出存在邻居的同社区节点（无邻居则视为无有效节点）
        community_nodes = {n for n in community_nodes if self.getSingleNodeNeighbor(n)}
        if not community_nodes:
            return tra

        # 3. 同社区内随机游走，无候选节点时直接返回
        while len(tra) < traj_length:
            current_node = tra[-1]
            current_neighbors = set(self.getSingleNodeNeighbor(current_node))
            # 仅保留同社区且未访问的候选节点
            candidates = [
                n for n in current_neighbors if n in community_nodes and n not in tra
            ]

            # 无候选节点：终止游走，返回当前轨迹
            if not candidates:
                return tra

            # 有候选节点：随机选择并加入轨迹
            selected_node = random.choice(candidates)
            tra.append(selected_node)

        # 4. 达到期望长度：返回完整轨迹
        return tra

    def singleNodeEmbed(self, node):
        """获取单个节点的嵌入向量（无str转换，直接匹配，无兜底）"""
        # 1. 基础校验
        if self.df_feature.empty:
            raise ValueError("特征表self.df_feature为空，无法获取节点嵌入")
        
        if 'address' not in self.df_feature.columns:
            raise KeyError("特征表缺少'address'列，无法匹配节点")
        
        # 2. 直接用原始类型匹配（移除所有str转换）
        # 核心：确保传入的node类型 和 df_feature['address']列类型完全一致
        node_row = self.df_feature[self.df_feature['address'] == node]
        
        # 3. 节点不存在直接报错
        if node_row.empty:
            # 报错时打印类型信息，便于排查类型不匹配问题
            node_type = type(node).__name__
            addr_type = self.df_feature['address'].dtype
            raise ValueError(
                f"节点 {node}（类型：{node_type}）不存在于特征表中 | "
                f"特征表address列类型：{addr_type}"
            )
        
        # 4. 提取嵌入向量并校验维度
        embed_vector = node_row.iloc[0, 1:].values.astype(np.float32)
        expected_dim = len(self.df_feature.columns) - 1  # 排除address列
        
        if len(embed_vector) != expected_dim:
            raise RuntimeError(
                f"节点 {node} 嵌入维度异常：预期{expected_dim}维，实际{len(embed_vector)}维"
            )
        
        return embed_vector

    def nodesEmbed(self, nodes: list):
        """批量获取节点嵌入"""
        embeds = [self.singleNodeEmbed(n) for n in nodes]

        # 确定目标维度
        target_dim = None
        for e in embeds:
            if e is not None:
                target_dim = len(e)
                break
        if target_dim is None:
            target_dim = 88

        # 统一嵌入维度
        result = []
        for e in embeds:
            if e is None:
                result.append(np.zeros(target_dim, dtype=np.float32))
            elif len(e) == target_dim:
                result.append(e)
            elif len(e) < target_dim:
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[: len(e)] = e
                result.append(padded)
            else:
                result.append(e[:target_dim])

        return result

    def calAdjMap(self, df_edge: pd.DataFrame):
        """构建出边邻接映射"""
        adj_map = dict()
        for _, row in df_edge.iterrows():
            u = row["from"]
            v = row["to"]
            edge_data = {"to": v}
            if u in adj_map:
                adj_map[u].append(edge_data)
            else:
                adj_map[u] = [edge_data]
        return adj_map

    def calAdjToMap(self, df_edge: pd.DataFrame):
        """构建入边邻接映射"""
        adj_map = dict()
        for _, row in df_edge.iterrows():
            u = row["to"]
            v = row["from"]
            edge_data = {"from": v}
            if u in adj_map:
                adj_map[u].append(edge_data)
            else:
                adj_map[u] = [edge_data]
        return adj_map


# from dataProcess import dataProcess
# def test():
#     """
#     适配dataProcess的测试函数：
#     1. 初始化数据处理类，获取训练数据
#     2. 初始化Graph类，参数对应训练集的node/feature/hacker/edge
#     3. 简单验证Graph核心功能（轨迹采样、节点嵌入）
#     """
#     try:
#         # 1. 初始化数据处理类（获取elliptic数据集的训练/测试拆分）
#         dp = dataProcess(dfname="elliptic",normal_node_ratio=1,min_community_size=5,expand_hop=3)
#         print("✅ dataProcess初始化成功")

#         # 2. 初始化Graph类（训练时使用train相关数据，node和hacker对应训练集）
#         # 核心适配：Graph的dfnode传train_nodes，dfhacker传train_hacker
#         g = Graph(
#             dfnode=dp.train_nodes,       # 训练节点数据
#             dffeature=dp.train_feature,  # 训练特征数据
#             dfhacker=dp.train_hacker,    # 训练黑客（社区）数据
#             dfedge=dp.train_edge         # 训练边数据
#         )
#         print("✅ Graph类初始化成功")

#         # 3. 简单验证核心功能（不删除原有方法，仅验证可用性）
#         if len(dp.train_hacker) > 0:
#             # 随机选一个训练集的黑客节点做轨迹采样
#             test_node = dp.train_hacker["address"].iloc[0]
#             traj = g.sampleTrajectory(node=test_node, traj_length=10)
#             print(f"✅ 轨迹采样成功，采样轨迹长度：{len(traj)}，轨迹：{traj[:5]}...")

#             # 验证节点嵌入功能
#             embed = g.singleNodeEmbed(test_node)
#             print(f"✅ 单个节点嵌入成功，嵌入维度：{len(embed)}")

#             # 验证邻居获取功能
#             neighbors = g.getSingleNodeNeighbor(test_node)
#             print(f"✅ 邻居获取成功，邻居数量：{len(neighbors)}")
#         else:
#             print("⚠️ 训练集无黑客节点，跳过功能验证")

#     except Exception as e:
#         print(f"❌ 测试失败：{e}")

# # 执行测试
# if __name__ == "__main__":
#     test()
