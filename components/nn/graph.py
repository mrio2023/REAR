import pandas as pd
import numpy as np
import random


class Graph:
    def __init__(self, dfnode, dffeature, dfhacker, dfedge):
        self.df_nodes = dfnode
        self.df_feature = dffeature
        self.df_edge = dfedge
        self.df_hacker = dfhacker

        # ========== 优化1：批量构建邻接表（替代逐行iterrows） ==========
        self.adjmap = self._cal_adj_map_batch(self.df_edge, direction="from_to")
        self.adjtomap = self._cal_adj_map_batch(self.df_edge, direction="to_from")

        # ========== 优化2：提前过滤空值+批量构建标签映射 ==========
        self.allNameTags = self._get_valid_tags()
        self.tag2idx = {tag: i for i, tag in enumerate(self.allNameTags)}
        self.n_nodes = len(self.df_nodes)

        # ========== 优化3：批量缓存社区种子（减少循环开销） ==========
        self.community_seeds = self._cache_community_seeds_batch()
        self.embedsize = self.initEmbedSize()

        # ========== 核心优化：初始化时预加载所有嵌入到内存 ==========
        self.embed_cache, self.embed_dim = self._preload_embeddings()
        # 邻居缓存（辅助优化，减少重复计算）
        self.neighbor_cache = {}

    def initEmbedSize(self):
        return self.df_feature.shape[1] - 1 if not self.df_feature.empty else 0

    # ========== 嵌入相关核心优化（保留+小幅精简） ==========
    def _preload_embeddings(self):
        """批量加载嵌入，用向量化操作替代逐行遍历"""
        embed_cache = {}
        embed_dim = 0

        if self.df_feature.empty or "address" not in self.df_feature.columns:
            return embed_cache, embed_dim

        # 优化：一次性提取所有地址和嵌入，避免逐行循环
        embed_dim = len(self.df_feature.columns) - 1
        if embed_dim == 0:
            return embed_cache, embed_dim

        # 向量化提取嵌入矩阵
        addresses = self.df_feature["address"].values
        embed_matrix = self.df_feature.iloc[:, 1:].values.astype(np.float32)

        # 统一维度（向量化操作，比循环快10倍+）
        if embed_matrix.shape[1] != embed_dim:
            pad_width = max(0, embed_dim - embed_matrix.shape[1])
            if pad_width > 0:
                embed_matrix = np.pad(
                    embed_matrix, ((0, 0), (0, pad_width)), mode="constant"
                )
            else:
                embed_matrix = embed_matrix[:, :embed_dim]

        # 批量构建缓存字典
        embed_cache = dict(zip(addresses, embed_matrix))

        return embed_cache, embed_dim

    def singleNodeEmbed(self, node):
        if node not in self.embed_cache:
            raise ValueError(
                f"节点 {node} 不存在于嵌入缓存中 | 缓存节点数：{len(self.embed_cache)}"
            )
        return self.embed_cache[node]

    def nodesEmbed(self, nodes: list):
        embeds = np.array(
            [
                self.embed_cache.get(n, np.zeros(self.embed_dim, dtype=np.float32))
                for n in nodes
            ],
            dtype=np.float32,
        )
        return embeds

    # ========== 邻居查询（保留原有优化） ==========
    def getSingleNodeNeighbor(self, node: str):
        if node in self.neighbor_cache:
            return self.neighbor_cache[node].copy()

        neigh = [d["to"] for d in self.adjmap.get(node, [])] + [
            d["from"] for d in self.adjtomap.get(node, [])
        ]
        neigh = list(set(neigh))
        self.neighbor_cache[node] = neigh
        return neigh.copy()

    def get_1hop_subgraph(self, node: str):
        neighbors = self.getSingleNodeNeighbor(node)
        return set([node] + neighbors)

    def getNodesNeigh(self, nodes: list):
        res = set()
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))
        return list(res)

    # ========== 批量优化的核心函数 ==========
    def _get_valid_tags(self):
        """批量获取有效标签，过滤空值"""
        return sorted(self.df_hacker["name_tag"].dropna().unique())

    def _cal_adj_map_batch(self, df_edge: pd.DataFrame, direction: str = "from_to"):
        """批量构建邻接表，替代逐行iterrows"""
        if df_edge.empty:
            return {}

        adj_map = {}
        if direction == "from_to":
            # 按from分组，批量聚合to节点
            grouped = df_edge.groupby("from")["to"].apply(list).to_dict()
            # 转换为原有格式（保持兼容性）
            adj_map = {k: [{"to": v_item} for v_item in v] for k, v in grouped.items()}
        else:  # to_from
            grouped = df_edge.groupby("to")["from"].apply(list).to_dict()
            adj_map = {
                k: [{"from": v_item} for v_item in v] for k, v in grouped.items()
            }

        return adj_map

    def _cache_community_seeds_batch(self):
        """批量缓存社区种子，减少循环开销"""
        if self.df_hacker.empty:
            return {}

        # 优化：按name_tag分组，批量提取地址并转集合
        community_seeds = (
            self.df_hacker.groupby("name_tag")["address"]
            .apply(lambda x: set(x.tolist()))  # 批量转集合去重
            .to_dict()
        )
        return community_seeds

    def sampleTrajectory(self, node: str, maxlen: int):
        """
        随机游走采样（优先选无重复节点，无则选重复节点）
        Args:
            node: 起始节点（用于匹配所属hacker社区）
            maxlen: 采样最大长度（核心控制，防止节点数过多）
        Returns:
            采样后的轨迹列表（长度≤maxlen，优先无重复，无则选重复）
        """
        # 提前构建addr2tag映射（避免重复查询df）
        if not hasattr(self, "_addr2tag"):
            self._addr2tag = dict(
                zip(self.df_hacker["address"], self.df_hacker["name_tag"])
            )

        # 初始化轨迹，起始节点必包含
        tra = [node]
        if len(tra) >= maxlen:
            return tra[:maxlen]

        # 获取节点所属hacker标签（确定采样范围）
        name_tag = self._addr2tag.get(node)
        if not name_tag:
            # 无所属社区，仅返回起始节点
            return tra

        # 获取该标签下的所有社区节点（采样范围）
        community_nodes = self.community_seeds.get(name_tag, set())
        if not community_nodes:
            return tra

        # ========== 核心：优先无重复节点，无则选重复节点 ==========
        # 已访问节点集合（用于判断是否重复）
        visited_nodes = set(tra)
        # 当前游走节点
        current_node = node

        # 循环采样，直到达到maxlen
        while len(tra) < maxlen:
            # 1. 获取当前节点的所有社区内邻居
            neighbors = self.getSingleNodeNeighbor(current_node)
            all_candidates = [n for n in neighbors if n in community_nodes]

            if not all_candidates:
                break  # 无社区内邻居，终止游走

            # 2. 拆分未访问/已访问候选节点
            unvisited_candidates = [n for n in all_candidates if n not in visited_nodes]
            visited_candidates = [n for n in all_candidates if n in visited_nodes]

            # 3. 核心逻辑：有未访问节点就必选，没有才选已访问的
            if unvisited_candidates:
                # 有未访问节点 → 随机选一个未访问的
                selected_node = random.choice(unvisited_candidates)
                visited_nodes.add(selected_node)  # 标记为已访问
            else:
                # 无未访问节点 → 随机选一个已访问的
                selected_node = random.choice(visited_candidates)

            # 4. 更新轨迹和当前节点
            tra.append(selected_node)
            current_node = selected_node

        # 最终兜底：确保长度不超过maxlen
        return tra[:maxlen]
