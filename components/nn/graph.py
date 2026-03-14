import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


class Graph:
    def __init__(self, dfnode, dffeature, dfhacker, dfedge):
        self.df_nodes = dfnode
        self.df_feature = dffeature
        self.df_edge = dfedge
        self.df_hacker = dfhacker

        # 批量构建邻接表
        self.adjmap = self._cal_adj_map_batch(self.df_edge, direction="from_to")
        self.adjtomap = self._cal_adj_map_batch(self.df_edge, direction="to_from")

        # 批量缓存社区种子
        self.community_seeds = self._cache_community_seeds_batch()
        self.embedsize = self.initEmbedSize()

        # 预加载所有节点嵌入
        self.embed_cache, self.embed_dim = self._preload_embeddings()
        # 邻居缓存（避免重复计算）
        self.neighbor_cache = {}

        # 地址到标签的映射（用于 sampleTrajectory）
        if not self.df_hacker.empty:
            self.addr2tag = dict(
                zip(self.df_hacker["address"], self.df_hacker["name_tag"])
            )
        else:
            self.addr2tag = {}

    def initEmbedSize(self):
        return self.df_feature.shape[1] - 1 if not self.df_feature.empty else 0

    # ========== 嵌入相关 ==========
    def _preload_embeddings(self):
        """批量加载嵌入，处理全零向量"""
        embed_cache = {}
        embed_dim = 0

        if self.df_feature.empty or "address" not in self.df_feature.columns:
            return embed_cache, embed_dim

        embed_dim = len(self.df_feature.columns) - 1
        if embed_dim == 0:
            return embed_cache, embed_dim

        addresses = self.df_feature["address"].values
        embed_matrix = self.df_feature.iloc[:, 1:].values.astype(np.float32)

        # 统一维度
        if embed_matrix.shape[1] != embed_dim:
            pad_width = max(0, embed_dim - embed_matrix.shape[1])
            if pad_width > 0:
                embed_matrix = np.pad(
                    embed_matrix, ((0, 0), (0, pad_width)), mode="constant"
                )
            else:
                embed_matrix = embed_matrix[:, :embed_dim]

        # 处理全零向量（添加微小噪声）
        norms = np.linalg.norm(embed_matrix, axis=1)
        zero_mask = norms < 1e-8
        if zero_mask.any():
            print(f"[WARN] 发现 {zero_mask.sum()} 个全零嵌入向量，添加微小噪声")
            noise = np.random.normal(0, 1e-6, size=(zero_mask.sum(), embed_dim)).astype(
                np.float32
            )
            embed_matrix[zero_mask] = noise

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

    # ========== 邻居查询 ==========
    def getSingleNodeNeighbor(self, node: str):
        if node in self.neighbor_cache:
            return self.neighbor_cache[node].copy()

        neigh = [d["to"] for d in self.adjmap.get(node, [])] + [
            d["from"] for d in self.adjtomap.get(node, [])
        ]
        neigh = list(set(neigh))
        self.neighbor_cache[node] = neigh
        return neigh.copy()

    def getNodesNeigh(self, nodes: list):
        res = set()
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))
        return list(res)

    # ========== 辅助批量构建函数 ==========
    def _cal_adj_map_batch(self, df_edge: pd.DataFrame, direction: str = "from_to"):
        """批量构建邻接表"""
        if df_edge.empty:
            return {}

        adj_map = {}
        if direction == "from_to":
            grouped = df_edge.groupby("from")["to"].apply(list).to_dict()
            adj_map = {k: [{"to": v_item} for v_item in v] for k, v in grouped.items()}
        else:  # to_from
            grouped = df_edge.groupby("to")["from"].apply(list).to_dict()
            adj_map = {
                k: [{"from": v_item} for v_item in v] for k, v in grouped.items()
            }
        return adj_map

    def _cache_community_seeds_batch(self):
        """批量缓存社区种子"""
        if self.df_hacker.empty:
            return {}
        return (
            self.df_hacker.groupby("name_tag")["address"]
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )

    # ========== 采样轨迹（用于训练） ==========
    def sampleTrajectory(self, node: str, maxlen: int):
        """
        返回节点所属整个社区的所有节点（去重）。
        原为随机游走，现简化为社区全量节点。
        """
        name_tag = self.addr2tag.get(node)
        if not name_tag:
            return [node]
        community_nodes = self.community_seeds.get(name_tag, set())
        if not community_nodes:
            return [node]
        return list(community_nodes)