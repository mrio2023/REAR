import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


class Graph:
    def __init__(
        self, adj: dict, features: dict, communities: dict, node_list: list = None
    ):
        """
        从预处理数据直接构建图对象。

        :param adj: 邻接表，格式 {node: set(neighbors)}
        :param features: 节点特征字典，格式 {node: list/ndarray}
        :param communities: 社区字典，格式 {community_name: list_of_nodes}
        :param node_list: 可选，所有节点的列表，用于快速遍历
        """
        self.adj = adj
        self.features = features
        self.communities = communities
        self.community_seeds = communities  # 添加此行，兼容 eval_model

        # 构建从节点到社区名的映射（用于 sampleTrajectory）
        self.node_to_community = {}
        for name, nodes in communities.items():
            for n in nodes:
                self.node_to_community[n] = name

        # 特征维度
        self.embed_dim = 0
        if features:
            sample_feat = next(iter(features.values()))
            self.embed_dim = (
                len(sample_feat) if isinstance(sample_feat, (list, np.ndarray)) else 1
            )
        print(
            f"初始化 Graph：节点数 {len(adj)}，社区数 {len(communities)}，特征维度 {self.embed_dim}"
        )

        # 预加载所有节点嵌入（向量化处理）
        self.embed_cache = {}
        for node, feat in features.items():
            if isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            elif not isinstance(feat, np.ndarray):
                feat = np.array([feat], dtype=np.float32)
            # 处理全零向量（添加微小噪声）
            if np.linalg.norm(feat) < 1e-8:
                feat += np.random.normal(0, 1e-6, size=feat.shape).astype(np.float32)
            self.embed_cache[node] = feat

        # 邻居缓存
        self.neighbor_cache = {}

    @property
    def embedsize(self):
        return self.embed_dim

    # 其余方法保持不变 ...
    def initEmbedSize(self):
        return self.embed_dim

    def singleNodeEmbed(self, node):
        """返回单个节点的嵌入向量"""
        if node not in self.embed_cache:
            # 如果节点没有特征，返回零向量
            return np.zeros(self.embed_dim, dtype=np.float32)
        return self.embed_cache[node]

    def nodesEmbed(self, nodes: list):
        """返回多个节点的嵌入矩阵"""
        embeds = []
        for n in nodes:
            embeds.append(self.singleNodeEmbed(n))
        return np.array(embeds, dtype=np.float32)

    def getSingleNodeNeighbor(self, node):
        """返回节点的邻居列表（去重）"""
        if node in self.neighbor_cache:
            return self.neighbor_cache[node].copy()
        neighbors = self.adj.get(node, set())
        # 确保返回列表
        neigh_list = list(neighbors)
        self.neighbor_cache[node] = neigh_list
        return neigh_list.copy()

    def getNodesNeigh(self, nodes: list):
        """返回多个节点的邻居集合"""
        res = set()
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))
        return list(res)

    def sampleTrajectory(self, node: str, maxlen: int = None):
        """
        返回节点所属整个社区的所有节点（去重）。
        原为随机游走，现直接返回社区全量节点。
        maxlen 参数保留但未使用，保持接口兼容。
        """
        community_name = self.node_to_community.get(node)
        if community_name:
            return list(self.communities[community_name])
        else:
            # 若节点不属于任何社区，返回仅包含自身的列表
            return [node]

    # 可选：添加图的基本信息
    @property
    def nodes(self):
        return list(self.adj.keys())

    @property
    def n_nodes(self):
        return len(self.adj)
