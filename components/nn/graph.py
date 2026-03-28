import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity


class Graph:
    def __init__(
        self, adj: dict, features: dict, communities: dict,
    ):
        
        self.adj = adj
        self.features = features
        self.communities = communities
        self.community_seeds = communities  
     
        self.node_to_community = {}
        for name, nodes in communities.items():
            for n in nodes:
                self.node_to_community[n] = name

        self.embed_dim = 0
        if features:
            sample_feat = next(iter(features.values()))
            self.embed_dim = (
                len(sample_feat) if isinstance(sample_feat, (list, np.ndarray)) else 1
            )
       

       
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
