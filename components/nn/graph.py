import numpy as np
from sklearn.metrics.pairwise import cosine_similarity   # kept for potential external use, but not used inside class


class Graph:
    def __init__(self, adj: dict, features: dict, communities: dict):
        """
        Graph data structure for community detection.

        Args:
            adj: adjacency dict {node: set(neighbors)}
            features: node features {node: list or np.ndarray}
            communities: ground-truth communities {community_name: list(nodes)}
        """
        self.adj = adj
        self.features = features
        self.communities = communities

        # Map each node to its community name
        self.node_to_community = {}
        for name, nodes in communities.items():
            for n in nodes:
                self.node_to_community[n] = name

        # Determine embedding dimension
        self.embed_dim = -1
        if features:
            sample_feat = next(iter(features.values()))
            self.embed_dim = (
                len(sample_feat) if isinstance(sample_feat, (list, np.ndarray)) else 1
            )

        # Cache node embeddings (add tiny noise to zero vectors to avoid numerical issues)
        self.embed_cache = {}
        for node, feat in features.items():
            if isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            elif not isinstance(feat, np.ndarray):
                feat = np.array([feat], dtype=np.float32)
            if np.linalg.norm(feat) < 1e-8:
                feat += np.random.normal(0, 1e-6, size=feat.shape).astype(np.float32)
            self.embed_cache[node] = feat

        # Cache neighbors
        self.neighbor_cache = {}

    @property
    def embedsize(self):
        """Return embedding dimension."""
        return self.embed_dim

    def singleNodeEmbed(self, node):
        """Return embedding vector for a single node."""
        if node not in self.embed_cache:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return self.embed_cache[node]

    def nodesEmbed(self, nodes: list):
        """Return embedding matrix for a list of nodes."""
        embeds = [self.singleNodeEmbed(n) for n in nodes]
        return np.array(embeds, dtype=np.float32)

    def getSingleNodeNeighbor(self, node):
        """Return neighbor list of a single node (deduplicated)."""
        if node in self.neighbor_cache:
            return self.neighbor_cache[node].copy()
        neighbors = self.adj.get(node, set())
        neigh_list = list(neighbors)
        self.neighbor_cache[node] = neigh_list
        return neigh_list.copy()

    def getNodesNeigh(self, nodes: list):
        """Return union of neighbors for a list of nodes."""
        res = set()
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))
        return list(res)

    def getNodeCom(self, node: str):
        """
        Return the full node list of the community that `node` belongs to.
        (Interface compatibility: ignores maxlen parameter from original random-walk version.)
        """
        community_name = self.node_to_community.get(node)
        if community_name:
            return list(self.communities[community_name])
        else:
            return [node]

    @property
    def nodes(self):
        return list(self.adj.keys())

    @property
    def n_nodes(self):
        return len(self.adj)