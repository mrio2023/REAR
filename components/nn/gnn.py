import numpy as np
from scipy import sparse as sp




class GNN:
    def __init__(self, graph, k: int = 2, alpha: float = 0.85):
        self.graph = graph
        self.k = k
        self.alpha = alpha
        self.adjmap = graph.adjmap
        self.node_degrees = self._compute_degrees()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if sp.issparse(x):
            x_dense = x.toarray()
        else:
            x_dense = x
            
        init_val = x_dense.copy()
        
        all_nodes = list(self.graph.df_nodes["address"])
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
        
        for _ in range(self.k):
            new_x = np.zeros_like(x_dense)
            
            for node in all_nodes:
                idx = node_to_idx[node]
                
                if node not in self.adjmap:
                    new_x[idx] = init_val[idx]
                    continue
                    
                edges = self.adjmap[node]
                neigh_features = []
                
                for edge in edges:
                    neighbor = edge["to"]
                    if neighbor in node_to_idx:
                        neighbor_idx = node_to_idx[neighbor]
                        weight = 1.0 / np.sqrt(
                            self.node_degrees.get(node, 1) * 
                            self.node_degrees.get(neighbor, 1)
                        )
                        neigh_features.append(weight * x_dense[neighbor_idx])
                
                if neigh_features:
                    agg_feat = np.mean(neigh_features, axis=0)
                    new_x[idx] = self.alpha * agg_feat + (1 - self.alpha) * init_val[idx]
                else:
                    new_x[idx] = init_val[idx]
            
            x_dense = new_x
        
        return sp.csr_matrix(x_dense)

    def updateGraph(self, graph):
        self.graph = graph
        self.adjmap = graph.adjmap
        self.node_degrees = self._compute_degrees()

    def _compute_degrees(self):
        degrees = {}
        for node, edges in self.adjmap.items():
            degrees[node] = len(edges)
        return degrees


