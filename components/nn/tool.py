from typing import List
import pandas as pd
from graph import Graph


class Tool:
    def __init__(self, graph: Graph, k: int = 2):
        self.graph = graph
        self.k = k  

    def initKego(self):
        seeds = set(self.graph.df_hacker["address"].values.tolist())
        sub_hacker, sub_nodes, sub_features = self.generateKego(seeds, 2)

        ego = Graph(dffeature=sub_features, dfhacker=sub_hacker, dfnode=sub_nodes)
        ego.parentGraph = self.graph
        return ego
    

    def generateKego(self, nodes: List[str], k: int):
        seeds = set(nodes)
        all_ego_nodes = seeds.copy()

        for _ in range(k):
            current_neighbors = set()
            for node in all_ego_nodes:
                if node in self.graph.adjmap:
                    neighbors = [edge["to"] for edge in self.graph.adjmap[node]]
                    current_neighbors.update(neighbors)
            all_ego_nodes.update(current_neighbors)

        sub_hacker = self.graph.df_hacker[
            self.graph.df_hacker["address"].isin(all_ego_nodes)
        ].copy()

        if hasattr(self.graph, "df_nodes") and self.graph.df_nodes is not None:
            sub_nodes = self.graph.df_nodes[
                self.graph.df_nodes["address"].isin(all_ego_nodes)
            ].copy()
        else:
            sub_nodes = pd.DataFrame()

        if hasattr(self.graph, "df_feature") and self.graph.df_feature is not None:
            sub_features = self.graph.df_feature[
                self.graph.df_feature["from"].isin(all_ego_nodes)
                & self.graph.df_feature["to"].isin(all_ego_nodes)
            ].copy()
        else:
            sub_features = pd.DataFrame()

        return sub_hacker, sub_nodes, sub_features
