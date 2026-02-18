import hashlib
import random
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


class Graph:
    def __init__(self, dfnode, dffeature, dfhacker):
        self.df_nodes = dfnode
        self.df_feature = dffeature
        self.df_hacker = dfhacker
        
        self.adjmap = self.calAdjMap(self.df_feature)  # from → to（出边）
        self.adjtomap = self.calAdjToMap(self.df_feature)  # to → from（入边）
        self.snapshot = self.calSnapshot(self.df_feature)
        self.deMap = self.calDegreeMap()
        
        self.allNameTags = sorted(self.df_hacker["name_tag"].dropna().unique())
        self.tag2idx = {tag: i for i, tag in enumerate(self.allNameTags)}
        self.n_nodes = len(self.df_nodes)
        self.t_scaler = self.initScaler()
        self.parentGraph: Graph = None
        self.MaxTrajectoryLen = 16  # 最大路径长度
        self.community_seeds = self._cache_community_seeds()
    
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
        community_seeds = {}
        for tag in self.allNameTags:
            seeds = set(
                self.df_hacker[self.df_hacker["name_tag"] == tag]["address"].tolist()
            )
            community_seeds[tag] = seeds
        return community_seeds

    def sampleTrajectory(self, node: str, min_community_ratio: float = 0.6, sample_ratio: float = 0.6):
        """
        在一阶子图上采样轨迹
        
        Args:
            node: 起始节点
            min_community_ratio: 最小社区纯度阈值，低于此值可能停止
            sample_ratio: 无同社区节点时继续采样的概率
        
        Returns:
            采样轨迹
        """
        tra = [node]
        
        # 获取节点的name_tag
        node_hacker_row = self.df_hacker[self.df_hacker["address"] == node]
        if node_hacker_row.empty or pd.isna(node_hacker_row["name_tag"].iloc[0]):
            return tra
        
        name_tag = node_hacker_row["name_tag"].iloc[0]
        sameCom = 1
        
        # 获取一阶子图节点集合
        one_hop_subgraph = self.get_1hop_subgraph(node)
        
        for step in range(self.MaxTrajectoryLen - 1):
            # 获取当前路径所有节点的邻居
            all_neighbors = set()
            for n in tra:
                all_neighbors.update(self.getSingleNodeNeighbor(n))
            
            # 候选节点 = (所有邻居 - 已访问节点) ∩ 一阶子图
            candidate = list(all_neighbors - set(tra))
            
            if len(candidate) <= 0:
                break
            
            # 检查社区纯度
            current_ratio = sameCom / len(tra)
            
            # 随机决定是否停止（基于纯度和随机性）
            stop_prob = max(0, 1 - current_ratio)  # 纯度越低，停止概率越高
            if random.random() < stop_prob and len(tra) > 1:
                break
            
            # 筛选候选中的同社区节点
            community_nodes = self.community_seeds.get(name_tag, set())
            same_community_candidates = [n for n in candidate if n in community_nodes]
            
            # 选择节点
            if same_community_candidates:
                # 优先选择同社区节点（80%概率选同社区，20%概率探索）
                if random.random() < 0.8 or len(same_community_candidates) == len(candidate):
                    selected_node = random.choice(same_community_candidates)
                    tra.append(selected_node)
                    sameCom += 1
                else:
                    selected_node = random.choice(candidate)
                    tra.append(selected_node)
            else:
                # 无同社区节点，按概率决定是否继续
                if random.random() < sample_ratio:
                    selected_node = random.choice(candidate)
                    tra.append(selected_node)
                else:
                    break
        
        return tra

    def initScaler(self):
        t_scaler = MinMaxScaler(feature_range=(0, 10))
        all_trade = self.df_feature["Amount"].values.reshape(-1, 1)
        t_scaler.fit(all_trade)
        return t_scaler

    def addressToBinaryEmbedding(self, address: str, embed_dim=64):
        if address.startswith("0x"):
            hex_str = address[2:].lower()
        else:
            hex_str = address.lower()
        if len(hex_str) != 40:
            raise ValueError(f"无效的以太坊地址长度: {address}")
        addr_bytes = bytes.fromhex(hex_str)
        bits = []
        for byte in addr_bytes:
            bits.extend([(byte >> i) & 1 for i in range(8)])
        if embed_dim > len(bits):
            bits = bits * (embed_dim // len(bits) + 1)
        return np.array(bits[:embed_dim], dtype=np.float32)

    def singleNodeEmbed(self, node):
        tag_dim = len(self.allNameTags)
        if node not in self.deMap:
            return None
        node_hacker_row = self.df_hacker[self.df_hacker["address"] == node]
        tag_embed = np.zeros(tag_dim, dtype=np.float32)
        if not node_hacker_row.empty:
            node_tag = node_hacker_row["name_tag"].iloc[0]
            if not pd.isna(node_tag) and node_tag in self.tag2idx:
                tag_embed[self.tag2idx[node_tag]] = 1.0
        self_degree = self.deMap[node]
        neighdata = self.adjmap.get(node, [])
        trades = []
        neighbor_degrees = []
        for data in neighdata:
            neighbor_node = data["to"]
            if neighbor_node in self.deMap:
                neighbor_degrees.append(self.deMap[neighbor_node])
            trades.append(data["amt"])
        t_max = t_min = t_mean = t_std = 0.0
        if trades:
            t_max_original = np.max(trades)
            t_min_original = np.min(trades)
            t_mean_original = np.mean(trades)
            t_std = np.std(trades)
            t_max = self.t_scaler.transform([[t_max_original]])[0][0]
            t_min = self.t_scaler.transform([[t_min_original]])[0][0]
            t_mean = self.t_scaler.transform([[t_mean_original]])[0][0]
        n_deg_max = n_deg_min = n_deg_mean = n_deg_std = 0.0
        if neighbor_degrees:
            n_deg_max = np.max(neighbor_degrees)
            n_deg_min = np.min(neighbor_degrees)
            n_deg_mean = np.mean(neighbor_degrees)
            n_deg_std = np.std(neighbor_degrees)
        hash_embed = self.addressToBinaryEmbedding(node, embed_dim=64)

        embed_vector = np.concatenate(
            [
                hash_embed,
                tag_embed,
                np.array([self_degree], dtype=np.float32),
                np.array(
                    [n_deg_max, n_deg_min, n_deg_mean, n_deg_std], dtype=np.float32
                ),
                np.array([t_max, t_min, t_mean, t_std], dtype=np.float32),
            ]
        )
        return embed_vector

    def nodesEmbed(self, nodes: list):
        """批量获取节点嵌入"""
        embeds = [self.singleNodeEmbed(n) for n in nodes]

        target_dim = None
        for e in embeds:
            if e is not None:
                target_dim = len(e)
                break

        if target_dim is None:
            target_dim = 64

        result = []
        none_count = 0
        for i, e in enumerate(embeds):
            if e is None:
                result.append(np.zeros(target_dim, dtype=np.float32))
                none_count += 1
            elif len(e) == target_dim:
                result.append(e)
            elif len(e) < target_dim:
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[: len(e)] = e
                result.append(padded)
            else:
                result.append(e[:target_dim])

        return result

    def calDegreeMap(self):
        deMap = dict()
        # 统计出边度数
        for n, datas in self.adjmap.items():
            if n not in deMap:
                deMap[n] = 0
            deMap[n] += len(datas)
            for edge_data in datas:
                m = edge_data["to"]
                if m not in deMap:
                    deMap[m] = 0
                deMap[m] += 1
        # 补充入边度数
        for n, datas in self.adjtomap.items():
            if n not in deMap:
                deMap[n] = 0
            deMap[n] += len(datas)
            for edge_data in datas:
                m = edge_data["from"]
                if m not in deMap:
                    deMap[m] = 0
                deMap[m] += 1
        return deMap

    def calAdjMap(self, df_features: pd.DataFrame):
        adj_map = dict()
        for _, row in df_features.iterrows():
            u = row["from"]
            v = row["to"]
            amt = row["amt"] if "amt" in row else row["Amount"]
            tsp = row["timeStamp"]
            edge_data = {"to": v, "amt": amt, "tsp": tsp}
            if u in adj_map:
                adj_map[u].append(edge_data)
            else:
                adj_map[u] = [edge_data]
        return adj_map

    def calAdjToMap(self, df_features: pd.DataFrame):
        adj_map = dict()
        for _, row in df_features.iterrows():
            u = row["to"]
            v = row["from"]
            amt = row["amt"] if "amt" in row else row["Amount"]
            tsp = row["timeStamp"]
            edge_data = {"from": v, "amt": amt, "tsp": tsp}
            if u in adj_map:
                adj_map[u].append(edge_data)
            else:
                adj_map[u] = [edge_data]
        return adj_map

    def calSnapshot(self, df_features: pd.DataFrame):
        snap_shots = []
        df_time_grouped = df_features.sort_values("timeStamp").groupby("timeStamp")
        for timestamp, df_t in df_time_grouped:
            snap_shot = self.calAdjMap(df_t)
            snap_shots.append({"timestamp": timestamp, "adjmap": snap_shot})
        return snap_shots