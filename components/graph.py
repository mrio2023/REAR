import hashlib
import random
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp


class Graph:
    def __init__(self, dfnode, dffeature, dfhacker):
        self.df_nodes = dfnode
        self.df_feature = dffeature
        self.df_hacker = dfhacker
        self.adjmap = self.calAdjMap(self.df_feature)
        self.snapshot = self.calSnapshot(self.df_feature)
        self.deMap = self.calDegreeMap()
        self.allNameTags = sorted(self.df_hacker["name_tag"].dropna().unique())
        self.tag2idx = {tag: i for i, tag in enumerate(self.allNameTags)}
        self.n_nodes = len(self.df_nodes)
        self.t_scaler = self.initScaler()
        self.parentGraph: Graph = None
        self.MaxTrajectoryLen = 16
        self.community_seeds = self._cache_community_seeds()
    def getBoundry(self, nodes):
        node_set = set(nodes)
        boundary_nodes = set()
        for n in node_set:
            out_edges = self.adjmap.get(n, [])
            for edge in out_edges:
                neighbor = edge["to"]
                if neighbor not in node_set:
                    boundary_nodes.add(neighbor)
        
        for source_node, edges in self.adjmap.items():
            if source_node in node_set:
                continue  
            for edge in edges:
                target_node = edge["to"]
                if target_node in node_set:
                    boundary_nodes.add(source_node)
        
        return list (boundary_nodes)

    def _cache_community_seeds(self):
        community_seeds = {}
        for tag in self.allNameTags:
            seeds = set(
                self.df_hacker[self.df_hacker["name_tag"] == tag]["address"].tolist()
            )
            community_seeds[tag] = seeds
        return community_seeds

    def sampleTrajectory(self, node: str, k: int = 1, min_community_ratio: float = 0.6):
        node_com_row = self.df_hacker[self.df_hacker["address"] == node]
        if node_com_row.empty or pd.isna(node_com_row["name_tag"].iloc[0]):
            return [node, "stp"]
        node_com = node_com_row["name_tag"].iloc[0]
        seed_nodes = self.community_seeds.get(node_com, set())
        if not seed_nodes:
            return [node, "stp"]
        ego_adjmap, _ = self.generateKego(list(seed_nodes), k=k)
        max_attempts = 5
        for _ in range(max_attempts):
            trajectory = [node]
            cur_node = node
            for _ in range(self.MaxTrajectoryLen - 1):
                if cur_node not in ego_adjmap or len(ego_adjmap[cur_node]) == 0:
                    trajectory.append("stp")
                    break
                neighbors = ego_adjmap[cur_node]
                same_com_neighbors = [
                    n["to"] for n in neighbors if n["to"] in seed_nodes
                ]
                ego_only_neighbors = [
                    n["to"] for n in neighbors if n["to"] not in seed_nodes
                ]
                if same_com_neighbors:
                    next_node = random.choice(same_com_neighbors)
                elif ego_only_neighbors:
                    next_node = random.choice(ego_only_neighbors)
                else:
                    trajectory.append("stp")
                    break
                trajectory.append(next_node)
                cur_node = next_node
            valid_nodes = [n for n in trajectory if n != "stp"]
            if len(valid_nodes) <= 1:
                continue
            community_node_count = sum(1 for n in valid_nodes if n in seed_nodes)
            community_ratio = community_node_count / len(valid_nodes)
            if community_ratio >= min_community_ratio:
                if len(trajectory) == self.MaxTrajectoryLen and trajectory[-1] != "stp":
                    trajectory.append("stp")
                return trajectory
        trajectory = [node]
        cur_node = node
        for _ in range(self.MaxTrajectoryLen - 1):
            if cur_node not in ego_adjmap or len(ego_adjmap[cur_node]) == 0:
                trajectory.append("stp")
                break
            neighbors = ego_adjmap[cur_node]
            same_com_neighbors = [n["to"] for n in neighbors if n["to"] in seed_nodes]
            next_node = (
                random.choice(same_com_neighbors)
                if same_com_neighbors
                else random.choice([n["to"] for n in neighbors])
            )
            trajectory.append(next_node)
            cur_node = next_node
        if len(trajectory) == self.MaxTrajectoryLen and trajectory[-1] != "stp":
            trajectory.append("stp")
        return trajectory

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
        # 获取所有嵌入
        embeds = [self.singleNodeEmbed(n) for n in nodes]

        # 找出第一个有效嵌入的维度
        target_dim = None
        for e in embeds:
            if e is not None:
                target_dim = len(e)
                break

        if target_dim is None:
            target_dim = 64  # 默认维度
            print(f"使用默认维度: {target_dim}")

        # 统一处理
        result = []
        none_count = 0
        for i, e in enumerate(embeds):
            if e is None:
                result.append(np.zeros(target_dim, dtype=np.float32))
                none_count += 1
            elif len(e) == target_dim:
                result.append(e)
            elif len(e) < target_dim:
                # 补零
                padded = np.zeros(target_dim, dtype=np.float32)
                padded[: len(e)] = e
                result.append(padded)
            else:
                # 截断
                result.append(e[:target_dim])

        if none_count > 0:
            print(f"修复了 {none_count} 个None值")

        return result

    def calDegreeMap(self):
        deMap = dict()
        for n, datas in self.adjmap.items():
            if n not in deMap:
                deMap[n] = 0
            deMap[n] += len(datas)
            for edge_data in datas:
                m = edge_data["to"]
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

    def calSnapshot(self, df_features: pd.DataFrame):
        snap_shots = []
        df_time_grouped = df_features.sort_values("timeStamp").groupby("timeStamp")
        for timestamp, df_t in df_time_grouped:
            snap_shot = self.calAdjMap(df_t)
            snap_shots.append({"timestamp": timestamp, "adjmap": snap_shot})
        return snap_shots
