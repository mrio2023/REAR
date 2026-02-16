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

    # ========== 修复bug + 优化邻居获取逻辑 ==========
    def getNodesNeigh(self, nodes: list):
        """获取多个节点的所有邻居（出边+入边，去重）"""
        res = set()  # 用集合去重
        for n in nodes:
            res.update(self.getSingleNodeNeighbor(n))  # 修复：set不能用+=，用update
        return list(res)

    def getSingleNodeNeighbor(self, node: str):
        """获取单个节点的所有邻居（出边+入边）"""
        fromdata = self.adjmap.get(node, [])  # 修复：用get避免KeyError
        todata = self.adjtomap.get(node, [])   # 修复：用get避免KeyError
        neigh = [d["to"] for d in fromdata] + [d["from"] for d in todata]
        return list(set(neigh))  # 去重，避免重复邻居

    def _cache_community_seeds(self):
        community_seeds = {}
        for tag in self.allNameTags:
            seeds = set(
                self.df_hacker[self.df_hacker["name_tag"] == tag]["address"].tolist()
            )
            community_seeds[tag] = seeds
        return community_seeds

    def sampleTrajectory(self, node: str, min_community_ratio: float = 0.6):
        """
        核心改动：
        1. 候选节点 = 所有社区节点的邻居（出边+入边）
        2. 保留“先扩16步→找最优子路径”逻辑
        3. 过滤已访问节点，避免重复
        """
        # 1. 基础校验：获取节点社区及种子集合
        node_com_row = self.df_hacker[self.df_hacker["address"] == node]
        if node_com_row.empty or pd.isna(node_com_row["name_tag"].iloc[0]):
            return [node, "stp"]
        
        node_com = node_com_row["name_tag"].iloc[0]
        seed_nodes = self.community_seeds.get(node_com, set())
        if not seed_nodes:
            return [node, "stp"]

        # ========== 核心改动1：获取所有社区节点的邻居（候选池） ==========
        # 候选节点 = 社区所有节点的出边+入边邻居
        com_nodes_list = list(seed_nodes)
        com_all_neighbors = self.getNodesNeigh(com_nodes_list)
        # 合并社区种子节点，扩大候选池（包含社区内节点+邻居）
        all_candidates = list(seed_nodes) + com_all_neighbors
        # 去重 + 过滤空值
        all_candidates = list(set([n for n in all_candidates if n is not None and n != ""]))
        if not all_candidates:  # 无候选节点，直接终止
            return [node, "stp"]

        # 2. 扩展到最大路径长度（16步，从全候选池采样）
        full_trajectory = [node]
        cur_node = node
        visited = {node}  # 过滤已访问节点，避免重复
        has_break_early = False
        
        for _ in range(self.MaxTrajectoryLen - 1):
            # ========== 核心改动2：从全候选池选节点（过滤已访问） ==========
            available_candidates = [n for n in all_candidates if n not in visited]
            if not available_candidates:  # 无未访问候选，终止
                has_break_early = True
                break
            
            # 拆分候选：同社区种子节点 → 其他邻居节点
            same_com = [n for n in available_candidates if n in seed_nodes]
            other = [n for n in available_candidates if n not in seed_nodes]

            # 节点选择逻辑：优先选同社区 → 其次其他候选
            if same_com:
                next_node = random.choice(same_com)
            elif other:
                next_node = random.choice(other)
            else:
                has_break_early = True
                break
            
            full_trajectory.append(next_node)
            visited.add(next_node)  # 标记已访问
            cur_node = next_node

        # 3. 遍历所有子路径，计算社区比率，找到最优子路径（逻辑不变）
        valid_full = [n for n in full_trajectory if n != "stp"]
        if len(valid_full) <= 1:
            return [node, "stp"]
        
        best_ratio = 0.0
        best_subtraj = [node]
        
        # 遍历所有可能的子路径长度（从2到完整轨迹长度）
        for end_idx in range(2, len(valid_full) + 1):
            subtraj = valid_full[:end_idx]
            com_count = sum(1 for n in subtraj if n in seed_nodes)
            ratio = com_count / len(subtraj)
            if ratio > best_ratio or (ratio == best_ratio and len(subtraj) > len(best_subtraj)):
                best_ratio = ratio
                best_subtraj = subtraj.copy()

        # 4. 判断最优比率是否达标，返回结果
        if best_ratio >= min_community_ratio:
            if len(best_subtraj) == self.MaxTrajectoryLen:
                best_subtraj.append("stp")
            return best_subtraj
        else:
            return [node, "stp"]

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
            print(f"使用默认维度: {target_dim}")

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

        if none_count > 0:
            print(f"修复了 {none_count} 个None值")

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
        # 补充入边度数（可选，根据需求）
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