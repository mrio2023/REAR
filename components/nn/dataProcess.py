import pandas as pd
import random
import os
import sys
import numpy as np
from collections import deque

class dataProcess():
    def __init__(self, dfname: str, normal_node_ratio: float , expand_hop: int , min_community_size: int ):
        self.dfname = dfname
        self.normal_ratio = normal_node_ratio
        self.expand_hop = expand_hop
        self.min_com_size = min_community_size
        
        # 1. 读取原始数据
        print("📥 读取数据...")
        df_edge, df_nodes, df_feature, df_hacker = self.read_data()

        # 2. 过滤小社区
        print("🔍 过滤小社区...")
        self.df_hacker = self.filter_small_com(df_hacker)
        self.all_com_tags = self.df_hacker["name_tag"].unique().tolist()
        print(f"   剩余社区数: {len(self.all_com_tags)}")

        # 3. 划分训练/测试社区
        print("✂️ 划分训练/测试社区...")
        self.train_com_tags, self.test_com_tags = self.split_by_community()
        self.train_hacker_ids = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.train_com_tags)]["address"])
        self.test_hacker_ids  = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.test_com_tags)]["address"])
        print(f"   训练社区: {len(self.train_com_tags)}, 测试社区: {len(self.test_com_tags)}")

        # 4. 筛选普通节点（非黑客）
        print("👤 筛选普通节点...")
        all_hacker_ids = self.train_hacker_ids | self.test_hacker_ids
        mask = ~df_nodes["address"].isin(all_hacker_ids)
        self.normal_nodes_all = set(df_nodes.loc[mask, "address"])
        print(f"   普通节点: {len(self.normal_nodes_all)}")

        # ★ 5. 构建全局邻接表（整数索引版）
        print("⚙️ 构建全局邻接表...")
        self.node2idx, self.idx2node, self.adj_list = self.build_global_adj(df_edge, df_nodes)
        print(f"   节点总数: {len(self.node2idx)}")
        # 将节点集合转为索引集
        self.train_hacker_idx = {self.node2idx[addr] for addr in self.train_hacker_ids if addr in self.node2idx}
        self.test_hacker_idx  = {self.node2idx[addr] for addr in self.test_hacker_ids if addr in self.node2idx}
        self.normal_nodes_all_idx = {self.node2idx[addr] for addr in self.normal_nodes_all if addr in self.node2idx}

        # 6. 构建训练集（k-hop扩张）
        print(f"🏋️ 构建训练集 (k={self.expand_hop})...")
        self.train_nodes, self.train_feature, self.train_edge = self.build_train(df_nodes, df_feature, df_edge)
        print(f"   训练节点: {len(self.train_nodes)}, 训练边: {len(self.train_edge)}")

        # 7. 构建测试集（k-hop扩张 + 排除训练节点）
        print(f"🧪 构建测试集...")
        self.test_nodes, self.test_feature, self.test_edge = self.build_test(df_nodes, df_feature, df_edge)
        print(f"   测试节点: {len(self.test_nodes)}, 测试边: {len(self.test_edge)}")

        # 8. 筛选训练/测试黑客节点
        print("🔬 筛选黑客节点...")
        self.train_hacker = self.df_hacker[self.df_hacker["address"].isin(self.train_nodes["address"])]
        self.test_hacker  = self.df_hacker[self.df_hacker["address"].isin(self.test_nodes["address"])]
        print(f"   训练黑客: {len(self.train_hacker)}, 测试黑客: {len(self.test_hacker)}")

        # 9. 清理临时数据
        del df_edge, df_nodes, df_feature, df_hacker
        print("✅ 数据预处理完成！")

    def read_data(self):
        base_path = os.path.join("codes", "df", self.dfname)
        def read_file(path):
            if os.path.exists(f"{path}.parquet"):
                return pd.read_parquet(f"{path}.parquet")
            elif os.path.exists(f"{path}.csv"):
                return pd.read_csv(f"{path}.csv", low_memory=False)
            else:
                raise FileNotFoundError(f"文件不存在：{path}")
        df_edge   = read_file(os.path.join(base_path, f"{self.dfname}_edgelist"))
        df_nodes  = read_file(os.path.join(base_path, f"{self.dfname}_node_classes"))
        df_feature= read_file(os.path.join(base_path, f"{self.dfname}_features"))
        df_hacker = read_file(os.path.join(base_path, f"{self.dfname}_hacker"))
        return df_edge, df_nodes, df_feature, df_hacker

    def filter_small_com(self, df_hacker):
        df = df_hacker.dropna(subset=["name_tag"])
        com_size = df["name_tag"].value_counts()
        large_tags = com_size[com_size >= self.min_com_size].index
        return df[df["name_tag"].isin(large_tags)]

    def split_by_community(self):
        tags = self.all_com_tags.copy()
        random.shuffle(tags)
        split = max(1, int(0.75 * len(tags)))
        return tags[:split], tags[split:]

    def build_global_adj(self, df_edge, df_nodes):
        """
        构建整数索引的双向邻接表
        返回: node2idx dict, idx2node list, adj_list (list of list)
        """
        # 收集所有节点：nodes表中的所有地址 + edge中出现的所有节点
        all_nodes = set(df_nodes["address"])
        all_nodes.update(df_edge["from"])
        all_nodes.update(df_edge["to"])
        # 排序以便调试
        node_list = sorted(all_nodes)
        node2idx = {node: i for i, node in enumerate(node_list)}
        idx2node = node_list

        # 初始化邻接表
        adj = [[] for _ in range(len(node_list))]

        # 填充双向边
        # 使用itertuples可能更快，但为通用性保留iterrows
        for _, row in df_edge.iterrows():
            u = node2idx[row["from"]]
            v = node2idx[row["to"]]
            adj[u].append(v)
            adj[v].append(u)

        # 可选：去重（如果数据可能有重复边）
        # for i in range(len(adj)):
        #     if adj[i]:
        #         adj[i] = list(set(adj[i]))

        return node2idx, idx2node, adj

    def get_k_hop_idx(self, seeds_idx, k, exclude_idx=None):
        """
        使用整数索引和队列进行BFS，返回节点索引列表
        seeds_idx: 种子节点索引的可迭代对象（如list/set）
        k: 扩展层数
        exclude_idx: 需要排除的节点索引集合（可选）
        """
        if k <= 0 or not seeds_idx:
            return list(seeds_idx)

        n = len(self.adj_list)
        visited = np.zeros(n, dtype=bool)
        for idx in seeds_idx:
            visited[idx] = True
        if exclude_idx:
            for idx in exclude_idx:
                visited[idx] = True

        current = deque(seeds_idx)
        for _ in range(k):
            if not current:
                break
            next_layer = deque()
            for u in current:
                for v in self.adj_list[u]:
                    if not visited[v]:
                        visited[v] = True
                        next_layer.append(v)
            current = next_layer

        # 返回所有访问过的节点（但排除掉exclude_idx中的节点）
        result = np.where(visited)[0].tolist()
        if exclude_idx:
            # 过滤掉exclude_idx中的节点（虽然它们已被标记，但可能被加入visited？实际上exclude_idx一开始就标记为True，所以不会加入current，但最后在result中它们存在，需要移除）
            result = [idx for idx in result if idx not in exclude_idx]
        return result

    def build_train(self, df_nodes, df_feature, df_edge):
        # 采样普通节点（索引版）
        normal_num = int(len(self.train_hacker_ids) * self.normal_ratio)
        normal_num = min(normal_num, len(self.normal_nodes_all_idx))
        train_normal_idx = random.sample(list(self.normal_nodes_all_idx), normal_num) if normal_num else []

        seeds_idx = list(self.train_hacker_idx) + train_normal_idx
        node_pool_idx = self.get_k_hop_idx(seeds_idx, self.expand_hop)

        # 转回地址
        node_pool_addr = [self.idx2node[i] for i in node_pool_idx]
        node_set = set(node_pool_addr)

        # 使用索引加速DataFrame筛选
        df_nodes_idx = df_nodes.set_index("address")
        train_nodes = df_nodes_idx.loc[list(node_set)].reset_index()

        df_feature_idx = df_feature.set_index("address")
        train_feat = df_feature_idx.loc[list(node_set)].reset_index()

        # 边筛选
        mask_from = df_edge["from"].isin(node_set)
        mask_to = df_edge["to"].isin(node_set)
        train_edge = df_edge[mask_from & mask_to]

        return train_nodes, train_feat, train_edge

    def build_test(self, df_nodes, df_feature, df_edge):
        # 训练集节点地址集合
        train_addr_set = set(self.train_nodes["address"])
        # 转成索引集
        train_idx_set = {self.node2idx[addr] for addr in train_addr_set if addr in self.node2idx}

        # 普通节点池（排除训练集节点）
        normal_pool_idx = self.normal_nodes_all_idx - train_idx_set
        test_normal_num = int(len(self.test_hacker_ids) * self.normal_ratio)
        test_normal_num = min(test_normal_num, len(normal_pool_idx))
        test_normal_idx = random.sample(list(normal_pool_idx), test_normal_num) if test_normal_num else []

        seeds_idx = list(self.test_hacker_idx) + test_normal_idx
        node_pool_idx = self.get_k_hop_idx(seeds_idx, self.expand_hop, exclude_idx=train_idx_set)

        node_pool_addr = [self.idx2node[i] for i in node_pool_idx]
        node_set = set(node_pool_addr)

        # 筛选DataFrame
        df_nodes_idx = df_nodes.set_index("address")
        test_nodes = df_nodes_idx.loc[list(node_set)].reset_index()

        df_feature_idx = df_feature.set_index("address")
        test_feat = df_feature_idx.loc[list(node_set)].reset_index()

        mask_from = df_edge["from"].isin(node_set)
        mask_to = df_edge["to"].isin(node_set)
        test_edge = df_edge[mask_from & mask_to]

        return test_nodes, test_feat, test_edge