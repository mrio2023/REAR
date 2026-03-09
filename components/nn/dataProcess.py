import pandas as pd
import random
import os
import sys
import numpy as np

class dataProcess():
    def __init__(self, dfname: str, normal_node_ratio: float = 0.2, expand_hop: int = 2, min_community_size: int = 10):
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

        # 5. 构建训练集（k-hop扩张）
        print(f"🏋️ 构建训练集 (k={self.expand_hop})...")
        self.train_nodes, self.train_feature, self.train_edge = self.build_train(df_nodes, df_feature, df_edge)
        print(f"   训练节点: {len(self.train_nodes)}, 训练边: {len(self.train_edge)}")

        # 6. 构建测试集（k-hop扩张 + 排除训练节点）
        print(f"🧪 构建测试集...")
        self.test_nodes, self.test_feature, self.test_edge = self.build_test(df_nodes, df_feature, df_edge)
        print(f"   测试节点: {len(self.test_nodes)}, 测试边: {len(self.test_edge)}")

        # 7. 筛选训练/测试黑客节点
        print("🔬 筛选黑客节点...")
        self.train_hacker = self.df_hacker[self.df_hacker["address"].isin(self.train_nodes["address"])]
        self.test_hacker  = self.df_hacker[self.df_hacker["address"].isin(self.test_nodes["address"])]
        print(f"   训练黑客: {len(self.train_hacker)}, 测试黑客: {len(self.test_hacker)}")

        # 8. 清理临时数据
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

    def get_k_hop(self, df_edge, seeds, k, exclude_set=None):
        if k < 1 or not seeds:
            return list(seeds)

        # 构建邻接表（双向）
        df_from = df_edge.groupby("from")["to"].apply(set).reset_index()
        df_to   = df_edge.groupby("to")["from"].apply(set).reset_index()
        df_from.columns = ["node", "neighbors"]
        df_to.columns   = ["node", "neighbors"]
        adj = pd.concat([df_from, df_to]).groupby("node")["neighbors"].apply(
            lambda x: set.union(*x) if len(x) else set()
        ).to_dict()

        visited = set(seeds)
        current = set(seeds)
        for _ in range(k):
            nxt = set()
            for u in current:
                nxt.update(adj.get(u, set()))
            nxt -= visited
            if exclude_set:
                nxt -= exclude_set
            visited.update(nxt)
            current = nxt
        return list(visited)

    def build_train(self, df_nodes, df_feature, df_edge):
        normal_num = int(len(self.train_hacker_ids) * self.normal_ratio)
        normal_num = min(normal_num, len(self.normal_nodes_all))
        train_normal = random.sample(list(self.normal_nodes_all), normal_num) if normal_num else []
        seeds = list(self.train_hacker_ids) + train_normal
        node_pool = self.get_k_hop(df_edge, seeds, self.expand_hop)
        node_set = set(node_pool)
        train_nodes = df_nodes[df_nodes["address"].isin(node_set)]
        train_feat  = df_feature[df_feature["address"].isin(node_set)]
        train_edge  = df_edge[df_edge["from"].isin(node_set) & df_edge["to"].isin(node_set)]
        return train_nodes, train_feat, train_edge

    def build_test(self, df_nodes, df_feature, df_edge):
        train_ids = set(self.train_nodes["address"])
        test_hacker_num = len(self.test_hacker_ids)
        test_normal_num = int(test_hacker_num * self.normal_ratio)
        normal_pool = list(self.normal_nodes_all - train_ids)
        test_normal_num = min(test_normal_num, len(normal_pool))
        test_normal = random.sample(normal_pool, test_normal_num) if test_normal_num else []
        seeds = list(self.test_hacker_ids) + test_normal
        node_pool = self.get_k_hop(df_edge, seeds, self.expand_hop, exclude_set=train_ids)
        node_set = set(node_pool)
        test_nodes = df_nodes[df_nodes["address"].isin(node_set)]
        test_feat  = df_feature[df_feature["address"].isin(node_set)]
        test_edge  = df_edge[df_edge["from"].isin(node_set) & df_edge["to"].isin(node_set)]
        return test_nodes, test_feat, test_edge