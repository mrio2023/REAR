import pandas as pd
import random
import os

class dataProcess():
    def __init__(self, dfname: str, normal_node_ratio: float = 0.2, expand_hop: int = 2, min_community_size:int=10):
        self.dfname = dfname
        self.normal_ratio = normal_node_ratio
        self.expand_hop = expand_hop
        self.min_com_size = min_community_size
        
        df_edge, df_nodes, df_feature, df_hacker = self.read_data()
      
        self.df_hacker = self.filter_small_com(df_hacker)
        self.all_com_tags = self.df_hacker["name_tag"].unique().tolist()
        
        self.train_com_tags, self.test_com_tags = self.split_by_community()
        self.train_hacker_ids = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.train_com_tags)]["address"].unique())
        self.test_hacker_ids  = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.test_com_tags)]["address"].unique())
        
        # 普通节点：完全和训练/测试黑客分开
        all_hacker_ids = self.train_hacker_ids | self.test_hacker_ids
        self.normal_nodes_all = set(df_nodes[~df_nodes["address"].isin(all_hacker_ids)]["address"].unique())

        self.hacker2tag = dict(zip(self.df_hacker["address"], self.df_hacker["name_tag"]))
        self.tag2nodes = self.df_hacker.groupby("name_tag")["address"].apply(set).to_dict()
        
        self.train_nodes, self.train_feature, self.train_edge = self.build_train(df_nodes, df_feature, df_edge)
        self.test_nodes,  self.test_feature,  self.test_edge  = self.build_test(df_nodes, df_feature, df_edge)
        
        self.train_hacker = self.df_hacker[self.df_hacker["address"].isin(self.train_nodes["address"])].copy()
        self.test_hacker  = self.df_hacker[self.df_hacker["address"].isin(self.test_nodes["address"])].copy()

        del df_edge, df_nodes, df_feature, df_hacker

    def read_data(self):
        base_path = os.path.join("codes", "df", self.dfname)
        paths = {
            "edge": os.path.join(base_path, f"{self.dfname}_edgelist.csv"),
            "node": os.path.join(base_path, f"{self.dfname}_node_classes.csv"),
            "feat": os.path.join(base_path, f"{self.dfname}_features.csv"),
            "hacker": os.path.join(base_path, f"{self.dfname}_hacker.csv")
        }
        df_edge = pd.read_csv(paths["edge"])
        df_nodes = pd.read_csv(paths["node"])
        df_feature = pd.read_csv(paths["feat"])
        df_hacker = pd.read_csv(paths["hacker"])
        return df_edge, df_nodes, df_feature, df_hacker

    def filter_small_com(self, df_hacker):
        df = df_hacker[df_hacker["name_tag"].notna()].copy()
        com_size = df["name_tag"].value_counts()
        large_tags = com_size[com_size >= self.min_com_size].index
        return df[df["name_tag"].isin(large_tags)].copy()

    def split_by_community(self):
        random.shuffle(self.all_com_tags)
        train_com_num = max(1, int(0.75 * len(self.all_com_tags)))
        train_com_tags = self.all_com_tags[:train_com_num]
        test_com_tags = self.all_com_tags[train_com_num:]
        return train_com_tags, test_com_tags

    def get_k_hop(self, df_edge, seeds, k):
        if k < 1 or not seeds:
            return list(seeds)
        adj = {}
        for _, row in df_edge.iterrows():
            u, v = row["from"], row["to"]
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        visited = set(seeds)
        current = set(seeds)
        for _ in range(k):
            nxt = set()
            for u in current:
                nxt.update(adj.get(u, set()))
            nxt -= visited
            visited.update(nxt)
            current = nxt
        return list(visited)

    def build_train(self, df_nodes, df_feature, df_edge):
        normal_num = int(len(self.train_hacker_ids) * self.normal_ratio)
        normal_num = min(normal_num, len(self.normal_nodes_all))
        train_normal = random.sample(list(self.normal_nodes_all), normal_num) if normal_num > 0 else []
        seeds = list(self.train_hacker_ids) + train_normal
        
        train_all = self.get_k_hop(df_edge, seeds, self.expand_hop)
        
        train_nodes = df_nodes[df_nodes["address"].isin(train_all)].copy()
        train_feat  = df_feature[df_feature["address"].isin(train_all)].copy()
        train_edge  = df_edge[(df_edge["from"].isin(train_all)) & (df_edge["to"].isin(train_all))].copy()
        return train_nodes, train_feat, train_edge

    # ===================== 关键：测试集 不做 k-hop =====================
    def build_test(self, df_nodes, df_feature, df_edge):
        """
        测试集规则（干净、不扩散、不k-hop）
        1. 只包含：测试黑客 + 普通节点
        2. 绝对不含训练集任何节点
        3. 边只保留：两端都在测试集
        """
        train_ids = set(self.train_nodes["address"].unique())

        # 测试节点 = 测试黑客 + 普通节点
        test_seed_nodes = list(self.test_hacker_ids) + list(self.normal_nodes_all)
        test_seed_nodes = list(set(test_seed_nodes))

        # 彻底剔除训练集节点（核心）
        test_nodes_pool = [n for n in test_seed_nodes if n not in train_ids]

        # 只取在 nodes/feature 里存在的
        test_nodes = df_nodes[df_nodes["address"].isin(test_nodes_pool)].copy()
        test_feat  = df_feature[df_feature["address"].isin(test_nodes_pool)].copy()
        test_edge  = df_edge[
            (df_edge["from"].isin(test_nodes_pool)) &
            (df_edge["to"].isin(test_nodes_pool))
        ].copy()

        return test_nodes, test_feat, test_edge

# # 测试
# def test():
#     dp = dataProcess("elliptic", normal_node_ratio=0.2, expand_hop=2, min_community_size=10)
#     print("train nodes:", len(dp.train_nodes))
#     print("test nodes:", len(dp.test_nodes))
#     print("intersection:", len(set(dp.train_nodes["address"]) & set(dp.test_nodes["address"])))

# if __name__ == "__main__":
#     test()