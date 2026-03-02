import pandas as pd
import random
import os
import sys
import time

class dataProcess():
    def __init__(self, dfname: str, normal_node_ratio: float = 0.2, expand_hop: int = 2, min_community_size:int=10):
        self.dfname = dfname
        self.normal_ratio = normal_node_ratio
        self.expand_hop = expand_hop
        self.min_com_size = min_community_size
        
        # 初始化进度记录
        self.step = 0
        self.total_steps = 8  # 总步骤数（用于进度计算）
        self.print_progress("开始数据预处理", 0)
        
        # 1. 读取原始数据
        self.step += 1
        self.print_progress("读取原始数据", self.step)
        df_edge, df_nodes, df_feature, df_hacker = self.read_data()
      
        # 2. 过滤小社区
        self.step += 1
        self.print_progress("过滤小社区（剔除<{}的社区）".format(self.min_com_size), self.step)
        self.df_hacker = self.filter_small_com(df_hacker)
        self.all_com_tags = self.df_hacker["name_tag"].unique().tolist()
        self.print_progress("过滤完成：剩余社区数={}".format(len(self.all_com_tags)), self.step, end="\n")
        
        # 3. 划分训练/测试社区
        self.step += 1
        self.print_progress("划分训练/测试社区（75%/25%）", self.step)
        self.train_com_tags, self.test_com_tags = self.split_by_community()
        self.train_hacker_ids = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.train_com_tags)]["address"].unique())
        self.test_hacker_ids  = set(self.df_hacker[self.df_hacker["name_tag"].isin(self.test_com_tags)]["address"].unique())
        self.print_progress("划分完成：训练社区={}，测试社区={}".format(len(self.train_com_tags), len(self.test_com_tags)), self.step, end="\n")
        
        # 4. 筛选普通节点（非黑客）
        self.step += 1
        self.print_progress("筛选普通节点（非黑客）", self.step)
        all_hacker_ids = self.train_hacker_ids | self.test_hacker_ids
        self.normal_nodes_all = set(df_nodes[~df_nodes["address"].isin(all_hacker_ids)]["address"].unique())
        self.hacker2tag = dict(zip(self.df_hacker["address"], self.df_hacker["name_tag"]))
        self.tag2nodes = self.df_hacker.groupby("name_tag")["address"].apply(set).to_dict()
        self.print_progress("筛选完成：普通节点数={}，黑客节点数={}".format(len(self.normal_nodes_all), len(all_hacker_ids)), self.step, end="\n")
        
        # 5. 构建训练集（最耗时：k-hop扩张）
        self.step += 1
        self.print_progress("构建训练集（k-hop扩张，k={}）".format(self.expand_hop), self.step)
        self.train_nodes, self.train_feature, self.train_edge = self.build_train(df_nodes, df_feature, df_edge)
        self.print_progress("训练集构建完成：节点数={}，边数={}".format(len(self.train_nodes), len(self.train_edge)), self.step, end="\n")
        
        # 6. 构建测试集
        self.step += 1
        self.print_progress("构建测试集（无k-hop，纯原始数据）", self.step)
        self.test_nodes,  self.test_feature,  self.test_edge  = self.build_test(df_nodes, df_feature, df_edge)
        self.print_progress("测试集构建完成：节点数={}，边数={}".format(len(self.test_nodes), len(self.test_edge)), self.step, end="\n")
        
        # 7. 筛选训练/测试黑客节点
        self.step += 1
        self.print_progress("筛选训练/测试黑客节点", self.step)
        self.train_hacker = self.df_hacker[self.df_hacker["address"].isin(self.train_nodes["address"])].copy()
        self.test_hacker  = self.df_hacker[self.df_hacker["address"].isin(self.test_nodes["address"])].copy()
        self.print_progress("筛选完成：训练黑客={}，测试黑客={}".format(len(self.train_hacker), len(self.test_hacker)), self.step, end="\n")
        
        # 8. 清理临时数据
        self.step += 1
        self.print_progress("清理临时数据，释放内存", self.step)
        del df_edge, df_nodes, df_feature, df_hacker
        self.print_progress("数据预处理完成✅", self.step, end="\n\n")

    def print_progress(self, msg: str, step: int, total_steps: int = None, end: str = "\r"):
        """
        打印带进度条的提示信息
        :param msg: 提示文本
        :param step: 当前步骤
        :param total_steps: 总步骤数（默认用self.total_steps）
        :param end: 换行符（\r=覆盖当前行，\n=换行）
        """
        if total_steps is None:
            total_steps = self.total_steps
        # 进度条长度
        bar_length = 30
        progress = step / total_steps
        filled_length = int(bar_length * progress)
        # 构建进度条
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        # 格式化输出
        sys.stdout.write(f"\r📊 [{bar}] {step}/{total_steps} - {msg}")
        sys.stdout.flush()
        if end == "\n":
            sys.stdout.write("\n")

    def read_data(self):
        base_path = os.path.join("codes", "df", self.dfname)
        paths = {
            "edge": os.path.join(base_path, f"{self.dfname}_edgelist.csv"),
            "node": os.path.join(base_path, f"{self.dfname}_node_classes.csv"),
            "feat": os.path.join(base_path, f"{self.dfname}_features.csv"),
            "hacker": os.path.join(base_path, f"{self.dfname}_hacker.csv")
        }
        # 逐个读取并显示进度
        df_edge = pd.read_csv(paths["edge"])
        self.print_progress(f"已读取边表：{len(df_edge)} 条", self.step)
        df_nodes = pd.read_csv(paths["node"])
        self.print_progress(f"已读取节点表：{len(df_nodes)} 个", self.step)
        df_feature = pd.read_csv(paths["feat"])
        self.print_progress(f"已读取特征表：{len(df_feature)} 行", self.step)
        df_hacker = pd.read_csv(paths["hacker"])
        self.print_progress(f"已读取黑客表：{len(df_hacker)} 条", self.step)
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
        """带进度显示的k-hop扩张"""
        if k < 1 or not seeds:
            return list(seeds)
        self.print_progress("构建邻接表...", self.step)
        adj = {}
        for idx, row in df_edge.iterrows():
            # 每10000行显示一次进度
            if idx % 10000 == 0:
                self.print_progress(f"构建邻接表：{idx}/{len(df_edge)} 行", self.step)
            u, v = row["from"], row["to"]
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        self.print_progress("开始k-hop扩张...", self.step)
        visited = set(seeds)
        current = set(seeds)
        for hop in range(k):
            self.print_progress(f"k-hop扩张：第{hop+1}/{k}跳", self.step)
            nxt = set()
            for u in current:
                nxt.update(adj.get(u, set()))
            nxt -= visited
            visited.update(nxt)
            current = nxt
            self.print_progress(f"第{hop+1}跳完成：新增节点{len(nxt)}个", self.step)
        
        self.print_progress(f"k-hop扩张完成：总节点数{len(visited)}个", self.step)
        return list(visited)

    def build_train(self, df_nodes, df_feature, df_edge):
        normal_num = int(len(self.train_hacker_ids) * self.normal_ratio)
        normal_num = min(normal_num, len(self.normal_nodes_all))
        self.print_progress(f"采样普通节点：{normal_num}个（黑客数×{self.normal_ratio}）", self.step)
        train_normal = random.sample(list(self.normal_nodes_all), normal_num) if normal_num > 0 else []
        seeds = list(self.train_hacker_ids) + train_normal
        
        self.print_progress(f"开始k-hop扩张，种子节点数={len(seeds)}", self.step)
        train_all = self.get_k_hop(df_edge, seeds, self.expand_hop)
        
        self.print_progress("筛选训练集节点/特征/边...", self.step)
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
        self.print_progress("获取训练集节点集合...", self.step)
        train_ids = set(self.train_nodes["address"].unique())

        # 测试节点 = 测试黑客 + 普通节点
        self.print_progress("构建测试集种子节点...", self.step)
        test_seed_nodes = list(self.test_hacker_ids) + list(self.normal_nodes_all)
        test_seed_nodes = list(set(test_seed_nodes))

        # 彻底剔除训练集节点（核心）
        self.print_progress("剔除训练集节点...", self.step)
        test_nodes_pool = [n for n in test_seed_nodes if n not in train_ids]

        # 只取在 nodes/feature 里存在的
        self.print_progress("筛选测试集节点/特征/边...", self.step)
        test_nodes = df_nodes[df_nodes["address"].isin(test_nodes_pool)].copy()
        test_feat  = df_feature[df_feature["address"].isin(test_nodes_pool)].copy()
        test_edge  = df_edge[
            (df_edge["from"].isin(test_nodes_pool)) &
            (df_edge["to"].isin(test_nodes_pool))
        ].copy()

        return test_nodes, test_feat, test_edge
