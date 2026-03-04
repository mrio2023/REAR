import pandas as pd
import random
import os
import sys
import time
import numpy as np

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
        
        # 1. 读取原始数据（优化IO）
        self.step += 1
        self.print_progress("读取原始数据", self.step)
        df_edge, df_nodes, df_feature, df_hacker = self.read_data()
      
        # 2. 过滤小社区（向量化优化）
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
        # 优化：用isin+~向量化操作，避免循环
        mask = ~df_nodes["address"].isin(all_hacker_ids)
        self.normal_nodes_all = set(df_nodes.loc[mask, "address"].unique())
        self.hacker2tag = dict(zip(self.df_hacker["address"], self.df_hacker["name_tag"]))
        self.tag2nodes = self.df_hacker.groupby("name_tag")["address"].apply(set).to_dict()
        self.print_progress("筛选完成：普通节点数={}，黑客节点数={}".format(len(self.normal_nodes_all), len(all_hacker_ids)), self.step, end="\n")
        
        # 5. 构建训练集（最耗时：k-hop扩张，重点优化）
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
        """优化IO读取：1. 支持Parquet 2. 批量读取 3. 类型指定"""
        base_path = os.path.join("codes", "df", self.dfname)
        
        # 优先读取Parquet格式（比CSV快5-10倍）
        paths = {
            "edge": os.path.join(base_path, f"{self.dfname}_edgelist"),
            "node": os.path.join(base_path, f"{self.dfname}_node_classes"),
            "feat": os.path.join(base_path, f"{self.dfname}_features"),
            "hacker": os.path.join(base_path, f"{self.dfname}_hacker")
        }
        
        # 定义读取函数，优先Parquet，兼容CSV
        def read_file(path):
            if os.path.exists(f"{path}.parquet"):
                return pd.read_parquet(f"{path}.parquet")
            elif os.path.exists(f"{path}.csv"):
                # 指定dtype减少内存占用，加速读取
                return pd.read_csv(
                    f"{path}.csv",
                    low_memory=False,
                    # 根据实际数据类型调整，示例：
                    # dtype={"address": str, "from": str, "to": str}
                )
            else:
                raise FileNotFoundError(f"文件不存在：{path}")
        
        # 逐个读取并显示进度
        self.print_progress(f"开始读取边表", self.step)
        df_edge = read_file(paths["edge"])
        self.print_progress(f"已读取边表：{len(df_edge)} 条", self.step)
        
        self.print_progress(f"开始读取节点表", self.step)
        df_nodes = read_file(paths["node"])
        self.print_progress(f"已读取节点表：{len(df_nodes)} 个", self.step)
        
        self.print_progress(f"开始读取特征表", self.step)
        df_feature = read_file(paths["feat"])
        self.print_progress(f"已读取特征表：{len(df_feature)} 行", self.step)
        
        self.print_progress(f"开始读取黑客表", self.step)
        df_hacker = read_file(paths["hacker"])
        self.print_progress(f"已读取黑客表：{len(df_hacker)} 条", self.step)
        
        return df_edge, df_nodes, df_feature, df_hacker

    def filter_small_com(self, df_hacker):
        """向量化优化，替代循环"""
        # 过滤空值（向量化操作）
        df = df_hacker.dropna(subset=["name_tag"]).copy()
        
        # 统计社区大小（向量化，比循环快）
        com_size = df["name_tag"].value_counts()
        large_tags = com_size[com_size >= self.min_com_size].index
        
        # 筛选大社区（向量化）
        df_filtered = df[df["name_tag"].isin(large_tags)].copy()
        
        return df_filtered

    def split_by_community(self):
        random.shuffle(self.all_com_tags)
        train_com_num = max(1, int(0.75 * len(self.all_com_tags)))
        train_com_tags = self.all_com_tags[:train_com_num]
        test_com_tags = self.all_com_tags[train_com_num:]
        return train_com_tags, test_com_tags

    def get_k_hop(self, df_edge, seeds, k):
        """
        核心优化：
        1. 用groupby替代iterrows构建邻接表（快10-100倍）
        2. 用numpy数组加速集合操作
        3. 减少进度打印频率，降低IO开销
        """
        if k < 1 or not seeds:
            return list(seeds)
        
        self.print_progress("构建邻接表（优化版）...", self.step)
        
        # ========== 优化1：用groupby构建邻接表，替代iterrows ==========
        # 将边表转换为邻接表（向量化操作，无循环）
        def build_adjacency(df):
            # 合并from和to的分组
            df_from = df.groupby("from")["to"].apply(set).reset_index()
            df_to = df.groupby("to")["from"].apply(set).reset_index()
            
            # 重命名列
            df_from.columns = ["node", "neighbors"]
            df_to.columns = ["node", "neighbors"]
            
            # 合并两个方向的邻居
            df_combined = pd.concat([df_from, df_to]).groupby("node")["neighbors"].apply(
                lambda x: set.union(*x) if len(x) > 0 else set()
            ).to_dict()
            
            return df_combined
        
        adj = build_adjacency(df_edge)
        
        self.print_progress("开始k-hop扩张（优化版）...", self.step)
        visited = set(seeds)
        current = set(seeds)
        
        # ========== 优化2：减少进度打印频率 ==========
        progress_interval = max(1000, len(adj) // 100)  # 每1%打印一次
        
        for hop in range(k):
            self.print_progress(f"k-hop扩张：第{hop+1}/{k}跳", self.step)
            nxt = set()
            
            # 遍历当前节点（用numpy加速）
            current_arr = np.array(list(current))
            for idx, u in enumerate(current_arr):
                # 每1000个节点打印一次进度，减少IO
                if idx % progress_interval == 0:
                    self.print_progress(f"处理第{hop+1}跳：{idx}/{len(current_arr)} 节点", self.step)
                nxt.update(adj.get(u, set()))
            
            # 去重并更新
            nxt -= visited
            visited.update(nxt)
            current = nxt
            
            self.print_progress(f"第{hop+1}跳完成：新增节点{len(nxt)}个", self.step)
        
        self.print_progress(f"k-hop扩张完成：总节点数{len(visited)}个", self.step)
        return list(visited)

    def build_train(self, df_nodes, df_feature, df_edge):
        """优化随机采样和筛选逻辑"""
        normal_num = int(len(self.train_hacker_ids) * self.normal_ratio)
        normal_num = min(normal_num, len(self.normal_nodes_all))
        self.print_progress(f"采样普通节点：{normal_num}个（黑客数×{self.normal_ratio}）", self.step)
        
        # 优化：提前转换为列表，避免重复转换
        normal_nodes_list = list(self.normal_nodes_all)
        train_normal = random.sample(normal_nodes_list, normal_num) if normal_num > 0 else []
        seeds = list(self.train_hacker_ids) + train_normal
        
        self.print_progress(f"开始k-hop扩张，种子节点数={len(seeds)}", self.step)
        train_all = self.get_k_hop(df_edge, seeds, self.expand_hop)
        
        self.print_progress("筛选训练集节点/特征/边（向量化）...", self.step)
        # 优化：提前转换为集合，加速isin操作
        train_all_set = set(train_all)
        
        # 向量化筛选，替代循环
        train_nodes = df_nodes[df_nodes["address"].isin(train_all_set)].copy()
        train_feat  = df_feature[df_feature["address"].isin(train_all_set)].copy()
        train_edge  = df_edge[
            (df_edge["from"].isin(train_all_set)) & 
            (df_edge["to"].isin(train_all_set))
        ].copy()
        
        return train_nodes, train_feat, train_edge

    def build_test(self, df_nodes, df_feature, df_edge):
        """优化测试集构建，减少重复计算"""
        self.print_progress("获取训练集节点集合...", self.step)
        train_ids = set(self.train_nodes["address"].unique())

        # 测试节点 = 测试黑客 + 普通节点
        self.print_progress("构建测试集种子节点...", self.step)
        # 优化：用集合操作替代列表拼接，加速去重
        test_seed_nodes = self.test_hacker_ids.union(self.normal_nodes_all)

        # 彻底剔除训练集节点（核心）
        self.print_progress("剔除训练集节点...", self.step)
        # 优化：集合差集操作，比列表推导式快
        test_nodes_pool = test_seed_nodes - train_ids

        # 只取在 nodes/feature 里存在的
        self.print_progress("筛选测试集节点/特征/边（向量化）...", self.step)
        test_nodes = df_nodes[df_nodes["address"].isin(test_nodes_pool)].copy()
        test_feat  = df_feature[df_feature["address"].isin(test_nodes_pool)].copy()
        test_edge  = df_edge[
            (df_edge["from"].isin(test_nodes_pool)) &
            (df_edge["to"].isin(test_nodes_pool))
        ].copy()

        return test_nodes, test_feat, test_edge

# ===================== 额外优化：CSV转Parquet（一次性操作） =====================
def convert_csv_to_parquet(dfname):
    """将CSV文件转换为Parquet格式，大幅提升后续读取速度"""
    base_path = os.path.join("codes", "df", dfname)
    files = [
        f"{dfname}_edgelist.csv",
        f"{dfname}_node_classes.csv",
        f"{dfname}_features.csv",
        f"{dfname}_hacker.csv"
    ]
    
    for file in files:
        csv_path = os.path.join(base_path, file)
        parquet_path = csv_path.replace(".csv", ".parquet")
        
        if os.path.exists(csv_path) and not os.path.exists(parquet_path):
            print(f"转换 {file} 到 Parquet 格式...")
            df = pd.read_csv(csv_path, low_memory=False)
            df.to_parquet(parquet_path, index=False)
            print(f"转换完成：{parquet_path}")
