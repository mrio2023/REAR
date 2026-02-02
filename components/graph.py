import numpy as np
import pandas as pd
import os

class Graph:  # 类名建议大写开头，符合Python命名规范
    def __init__(self, df_name):
        # 图的核心属性
        self.adjacency_ma = []  # 当前图（K-ego子图）的邻接矩阵列表（对应各离散快照）
        self.degree_tab = pd.DataFrame()  # 节点度数
        self.parent = None  # 原始大图（父图），初始化为None，后续赋值为Graph实例
        self.snapshots = []  # 离散快照标记列表 [1,2,3,4,5,...]（直接存储timestamp离散值）
        self.df_name = df_name  # 数据文件名（AscendEXHascker）
        self.ego_nodes = []  # K-ego子图的节点列表
        self.node_trade_freq = []  # K-ego子图节点的归一化交易频率
        self.node_count = 0  # K-ego子图节点数量
        self.adj_dict = {}  # 图的邻接字典 {node: [(neighbor, timestamp, amount)]}（核心数据结构）
        self.seeds = []  # 种子节点列表
        # 新增：原始大图专属属性（父实例会用到）
        self.nodes = pd.DataFrame()  # 原始节点数据
        self.snapshots_df = pd.DataFrame()  # 原始快照边数据
        self.edge_count = 0  # 原始边数量
        self.features=[]


    def init_graph(self, k=3, 
                   src_col='from', tgt_col='to',  # 适配snapshot边列表字段（from/to）
                   ts_col='timeStamp', amt_col='amount'):  # 适配你的timeStamp列名
        """
        完整初始化流程：加载原始大图→构建K-ego子图→保存原始大图到parent（Graph实例）
        适配离散timestamp标记（1,2,3...），摒弃无意义的时间区间
        :param k: K阶邻居，默认3
        :param src_col: CSV中源节点字段名（snapshot是from）
        :param tgt_col: CSV中目标节点字段名（snapshot是to）
        :param ts_col: CSV中时间戳字段名（你的数据是timeStamp）
        :param amt_col: CSV中交易金额字段名（你的数据无该字段，后续扩展）
        """
        # ========== 步骤1 ==========
        df_dir = os.path.join("df", self.df_name)  # 数据集文件夹路径（df/AscendEXHascker）
        node_path = os.path.join(df_dir, f"{self.df_name}_node_classes.csv")
        snapshot_path = os.path.join(df_dir, f"{self.df_name}_features.csv")  # 你的核心快照文件
        
        # 打印路径验证（方便排查）
        print(f"节点文件路径：{node_path}")
        print(f"快照文件路径：{snapshot_path}")

        # ========== 步骤2：读取种子节点（从seed.txt，路径修正） ==========
        try:
            with open("seed.txt", "r", encoding="utf-8") as f:
                seed_content = f.read()
                seed_str_list = seed_content.split()
                self.seeds = [seed_str for seed_str in seed_str_list]
            print(f"成功读取种子节点：{self.seeds[:5]}...（共{len(self.seeds)}个）")
        except FileNotFoundError:
            raise FileNotFoundError("seed.txt不存在于项目根目录，请将其放在根目录下")
     

        # ========== 步骤3：加载节点和快照CSV ==========
        try:
         snapshots_df = pd.read_csv(snapshot_path)
         print(f"快照数据形状: {snapshots_df.shape}")
         print(f"快照数据列: {list(snapshots_df.columns)}")
         nodes = pd.read_csv(node_path)
        except FileNotFoundError as e:
         raise FileNotFoundError(f"文件不存在，请检查文件夹/文件名是否正确：{e.filename}")
        except Exception as e:
         raise Exception(f"加载CSV失败：{str(e)}")

        # ========== 步骤4：修复节点mapping赋值，给每个节点分配唯一索引 ==========
        nodes["mapping"] = range(len(nodes))
        address_to_mapping = dict(zip(nodes["address"], nodes["mapping"]))
        node_count = len(nodes)
        snap_edge_count = len(snapshots_df)
        print(f"成功加载{node_count}个节点，{snap_edge_count}条时序边（快照数据）")

        # ========== 步骤5：提取离散timestamp，生成快照列表（核心修改：摒弃区间） ==========
        # 1. 获取所有唯一离散timestamp并排序（直接作为快照列表，无区间）
        self.snapshots = sorted(snapshots_df[ts_col].unique())
        # 2. 直接赋值快照数量（列表长度即为快照数）
        self.snapshot_num = len(self.snapshots)
        
        print(f"提取到{self.snapshot_num}个离散快照标记：{self.snapshots}")


        print(f"\n=== 验证ts类型是否匹配 ===")
        print(f"快照列表self.snapshots前5个元素：{self.snapshots[:5]}")
        print(f"快照元素类型：{type(self.snapshots[0]) if len(self.snapshots) > 0 else '无'}")

        # ========== 步骤6：构建raw_adj_dict（填充真实离散timestamp，适配你的数据） ==========
        raw_adj_dict = {}
        for idx, row in snapshots_df.iterrows():
            try:
                u = row[src_col]  # 源节点address
                v = row[tgt_col]  # 目标节点address
                ts = int(row[ts_col])  # 离散标记直接转为整数（贴合1,2,3...格式）
                amt = 0.0  # 你的数据无amount字段，暂时填充0，后续可扩展
                
                # 填充邻接字典
                if u not in raw_adj_dict:
                    raw_adj_dict[u] = []
                raw_adj_dict[u].append((v, ts, amt))
            except KeyError as e:
                print(f"警告：第{idx}行缺少字段，跳过 → {e}")
                continue
        
        # 关键：将构建好的raw_adj_dict赋值给类属性self.adj_dict
        self.adj_dict = raw_adj_dict

        # ========== 新增：步骤7：初始化parent为Graph实例（原始大图） ==========
        # 1. 创建一个全新的Graph实例作为父图（原始大图）
        self.parent = Graph(self.df_name)
        # 2. 为父图填充原始大图的所有核心数据
        self.parent.adj_dict = raw_adj_dict  # 原始完整邻接字典
        self.parent.nodes = nodes  # 原始完整节点数据
        self.parent.snapshots_df = snapshots_df  # 原始完整快照边数据
        self.parent.snapshots = self.snapshots  # 原始快照列表
        self.parent.snapshot_num = self.snapshot_num  # 原始快照数量
        self.parent.node_count = node_count  # 原始节点总数
        self.parent.edge_count = snap_edge_count  # 原始边总数
        self.parent.seeds = self.seeds  # 种子节点列表
        # 3. 父图无需构建ego子图，因此ego相关属性保持默认空值即可
        print(f"成功初始化parent为Graph实例，存储原始大图完整数据")

        # ========== 打印adj_dict中ts的类型（修复UnboundLocalError，补充完整边界判断） ==========
        first_ts = '无有效边数据'
        first_ts_type = '无'
        node_edges=-1
        if len(self.adj_dict) > 0:
            first_node = list(self.adj_dict.keys())[0]
            node_edges = self.adj_dict.get(first_node, [])
            if len(node_edges) > 0:
                first_ts = node_edges[0][1]
                first_ts_type = type(first_ts)

        print(f"adj_dict中第一个ts值：{first_ts}")
        print(f"adj_dict中ts的类型：{first_ts_type}")

        # ========== 步骤8：填充静态邻接矩阵（基于所有时序边） ==========
        adj_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
        self.adjacency_ma.append(adj_matrix)
        
        for idx, row in snapshots_df.iterrows():
            try:
                u = row[src_col]
                v = row[tgt_col]
                u_mapping = address_to_mapping[u]
                v_mapping = address_to_mapping[v]
                self.adjacency_ma[0][u_mapping][v_mapping] = 1
            except KeyError as e:
                print(f"警告：第{idx}行节点不存在，跳过 → {e}")
                continue

        # ========== 步骤9：基于种子节点seeds构建K-ego子图（适配离散快照） ==========
        all_ego_nodes_set = set()
        subgraph_list = []

        # 第一遍：收集所有子图的节点和边信息
        for seed in self.seeds:
            if seed not in raw_adj_dict:
                print(f"警告：种子节点{seed}不存在于原始大图中，跳过该节点")
                continue

            # 调用修改后的K-ego子图构建方法（传入离散快照列表）
            ego_nodes, snapshot_adjs, node_trade_freq, node_count, _ = \
                self.build_temporal_k_ego_subgraph(seed, raw_adj_dict, self.snapshots, k)

            # 记录子图信息
            subgraph_info = {
                'seed': seed,
                'ego_nodes': ego_nodes,
                'snapshot_adjs': snapshot_adjs,
                'node_trade_freq': node_trade_freq,
                'node_count': node_count
            }
            subgraph_list.append(subgraph_info)
            all_ego_nodes_set.update(ego_nodes)
            
            print(f"种子节点{seed}: {node_count}个节点")

        # 如果没有有效子图，直接返回
        if not all_ego_nodes_set:
            print("警告：没有有效的K-ego子图被构建")
            self.ego_nodes = []
            self.node_count = 0
            self.adjacency_ma = []
            self.node_trade_freq = np.array([])
        else:
            # 创建统一的节点映射（重新编号0,1,2,...）
            all_ego_nodes = list(all_ego_nodes_set)
            node_to_new_id = {node: idx for idx, node in enumerate(all_ego_nodes)}
            total_nodes = len(all_ego_nodes)
            
            print(f"合并后总节点数: {total_nodes}")

            # 初始化统一的邻接矩阵和交易频率（适配离散快照数）
            unified_adjacencies = [np.zeros((total_nodes, total_nodes), dtype=int) for _ in range(self.snapshot_num)]
            unified_trade_freq = np.zeros(total_nodes, dtype=float)

            # 第二遍：合并所有子图到统一空间
            for subgraph in subgraph_list:
                ego_nodes = subgraph['ego_nodes']
                snapshot_adjs = subgraph['snapshot_adjs']
                node_trade_freq = subgraph['node_trade_freq']
                
                # 将子图节点映射到统一ID
                local_to_unified = [node_to_new_id[node] for node in ego_nodes]
                
                # 合并邻接矩阵
                for snap_idx in range(min(self.snapshot_num, len(snapshot_adjs))):
                    local_adj = snapshot_adjs[snap_idx]
                    for local_i in range(local_adj.shape[0]):
                        for local_j in range(local_adj.shape[1]):
                            if local_adj[local_i, local_j] != 0:
                                unified_i = local_to_unified[local_i]
                                unified_j = local_to_unified[local_j]
                                unified_adjacencies[snap_idx][unified_i, unified_j] = 1
                
                # 合并交易频率（累加）
                for local_idx, node in enumerate(ego_nodes):
                    if local_idx < len(node_trade_freq):
                        unified_id = node_to_new_id[node]
                        unified_trade_freq[unified_id] += node_trade_freq[local_idx]

            # 归一化交易频率
            if np.max(unified_trade_freq) > 0:
                unified_trade_freq = unified_trade_freq / np.max(unified_trade_freq)
                

            # 更新类属性
            self.ego_nodes = all_ego_nodes
            self.node_count = total_nodes
            self.adjacency_ma = unified_adjacencies
            self.node_trade_freq = unified_trade_freq

            print(f"图初始化完成，K-ego子图合并成功！总节点: {self.node_count}, 快照数: {self.snapshot_num}")

        # ========== 步骤10：补充节点度数表 ==========
        if self.ego_nodes and self.adj_dict:
            self.degree_tab = pd.DataFrame({
                "node": self.ego_nodes,
                "degree": [len([neigh for neigh, _, _ in self.adj_dict.get(node, [])]) for node in self.ego_nodes],
                "normalized_trade_freq": self.node_trade_freq[:len(self.ego_nodes)]
            })
            print(f"节点度数表生成成功，共{len(self.degree_tab)}条记录")
        else:
            self.degree_tab = pd.DataFrame()
            print("节点度数表生成失败：无有效节点或邻接字典")

        print("===== 图初始化流程全部完成 =====")

    # 计算时序密度（核心指标：单位时间内节点的平均边数）
    @staticmethod
    def calc_temporal_density(edge_count, node_count, snapshot_num):
        """
        edge_count: 子图总边数
        node_count: 子图节点数
        snapshot_num: 快照数量（离散标记数）
        """
        return edge_count / (node_count * snapshot_num) if node_count * snapshot_num != 0 else 0.0

    # 构建带离散快照的K-ego子图（核心修改：匹配离散timestamp，无区间）
    @staticmethod
    def build_temporal_k_ego_subgraph(central_node, adj_dict, discrete_snapshots, k=3):
        """
        central_node: 中心节点ID
        adj_dict: 带离散时间戳的邻接字典 {node: [(neighbor, timestamp, amount)]}
        discrete_snapshots: 离散快照标记列表 [1,2,3,...]
        k: K阶邻居
        return: 子图节点列表、每个快照的邻接矩阵、节点交易频率权重、节点数、快照数
        """
        # 第一步：获取K-ego子图的所有节点（拓扑层面）
        ego_nodes = {central_node}
        current_nodes = {central_node}
        for _ in range(k):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update([neigh for neigh, _, _ in adj_dict.get(node, [])])
            current_nodes = next_nodes - ego_nodes
            ego_nodes.update(current_nodes)
        ego_nodes = list(ego_nodes)
        node2idx = {n: i for i, n in enumerate(ego_nodes)}
        node_count = len(ego_nodes)
        snapshot_num = len(discrete_snapshots)

        # 第二步：为每个离散快照构建邻接矩阵（核心修改：直接匹配离散timestamp）
        snapshot_adjs = []  # [snapshot_num, node_count, node_count]
        node_trade_freq = np.zeros(node_count)

        for current_ts in discrete_snapshots:  # 遍历离散快照标记（1,2,3...）
            adj = np.zeros((node_count, node_count))
            for i, u in enumerate(ego_nodes):
                for (v, ts, amt) in adj_dict.get(u, []):
                    # 核心修改：不再判断区间，直接匹配「交易ts是否等于当前快照标记」
                    if ts == current_ts and v in node2idx:
                        j = node2idx[v]
                        adj[i, j] += 1
                        node_trade_freq[i] += 1
            snapshot_adjs.append(adj)

        # 归一化交易频率（添加1e-8避免除以0）
        node_trade_freq = node_trade_freq / (np.max(node_trade_freq) + 1e-8)

        return ego_nodes, snapshot_adjs, node_trade_freq, node_count, snapshot_num

