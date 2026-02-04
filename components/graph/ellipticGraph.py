import numpy as np
import pandas as pd
import os

class EllipticGraph:  # 独立类，与原Graph隔离，专门适配Elliptic
    def __init__(self, df_name):
        # 核心图属性（与原Graph保持一致，保证调用逻辑统一）
        self.adjacency_ma = []  # K-ego子图的邻接矩阵列表（对应各离散快照）
        self.degree_tab = pd.DataFrame()  # 节点度数表
        self.parent = None  # 原始大图（父实例，EllipticGraph类型）
        self.snapshots = []  # 离散快照标记列表 [1,2,...,49]
        self.snapshot_num = 0  # 快照数量
        self.df_name = df_name  # 数据集名称（Elliptic）
        self.ego_nodes = []  # K-ego子图节点列表
        self.node_trade_freq = []  # 节点归一化交易频率
        self.node_count = 0  # K-ego子图节点数量
        self.adj_dict = {}  # 核心邻接字典 {node: [(neighbor, timestamp, amount)]}
        self.seeds = []  # 种子节点列表
        
        # Elliptic 专属属性（适配数据集结构，与原Graph区分）
        self.node_classes = pd.DataFrame()  # 节点标签数据（合法/非法/未知）
        self.node_features = pd.DataFrame()  # 节点精简特征（去共线性后）
        self.edge_df = pd.DataFrame()  # 原始边数据（无时间戳）
        self.node_active_ts = {}  # 节点活跃时间映射 {node: [ts1, ts2,...]}
        self.edge_count = 0  # 原始边总数

    def init_graph(self, k=3, 
                   edge_src_col='from', edge_tgt_col='to',
                   ts_col='timeStamp', core_cols=["address", "timeStamp"],
                   feat_threshold=0.8, var_threshold=1e-4):
        """
        完整初始化流程（专门适配Elliptic）：
        1. 加载节点特征、边、种子、标签数据
        2. 预处理特征（去低方差、去共线性）
        3. 构建节点活跃时间映射
        4. 为无时间戳边分配时间（基于节点共同活跃时间）
        5. 构建邻接字典与原始大图
        6. 构建并合并K-ego子图
        """
        # ========== 步骤1：定义Elliptic文件路径（固定格式，无需修改） ==========
        df_dir = os.path.join("df", self.df_name)
        # 定义各类文件路径（适配Elliptic标准结构）
        feat_path = os.path.join(df_dir, f"{self.df_name}_features.csv")
        cleaned_feat_path = os.path.join(df_dir, f"{self.df_name}_features_non_collinear.csv")
        edge_path = os.path.join(df_dir, f"{self.df_name}_edgelist.csv")
        seed_path = os.path.join(df_dir, f"{self.df_name}_seed.csv")
        node_class_path = os.path.join(df_dir, f"{self.df_name}_node_classes.csv")

        # 打印路径验证（方便排查文件缺失）
        print("="*50)
        print(f"开始初始化 Elliptic 图数据，数据集路径：{df_dir}")
        print(f"原始特征路径：{feat_path}")
        print(f"精简特征路径：{cleaned_feat_path}")
        print(f"边数据路径：{edge_path}")
        print("="*50)

        # ========== 步骤2：读取种子节点（处理类型，避免数字/字符串冲突） ==========
        try:
            df_seed = pd.read_csv(seed_path)
            print(f"\n加载种子节点成功，共 {len(df_seed)} 个种子")
            # 转为字符串，保证后续映射一致（Elliptic address是数字）
            self.seeds = [str(n) for n in df_seed["address"].values]
            print(f"种子节点前5个：{self.seeds[:5]}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"种子文件不存在：{e.filename}")
        except Exception as e:
            raise Exception(f"读取种子节点失败：{str(e)}")

        # ========== 步骤3：加载Elliptic核心数据（节点特征、边、标签） ==========
        try:
            # 优先加载去共线性后的精简特征，若不存在则加载原始特征并自动处理
            if os.path.exists(cleaned_feat_path):
                self.node_features = pd.read_csv(cleaned_feat_path)
                print(f"\n加载精简节点特征成功，数据形状：{self.node_features.shape}")
            else:
                print(f"\n未找到精简特征文件，自动处理原始特征（去低方差+去共线性）")
                self.node_features = self._process_elliptic_features(feat_path, core_cols, feat_threshold, var_threshold)
            
            # 加载边数据（无时间戳）
            self.edge_df = pd.read_csv(edge_path)
            # 边节点转为字符串，统一格式
            self.edge_df[edge_src_col] = self.edge_df[edge_src_col].astype(str)
            self.edge_df[edge_tgt_col] = self.edge_df[edge_tgt_col].astype(str)
            
            # 加载节点标签（可选，不存在则跳过）
            if os.path.exists(node_class_path):
                self.node_classes = pd.read_csv(node_class_path)
                self.node_classes["address"] = self.node_classes["address"].astype(str)
                print(f"加载节点标签成功，数据形状：{self.node_classes.shape}")
            
            # 统计基础信息
            self.edge_count = len(self.edge_df)
            unique_nodes = set(self.node_features["address"].astype(str).values)
            self.node_count_total = len(unique_nodes)
            print(f"加载边数据成功，共 {self.edge_count} 条边，{self.node_count_total} 个唯一节点")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"核心文件不存在：{e.filename}")
        except Exception as e:
            raise Exception(f"加载Elliptic数据失败：{str(e)}")

        # ========== 步骤4：提取离散快照+构建节点活跃时间映射 ==========
        # 提取唯一时间戳并排序（快照列表）
        self.snapshots = sorted(self.node_features[ts_col].unique())
        self.snapshot_num = len(self.snapshots)
        print(f"\n提取离散快照成功，共 {self.snapshot_num} 个快照，快照范围：{self.snapshots[0]} ~ {self.snapshots[-1]}")

        # 构建 {节点: [活跃时间列表]} 映射
        for addr, ts_group in self.node_features.groupby("address")["timeStamp"]:
            self.node_active_ts[str(addr)] = sorted(ts_group.unique())
        print(f"构建节点活跃时间映射成功，示例（第一个节点）：{list(self.node_active_ts.items())[0]}")

        # ========== 步骤5：为无时间戳边分配时间（核心：基于节点共同活跃时间） ==========
        raw_adj_dict = {}
        for idx, row in self.edge_df.iterrows():
            u = str(row[edge_src_col])
            v = str(row[edge_tgt_col])
            amt = 0.0  # Elliptic无交易金额，填充0

            # 获取u和v的共同活跃时间（边仅在共同活跃时间出现，符合时序逻辑）
            u_active = self.node_active_ts.get(u, [])
            v_active = self.node_active_ts.get(v, [])
            common_ts = list(set(u_active) & set(v_active))

            # 若无共同活跃时间，取任意一方的活跃时间（保证边不丢失）
            if not common_ts:
                common_ts = u_active + v_active
            if not common_ts:
                common_ts = self.snapshots[:1]  # 兜底：取第一个快照

            # 填充邻接字典（一条边对应所有共同活跃时间）
            for ts in common_ts:
                ts_int = int(ts)
                # 填充u的邻居
                if u not in raw_adj_dict:
                    raw_adj_dict[u] = []
                raw_adj_dict[u].append((v, ts_int, amt))
                # 填充v的邻居（无向图逻辑，Elliptic边无方向）
                if v not in raw_adj_dict:
                    raw_adj_dict[v] = []
                raw_adj_dict[v].append((u, ts_int, amt))
        self.adj_dict = raw_adj_dict
        print(f"\n构建邻接字典成功，共 {len(raw_adj_dict)} 个有边节点，示例（第一个节点）：{list(raw_adj_dict.items())[0][:2]}")

        # ========== 步骤6：初始化parent（原始大图，EllipticGraph实例） ==========
        self.parent = EllipticGraph(self.df_name)
        self.parent.adj_dict = raw_adj_dict
        self.parent.node_features = self.node_features
        self.parent.edge_df = self.edge_df
        self.parent.snapshots = self.snapshots
        self.parent.snapshot_num = self.snapshot_num
        self.parent.node_count_total = self.node_count_total
        self.parent.edge_count = self.edge_count
        self.parent.seeds = self.seeds
        print(f"初始化原始大图（parent）成功，存储完整Elliptic数据")

        # ========== 步骤7：构建并合并K-ego子图（适配Elliptic时序逻辑） ==========
        all_ego_nodes_set = set()
        subgraph_list = []

        # 遍历种子节点，构建单个K-ego子图
        for seed in self.seeds:
            if seed not in raw_adj_dict:
                print(f"\n警告：种子节点 {seed} 无关联边，跳过")
                continue

            # 调用子图构建方法
            ego_nodes, snapshot_adjs, node_trade_freq, node_count, _ = \
                self._build_temporal_k_ego_subgraph(seed, raw_adj_dict, self.snapshots, k)

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
            print(f"种子节点 {seed}：构建K-ego子图成功，共 {node_count} 个节点")

        # 处理无有效子图的情况
        if not all_ego_nodes_set:
            print("\n警告：无有效K-ego子图被构建")
            self.ego_nodes = []
            self.node_count = 0
            self.adjacency_ma = []
            self.node_trade_freq = np.array([])
            return

        # ========== 步骤8：合并所有种子节点的K-ego子图 ==========
        all_ego_nodes = list(all_ego_nodes_set)
        node_to_new_id = {node: idx for idx, node in enumerate(all_ego_nodes)}
        self.node_count = len(all_ego_nodes)

        # 初始化统一邻接矩阵和交易频率
        unified_adjacencies = [np.zeros((self.node_count, self.node_count), dtype=int) for _ in range(self.snapshot_num)]
        unified_trade_freq = np.zeros(self.node_count, dtype=float)

        # 合并每个子图的数据
        for subgraph in subgraph_list:
            ego_nodes = subgraph['ego_nodes']
            snapshot_adjs = subgraph['snapshot_adjs']
            node_trade_freq = subgraph['node_trade_freq']

            # 构建局部节点到统一节点的映射
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

        # 归一化交易频率（避免除以0）
        if np.max(unified_trade_freq) > 0:
            unified_trade_freq = unified_trade_freq / (np.max(unified_trade_freq) + 1e-8)

        # 更新K-ego子图核心属性
        self.ego_nodes = all_ego_nodes
        self.adjacency_ma = unified_adjacencies
        self.node_trade_freq = unified_trade_freq
        print(f"\n合并所有K-ego子图成功，总节点：{self.node_count}，总快照：{self.snapshot_num}")

        # ========== 步骤9：生成节点度数表 ==========
        if self.ego_nodes and self.adj_dict:
            self.degree_tab = pd.DataFrame({
                "node": self.ego_nodes,
                "degree": [len(set([neigh for neigh, _, _ in self.adj_dict.get(node, [])])) for node in self.ego_nodes],
                "normalized_trade_freq": self.node_trade_freq[:len(self.ego_nodes)]
            })
            # 合并节点标签（可选）
            if not self.node_classes.empty:
                self.degree_tab = pd.merge(
                    self.degree_tab,
                    self.node_classes,
                    left_on="node",
                    right_on="address",
                    how="left"
                ).drop(columns=["address"])
            print(f"生成节点度数表成功，共 {len(self.degree_tab)} 条记录")
        else:
            self.degree_tab = pd.DataFrame()
            print("生成节点度数表失败：无有效节点或邻接字典")

        print("\n===== Elliptic 图初始化流程全部完成 =====")

    def _process_elliptic_features(self, feat_path, core_cols, feat_threshold, var_threshold):
        """
        内部辅助方法：处理Elliptic原始特征（去低方差+去共线性），保存精简特征文件
        """
        # 加载原始特征
        df = pd.read_csv(feat_path)
        df["address"] = df["address"].astype(str)
        feature_cols = [col for col in df.columns if col.startswith("feature_")]
        X = df[feature_cols].copy()

        # 步骤1：去低方差特征
        high_var_features = X.var()[X.var() >= var_threshold].index.tolist()
        X_high_var = X[high_var_features].copy()

        # 步骤2：去共线性特征（分组保留方差最大）
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(high_var_features)
            for i, f1 in enumerate(high_var_features):
                for j, f2 in enumerate(high_var_features[i+1:]):
                    corr, _ = np.corrcoef(X_high_var[f1], X_high_var[f2])[0, 1]
                    if abs(corr) >= feat_threshold:
                        G.add_edge(f1, f2)
            # 分组保留方差最大特征
            collinear_groups = list(nx.connected_components(G))
            keep_features = []
            feat_var = X_high_var.var()
            for group in collinear_groups:
                max_var_feat = max(group, key=lambda f: feat_var[f])
                keep_features.append(max_var_feat)
        except ImportError:
            print("未安装networkx，采用简单两两筛选（保留先出现特征）")
            keep_features = []
            drop_features = set()
            for i, f1 in enumerate(high_var_features):
                if f1 in drop_features:
                    continue
                keep_features.append(f1)
                for f2 in high_var_features[i+1:]:
                    if f2 in drop_features:
                        continue
                    corr, _ = np.corrcoef(X_high_var[f1], X_high_var[f2])[0, 1]
                    if abs(corr) >= feat_threshold:
                        drop_features.add(f2)

        # 步骤3：拼接精简特征并保存
        X_final = X_high_var[keep_features].copy()
        df_final = pd.concat([df[core_cols].reset_index(drop=True), X_final.reset_index(drop=True)], axis=1)
        df_final["address"] = df_final["address"].astype(str)

        # 保存精简特征文件（后续可直接加载）
        output_path = os.path.join(os.path.dirname(feat_path), f"{self.df_name}_features_non_collinear.csv")
        df_final.to_csv(output_path, index=False)
        print(f"精简特征文件已保存至：{output_path}，数据形状：{df_final.shape}")

        return df_final

    @staticmethod
    def calc_temporal_density(edge_count, node_count, snapshot_num):
        """
        计算时序密度（单位时间内节点的平均边数）
        """
        return edge_count / (node_count * snapshot_num) if node_count * snapshot_num != 0 else 0.0

    @staticmethod
    def _build_temporal_k_ego_subgraph(central_node, adj_dict, discrete_snapshots, k=3):
        """
        内部辅助方法：构建Elliptic专属K-ego子图（基于节点活跃时间的时序子图）
        """
        # 步骤1：获取K阶邻居节点（拓扑层面）
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

        # 步骤2：为每个快照构建邻接矩阵（基于节点活跃时间）
        snapshot_adjs = []
        node_trade_freq = np.zeros(node_count)

        for current_ts in discrete_snapshots:
            adj = np.zeros((node_count, node_count), dtype=int)
            for i, u in enumerate(ego_nodes):
                for (v, ts, amt) in adj_dict.get(u, []):
                    if ts == current_ts and v in node2idx:
                        j = node2idx[v]
                        adj[i, j] += 1
                        node_trade_freq[i] += 1
            snapshot_adjs.append(adj)

        # 归一化交易频率（避免除以0）
        node_trade_freq = node_trade_freq / (np.max(node_trade_freq) + 1e-8)

        return ego_nodes, snapshot_adjs, node_trade_freq, node_count, snapshot_num

# ========== EllipticGraph 测试函数（独立调用，不影响老代码） ==========
def test_elliptic(name="Elliptic"):
    try:
        g = EllipticGraph(name)
        g.init_graph(k=3)
        print(f"\n=== EllipticGraph 测试成功 ===")
    except Exception as e:
        print(f"\n=== EllipticGraph 测试失败：{str(e)} ===")


if __name__ == "__main__":
    test_elliptic("elliptic_txs")