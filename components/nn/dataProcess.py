import os
import random
import numpy as np

class PreprocessedDataLoader:
    def __init__(self, dataset_name, train_ratio=0.8, min_com_size=3, seed=42):
        """
        返回全局图，以及训练/测试社区划分。

        :param dataset_name: 数据集名称
        :param train_ratio: 训练社区比例（剩余为测试）
        :param min_com_size: 最小社区大小
        :param seed: 随机种子
        """
        self.dataset = dataset_name
        self.root = "/home/u2023312299/sci/compare/compareDatasets"
        self.train_ratio = train_ratio
        self.min_com_size = min_com_size
        random.seed(seed)

        # 文件路径
        self.ungraph_file = os.path.join(self.root, dataset_name, f"{dataset_name}-1.90.ungraph.txt")
        self.cmty_file = os.path.join(self.root, dataset_name, f"{dataset_name}-1.90.cmty.txt")
        self.feat_file = os.path.join(self.root, dataset_name, f"{dataset_name}-1.90.nodefeat.txt")

        # 加载全局数据
        self.graph = None      # 邻接表、节点数等
        self.comms = None      # 社区列表
        self.nodefeats = None  # 特征字典
        self._load_data()

        # 划分社区
        self._split_comms()
        self.check_data()


    def check_data(self):
        """检查数据是否存在异常（NaN、节点缺失、特征分布等）"""
        print("\n=== 数据检查 ===")
        # 1. 检查节点编号范围与邻接表
        nodes = set(self.graph['adj'].keys())
        if nodes:
            max_node = max(nodes)
            print(f"节点总数: {len(nodes)}，最大节点ID: {max_node}")
            # 节点ID是否连续（不要求严格连续，但可输出范围）
            if max_node + 1 != len(nodes):
                print(f"⚠️ 节点ID不连续（最大ID={max_node}，节点数={len(nodes)}），但通常无影响")
        else:
            print("❌ 图中无节点！")
            return

        # 2. 检查边数及是否自环
        edge_count = len(self.graph['edges'])
        self_loops = [e for e in self.graph['edges'] if e[0] == e[1]]
        print(f"总边数: {edge_count}，自环边数: {len(self_loops)}")
        if self_loops:
            print(f"前5个自环: {self_loops[:5]}")

        # 3. 检查社区
        comm_sizes = [len(c) for c in self.comms]
        print(f"社区数: {len(self.comms)}，平均大小: {np.mean(comm_sizes):.1f}，最小/最大: {min(comm_sizes)}/{max(comm_sizes)}")

        # 4. 检查特征文件
        if self.nodefeats:
            # 获取所有特征值的列表
            feat_values = list(self.nodefeats.values())
            if feat_values:
                # 检查是否有 NaN
                for i, vals in enumerate(feat_values[:100]):  # 检查前100个
                    if any(np.isnan(v) for v in vals):
                        print(f"❌ 节点 {list(self.nodefeats.keys())[i]} 的特征包含 NaN")
                        break
                else:
                    print("✅ 未检测到 NaN 特征值（前100个节点）")
                # 检查特征维度是否一致
                dims = [len(v) for v in feat_values]
                if len(set(dims)) != 1:
                    print(f"❌ 特征维度不一致：{set(dims)}")
                else:
                    print(f"特征维度: {dims[0]}")
                # 输出特征值范围（前5个特征）
                feat_array = np.array(feat_values[:1000])  # 取前1000个节点统计
                min_vals = feat_array.min(axis=0)
                max_vals = feat_array.max(axis=0)
                print(f"特征值范围（前5列）: {list(zip(min_vals[:5], max_vals[:5]))}")
            else:
                print("❌ 特征字典为空")
        else:
            print("⚠️ 无特征文件")

        # 5. 检查训练/测试社区划分
        print(f"训练社区数: {len(self.train_comms)}，测试社区数: {len(self.test_comms)}")
        train_nodes = set().union(*self.train_comms) if self.train_comms else set()
        test_nodes = set().union(*self.test_comms) if self.test_comms else set()
        overlap = train_nodes & test_nodes
        if overlap:
            print(f"⚠️ 训练与测试社区节点重叠，重叠数: {len(overlap)}")
        else:
            print("✅ 训练与测试社区节点无重叠")
        print("=== 检查结束 ===\n")
    def _load_data(self, outlier_threshold=2):
        """读取边、社区、特征，并自动剔除大小异常的社区"""
        edges = []
        with open(self.ungraph_file, 'r') as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append((u, v))

        comms = []
        with open(self.cmty_file, 'r') as f:
            for line in f:
                nodes = list(map(int, line.strip().split()))
                if len(nodes) >= self.min_com_size:
                    comms.append(nodes)

        # 统计社区大小
        sizes = [len(c) for c in comms]
        if sizes:
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            lower_bound = self.min_com_size
            upper_bound = mean_size + outlier_threshold * std_size
            # 过滤异常大小的社区
            filtered_comms = [c for c in comms if lower_bound <= len(c) <= upper_bound]
            print(f"原始社区数: {len(comms)}，过滤后: {len(filtered_comms)}，剔除 {len(comms)-len(filtered_comms)} 个异常大小社区（下限={lower_bound}, 上限={upper_bound:.1f})")
            comms = filtered_comms
        else:
            print("警告：没有符合条件的社区！")

        nodefeats = {}
        if os.path.exists(self.feat_file):
            with open(self.feat_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    node = int(parts[0])
                    feats = list(map(float, parts[1:]))
                    nodefeats[node] = feats

        # 构建邻接表
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        self.graph = {
            'n': max(adj.keys(), default=-1) + 1,
            'edges': edges,
            'adj': adj
        }
        self.comms = comms
        self.nodefeats = nodefeats
        print(f"加载完成：节点数 {self.graph['n']}，边数 {len(edges)}，社区数 {len(comms)}，特征节点数 {len(nodefeats)}")

    def _split_comms(self):
        """随机划分社区为训练集和测试集"""
        indices = list(range(len(self.comms)))
        random.shuffle(indices)
        split = int(len(indices) * self.train_ratio)
        train_idx = indices[:split]
        test_idx = indices[split:]

        self.train_comms = [self.comms[i] for i in train_idx]
        self.test_comms = [self.comms[i] for i in test_idx]