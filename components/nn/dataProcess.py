import os
import random
import numpy as np


class DataLoader:
    def __init__(self, dataset_name, train_ratio=0.8, min_com_size=3, seed=2026):
        """
        返回全局图，以及训练/测试社区划分。

        :param dataset_name: 数据集名称
        :param train_ratio: 训练社区比例（剩余为测试）
        :param min_com_size: 最小社区大小
        :param seed: 随机种子
        """
        self.dataset = dataset_name

        self.root = "D:\CodeSummary\sci\codes\df"

        self.train_ratio = train_ratio
        self.min_com_size = min_com_size
        random.seed(seed)

        # 文件路径
        self.ungraph_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.ungraph.txt"
        )
        self.cmty_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.cmty.txt"
        )
        self.feat_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.nodefeat.txt"
        )

        # 加载全局数据
        self.graph = None  # 邻接表、节点数等
        self.comms = None  # 社区列表
        self.nodefeats = None  # 特征字典
        self._load_data()

        # 划分社区
        self._split_comms()

    def _load_data(self, outlier_threshold=2):
        """读取边、社区、特征，并自动剔除大小异常的社区"""
        edges = []
        with open(self.ungraph_file, "r") as f:
            for line in f:
                u, v = map(int, line.strip().split())
                edges.append((u, v))

        comms = []
        with open(self.cmty_file, "r") as f:
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
            print(
                f"原始社区数: {len(comms)}，过滤后: {len(filtered_comms)}，剔除 {len(comms)-len(filtered_comms)} 个异常大小社区（下限={lower_bound}, 上限={upper_bound:.1f})"
            )
            comms = filtered_comms
        else:
            print("警告：没有符合条件的社区！")

        nodefeats = {}
        if os.path.exists(self.feat_file):
            with open(self.feat_file, "r") as f:
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

        self.graph = {"n": max(adj.keys(), default=-1) + 1, "edges": edges, "adj": adj}
        self.comms = comms
        self.nodefeats = nodefeats
        print(
            f"加载完成：节点数 {self.graph['n']}，边数 {len(edges)}，社区数 {len(comms)}，特征节点数 {len(nodefeats)}"
        )

    def _split_comms(self):
        """随机划分社区为训练集和测试集"""
        indices = list(range(len(self.comms)))
        random.shuffle(indices)
        split = int(len(indices) * self.train_ratio)
        train_idx = indices[:split]
        test_idx = indices[split:]

        self.train_comms = [self.comms[i] for i in train_idx]
        self.test_comms = [self.comms[i] for i in test_idx]
