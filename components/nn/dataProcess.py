import os
import random
import numpy as np

class PreprocessedDataLoader:
    def __init__(self, dataset_name, train_ratio=0.8, min_com_size=3, normal_ratio=2.0, seed=42):
        """
        返回全局图，以及训练/测试社区划分，并提供普通节点池用于负采样。

        :param dataset_name: 数据集名称
        :param train_ratio: 训练社区比例（剩余为测试）
        :param min_com_size: 最小社区大小
        :param normal_ratio: 普通节点采样比例（相对于训练社区节点数）
        :param seed: 随机种子
        """
        self.dataset = dataset_name
        self.root = "/home/u2023312299/sci/compare/compareDatasets"
        self.train_ratio = train_ratio
        self.min_com_size = min_com_size
        self.normal_ratio = normal_ratio
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

        # 构建普通节点池（不在任何社区中的节点）
        all_community_nodes = set()
        for comm in self.comms:
            all_community_nodes.update(comm)
        all_nodes = set(self.graph['adj'].keys())
        self.normal_nodes = all_nodes - all_community_nodes
        print(f"全局普通节点数: {len(self.normal_nodes)}")

        # 为训练准备普通节点池（可采样负样本）
        train_community_nodes = set()
        for comm in self.train_comms:
            train_community_nodes.update(comm)
        normal_needed = int(len(train_community_nodes) * self.normal_ratio)
        self.train_normal_pool = random.sample(list(self.normal_nodes), min(normal_needed, len(self.normal_nodes)))
        print(f"训练普通节点池大小: {len(self.train_normal_pool)}")

    def _load_data(self):
        """读取边、社区、特征"""
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

        nodefeats = {}
        if os.path.exists(self.feat_file):
            with open(self.feat_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    node = int(parts[0])
                    feats = list(map(int, parts[1:]))
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

    def get_train_seeds(self, num_seeds=None):
        """返回训练社区中每个社区的第一个节点作为种子（或者随机选择）"""
        seeds = []
        for comm in self.train_comms:
            seeds.append(comm[0])
        if num_seeds is not None:
            seeds = seeds[:num_seeds]
        return seeds

    def get_test_seeds(self, num_seeds=None):
        """返回测试社区中每个社区的第一个节点作为种子"""
        seeds = []
        for comm in self.test_comms:
            seeds.append(comm[0])
        if num_seeds is not None:
            seeds = seeds[:num_seeds]
        return seeds