import os
import random
import numpy as np


class DataLoader:
    def __init__(self, dataset_name, train_ratio=0.8, min_com_size=3, seed=2026):
        """
        Load global graph and split communities into train/test sets.

        :param dataset_name: Name of the dataset
        :param train_ratio: Proportion of communities for training (rest for test)
        :param min_com_size: Minimum community size to include
        :param seed: Random seed
        """
        self.dataset = dataset_name

        self.root = "df"

        self.train_ratio = train_ratio
        self.min_com_size = min_com_size
        random.seed(seed)

        # File paths
        self.ungraph_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.ungraph.txt"
        )
        self.cmty_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.cmty.txt"
        )
        self.feat_file = os.path.join(
            self.root, dataset_name, f"{dataset_name}-1.90.nodefeat.txt"
        )

        # Load global data
        self.graph = None
        self.comms = None
        self.nodefeats = None
        self._load_data()

        # Split communities
        self._split_comms()

    def _load_data(self, outlier_threshold=2):
        """Read edges, communities, features, and filter out outlier-sized communities."""
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

        # Filter communities by size (remove outliers)
        sizes = [len(c) for c in comms]
        if sizes:
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            lower_bound = self.min_com_size
            upper_bound = mean_size + outlier_threshold * std_size
            filtered_comms = [c for c in comms if lower_bound <= len(c) <= upper_bound]
            print(
                f"Original communities: {len(comms)}, after filtering: {len(filtered_comms)} "
                f"(removed {len(comms)-len(filtered_comms)} outliers, size range [{lower_bound}, {upper_bound:.1f}])"
            )
            comms = filtered_comms
        else:
            print("Warning: No communities satisfying minimum size condition.")

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

        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        self.graph = {"n": max(adj.keys(), default=-1) + 1, "edges": edges, "adj": adj}
        self.comms = comms
        self.nodefeats = nodefeats
        print(
            f"Loaded: nodes={self.graph['n']}, edges={len(edges)}, communities={len(comms)}, nodes_with_features={len(nodefeats)}"
        )

    def _split_comms(self):
        """Randomly split communities into training and test sets."""
        indices = list(range(len(self.comms)))
        random.shuffle(indices)
        split = int(len(indices) * self.train_ratio)
        train_idx = indices[:split]
        test_idx = indices[split:]

        self.train_comms = [self.comms[i] for i in train_idx]
        self.test_comms = [self.comms[i] for i in test_idx]