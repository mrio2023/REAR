import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import random
from typing import List, Optional, Tuple, Set
from .expander import Expander
from .graph import Graph

class Refiner:
    def __init__(self, train_g: Graph, expander: Expander):
        self.expander = expander
        self.train_g = train_g
        self.clf = None          # XGBoost 分类器
        self.scaler = None       # 标准化器

    # ---------- 内部辅助方法 ----------
    def _community_avg_embed(self, nodes: List[int], exclude_node: Optional[int] = None) -> np.ndarray:
        """
        计算节点集合的平均嵌入，可选择排除某个节点（用于边界节点特征）。
        """
        embeds = self.train_g.nodesEmbed(nodes)
        if exclude_node is not None and exclude_node in nodes:
            idx = nodes.index(exclude_node)
            embeds = np.delete(embeds, idx, axis=0)
        return np.mean(embeds, axis=0) if len(embeds) > 0 else np.zeros(self.train_g.embedsize)

    def _node_feature(self, node: int, community_nodes: List[int], exclude_node: bool = False) -> np.ndarray:
        """
        为节点构建特征向量：[节点嵌入, 社区平均嵌入]
        - 若 exclude_node = True，计算社区平均嵌入时排除该节点（用于边界节点）
        """
        node_emb = self.train_g.singleNodeEmbed(node)
        if exclude_node:
            avg_emb = self._community_avg_embed(community_nodes, exclude_node=node)
        else:
            avg_emb = self._community_avg_embed(community_nodes, exclude_node=None)
        return np.concatenate([node_emb, avg_emb])

    # ---------- 数据构建 ----------
    def build_training_data(self, num_samples: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建训练数据（内部节点 + 边界节点）。
        参数 num_samples: 随机采样多少次（每次重新生成粗糙社区），增加样本多样性。
        返回 (X, y) 两个 numpy 数组。
        """
        all_X = []
        all_y = []

        true_comms = list(self.train_g.communities.values())

        for _ in range(num_samples):
            # 随机选择种子，每个真实社区一个种子
            seeds = [random.choice(com) for com in true_comms]

            # 使用 expander 生成粗糙社区
            pred_coms, _ = self.expander.sample_bs_trajectories(seeds=seeds)

            for pred, true_set in zip(pred_coms, true_comms):
                if not pred:
                    continue
                true_set = set(true_set)
                pred_set = set(pred)

                # ----- 内部节点样本 -----
                for node in pred:
                    feat = self._node_feature(node, pred, exclude_node=False)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

                # ----- 边界节点样本 -----
                neighbors = self.train_g.getNodesNeigh(pred)
                unique_neighbors = set(neighbors) - pred_set
                for node in unique_neighbors:
                    # 边界节点特征：社区平均嵌入不包含该节点
                    feat = self._node_feature(node, pred, exclude_node=True)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

        if not all_X:
            raise ValueError("没有生成任何训练样本，请检查数据或 expander 行为。")

        X = np.array(all_X)
        y = np.array(all_y)
        return X, y

    # ---------- 训练 ----------
    def trainRefiner(self, num_samples: int = 1, **xgb_params):
        """
        训练 Refiner 分类器。
        :param num_samples: 生成训练数据时的重复采样次数
        :param xgb_params: XGBoost 参数字典，可覆盖默认值
        """
        print("构建 Refiner 训练数据...")
        X, y = self.build_training_data(num_samples=num_samples)
        print(f"样本数: {len(X)}, 正样本比例: {np.mean(y):.3f}")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost 默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        }
        # 处理类别不平衡
        pos = np.sum(y)
        if pos > 0 and pos < len(y):
            default_params['scale_pos_weight'] = (len(y) - pos) / pos
        default_params.update(xgb_params)

        self.clf = xgb.XGBClassifier(**default_params)
        self.clf.fit(X_scaled, y)

        # 简单评估
        y_pred = self.clf.predict(X_scaled)
        from sklearn.metrics import classification_report
        print("\n【训练集上分类报告】")
        print(classification_report(y, y_pred, target_names=["剔除/不加入", "保留/加入"]))

        print("Refiner 训练完成")

    # ---------- 精炼 ----------
    def refine_community(self, comm_nodes: List[int], threshold: float = 0.5) -> List[int]:
        """
        对单个社区进行精炼：剔除噪声节点，添加缺失节点。
        """
        if self.clf is None or self.scaler is None:
            raise RuntimeError("Refiner 尚未训练，请先调用 train() 方法。")

        if not comm_nodes:
            return []

        # 1. 社区平均嵌入（用于所有节点特征）
        avg_emb = self._community_avg_embed(comm_nodes, exclude_node=None)

        # 2. 内部节点：保留预测
        keep_nodes = []
        for node in comm_nodes:
            node_emb = self.train_g.singleNodeEmbed(node)
            feat = np.concatenate([node_emb, avg_emb])
            X = self.scaler.transform([feat])
            prob = self.clf.predict_proba(X)[0, 1]   # 正类概率
            if prob >= threshold:
                keep_nodes.append(node)

        # 3. 边界节点：添加预测
        neighbors = self.train_g.getNodesNeigh(comm_nodes)
        candidate_nodes = set(neighbors) - set(comm_nodes)
        add_nodes = []
        for node in candidate_nodes:
            node_emb = self.train_g.singleNodeEmbed(node)
            # 边界节点特征：社区平均嵌入不包含该节点（但实际它不在社区内，所以相同）
            feat = np.concatenate([node_emb, avg_emb])
            X = self.scaler.transform([feat])
            prob = self.clf.predict_proba(X)[0, 1]
            if prob >= threshold:
                add_nodes.append(node)

        return list(set(keep_nodes + add_nodes))

 