import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
from typing import List, Optional, Tuple, Set
from .expander import Expander
from .graph import Graph


class Refiner:
    def __init__(self, train_g: Graph, expander: Expander):
        self.expander = expander
        self.train_g = train_g
        self.clf = None  # 分类器（统一接口）
        self.scaler = None  # 标准化器
        self.classifier_type = None  # 记录使用的分类器类型

    # ---------- 内部辅助方法（不变） ----------
    def _community_avg_embed(
        self, nodes: List[int], exclude_node: Optional[int] = None
    ) -> np.ndarray:
        embeds = self.train_g.nodesEmbed(nodes)
        if exclude_node is not None and exclude_node in nodes:
            idx = nodes.index(exclude_node)
            embeds = np.delete(embeds, idx, axis=0)
        return (
            np.mean(embeds, axis=0)
            if len(embeds) > 0
            else np.zeros(self.train_g.embedsize)
        )

    def _node_feature(
        self, node: int, community_nodes: List[int], exclude_node: bool = False
    ) -> np.ndarray:
        node_emb = self.train_g.singleNodeEmbed(node)
        if exclude_node:
            avg_emb = self._community_avg_embed(community_nodes, exclude_node=node)
        else:
            avg_emb = self._community_avg_embed(community_nodes, exclude_node=None)
        return np.concatenate([node_emb, avg_emb])

    # ---------- 数据构建（不变） ----------
    def build_training_data(
        self, num_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_X = []
        all_y = []

        true_comms = list(self.train_g.communities.values())

        for _ in range(num_samples):
            seeds = [random.choice(com) for com in true_comms]
            pred_coms, _ = self.expander.sample_bs_trajectories(seeds=seeds)

            for pred, true_set in zip(pred_coms, true_comms):
                if not pred:
                    continue
                true_set = set(true_set)
                pred_set = set(pred)

                for node in pred:
                    feat = self._node_feature(node, pred, exclude_node=False)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

                neighbors = self.train_g.getNodesNeigh(pred)
                unique_neighbors = set(neighbors) - pred_set
                for node in unique_neighbors:
                    feat = self._node_feature(node, pred, exclude_node=True)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

        if not all_X:
            raise ValueError("没有生成任何训练样本，请检查数据或 expander 行为。")

        X = np.array(all_X)
        y = np.array(all_y)
        return X, y

    # ---------- 训练（支持多种分类器） ----------
    def trainRefiner(
        self,
        num_samples: int = 1,
        classifier_type: str = "xgb",
        classifier_params: Optional[dict] = None,
        **xgb_params,
    ):
        """
        训练 Refiner 分类器。

        :param num_samples: 生成训练数据时的重复采样次数
        :param classifier_type: 分类器类型，可选 'xgb', 'logistic', 'svm', 'rf'
        :param classifier_params: 传递给分类器的参数字典（会覆盖默认参数）
        :param xgb_params: 当 classifier_type='xgb' 时，可额外传入 XGBoost 参数（向后兼容）
        """
        print("构建 Refiner 训练数据...")
        X, y = self.build_training_data(num_samples=num_samples)
        print(f"样本数: {len(X)}, 正样本比例: {np.mean(y):.3f}")

        # 标准化（对逻辑回归、SVM 非常重要，对树模型也可选）
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 根据类型初始化分类器
        self.classifier_type = classifier_type
        print("refiner类型", self.classifier_type)
        # 默认参数
        if classifier_params is None:
            classifier_params = {}

        if classifier_type == "xgb":
            default_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False,
            }
            # 处理类别不平衡
            pos = np.sum(y)
            if 0 < pos < len(y):
                default_params["scale_pos_weight"] = (len(y) - pos) / pos
            # 合并用户参数（向后兼容 xgb_params）
            default_params.update(classifier_params)
            default_params.update(xgb_params)
            self.clf = xgb.XGBClassifier(**default_params)

        elif classifier_type == "logistic":
            default_params = {
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",  # 自动处理不平衡
            }
            default_params.update(classifier_params)
            self.clf = LogisticRegression(**default_params)

        elif classifier_type == "svm":
            # 使用概率输出，以便统一调用 predict_proba
            default_params = {
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced",
                "max_iter": 1000,  # 避免不收敛
            }
            default_params.update(classifier_params)
            self.clf = SVC(**default_params)

        elif classifier_type == "rf":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "class_weight": "balanced",
            }
            default_params.update(classifier_params)
            self.clf = RandomForestClassifier(**default_params)

        else:
            raise ValueError(
                f"不支持的分类器类型: {classifier_type}，可选 'xgb', 'logistic', 'svm', 'rf'"
            )

        # 训练
        self.clf.fit(X_scaled, y)

        # 简单评估（训练集）
        if hasattr(self.clf, "predict_proba"):
            y_pred_prob = self.clf.predict_proba(X_scaled)[:, 1]
            y_pred = (y_pred_prob >= 0.5).astype(int)
        else:
            # SVM 如果 probability=False 会没有 predict_proba，但我们已强制 True，这里只是兜底
            y_pred = self.clf.predict(X_scaled)

        from sklearn.metrics import classification_report

        print(f"\n【{classifier_type.upper()} 分类器 - 训练集报告】")
        print(
            classification_report(y, y_pred, target_names=["剔除/不加入", "保留/加入"])
        )

        print(f"Refiner ({classifier_type}) 训练完成")

    # ---------- 精炼（统一使用 predict_proba） ----------
    def refine_community(
        self, comm_nodes: List[int], threshold: float = 0.5
    ) -> List[int]:
        if self.clf is None or self.scaler is None:
            raise RuntimeError("Refiner 尚未训练，请先调用 trainRefiner() 方法。")

        if not comm_nodes:
            return []

        avg_emb = self._community_avg_embed(comm_nodes, exclude_node=None)

        keep_nodes = []
        for node in comm_nodes:
            node_emb = self.train_g.singleNodeEmbed(node)
            feat = np.concatenate([node_emb, avg_emb])
            X = self.scaler.transform([feat])
            # 统一使用 predict_proba 获得正类概率
            prob = self.clf.predict_proba(X)[0, 1]
            if prob >= threshold:
                keep_nodes.append(node)

        neighbors = self.train_g.getNodesNeigh(comm_nodes)
        candidate_nodes = set(neighbors) - set(comm_nodes)
        add_nodes = []
        for node in candidate_nodes:
            node_emb = self.train_g.singleNodeEmbed(node)
            feat = np.concatenate([node_emb, avg_emb])
            X = self.scaler.transform([feat])
            prob = self.clf.predict_proba(X)[0, 1]
            if prob >= threshold:
                add_nodes.append(node)

        return list(set(keep_nodes + add_nodes))
