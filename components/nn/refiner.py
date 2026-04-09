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
        self.clf = None
        self.scaler = None
        self.classifier_type = None

    # ---------- Helper methods ----------
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

    # ---------- Data construction ----------
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

                # Positive/negative samples from predicted community
                for node in pred:
                    feat = self._node_feature(node, pred, exclude_node=False)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

                # Negative samples from neighbours not in predicted community
                neighbors = self.train_g.getNodesNeigh(pred)
                unique_neighbors = set(neighbors) - pred_set
                for node in unique_neighbors:
                    feat = self._node_feature(node, pred, exclude_node=True)
                    all_X.append(feat)
                    all_y.append(1 if node in true_set else 0)

        if not all_X:
            raise ValueError("No training samples generated. Check data or expander behavior.")

        X = np.array(all_X)
        y = np.array(all_y)
        return X, y

    # ---------- Training (supports multiple classifiers) ----------
    def trainRefiner(
        self,
        num_samples: int = 1,
        classifier_type: str = "xgb",
        classifier_params: Optional[dict] = None,
        **xgb_params,
    ):
        """
        Train the Refiner classifier.

        Args:
            num_samples: Number of repeated sampling passes to generate data.
            classifier_type: Type of classifier ('xgb', 'logistic', 'svm', 'rf').
            classifier_params: Parameter dict (overrides defaults).
            xgb_params: Additional XGBoost parameters (backward compatibility).
        """
        print("Building Refiner training data...")
        X, y = self.build_training_data(num_samples=num_samples)
        print(f"Samples: {len(X)}, Positive ratio: {np.mean(y):.3f}")

        # Standardize features (important for linear models, optional for trees)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.classifier_type = classifier_type

        # Default parameters
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
            # Handle class imbalance
            pos = np.sum(y)
            if 0 < pos < len(y):
                default_params["scale_pos_weight"] = (len(y) - pos) / pos
            default_params.update(classifier_params)
            default_params.update(xgb_params)
            self.clf = xgb.XGBClassifier(**default_params)

        elif classifier_type == "logistic":
            default_params = {
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",
            }
            default_params.update(classifier_params)
            self.clf = LogisticRegression(**default_params)

        elif classifier_type == "svm":
            default_params = {
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced",
                "max_iter": 1000,
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
                f"Unsupported classifier type: {classifier_type}. "
                f"Choose from 'xgb', 'logistic', 'svm', 'rf'."
            )

        self.clf.fit(X_scaled, y)

        # Simple training accuracy evaluation
        if hasattr(self.clf, "predict_proba"):
            y_pred = (self.clf.predict_proba(X_scaled)[:, 1] >= 0.5).astype(int)
        else:
            y_pred = self.clf.predict(X_scaled)
        acc = np.mean(y_pred == y)
        print(f"{classifier_type.upper()} training accuracy: {acc:.4f}")
        print(f"Refiner ({classifier_type}) training completed.")

    # ---------- Refinement (unified predict_proba interface) ----------
    def refine_community(
        self, comm_nodes: List[int], threshold: float = 0.5
    ) -> List[int]:
        if self.clf is None or self.scaler is None:
            raise RuntimeError("Refiner not trained. Call trainRefiner() first.")

        if not comm_nodes:
            return []

        avg_emb = self._community_avg_embed(comm_nodes, exclude_node=None)

        keep_nodes = []
        for node in comm_nodes:
            node_emb = self.train_g.singleNodeEmbed(node)
            feat = np.concatenate([node_emb, avg_emb])
            X = self.scaler.transform([feat])
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