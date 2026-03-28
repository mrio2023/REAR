import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import random
from .expander import Expander
from .graph import Graph
from .tool import pruning

class Refiner:
    def __init__(self, train_g: Graph, expander: Expander):
        self.expander = expander
        self.train_g = train_g
        self.clf = None          # 训练后的分类器
        self.scaler = None       # 标准化器

    def getTraVec(self, pred_com):
        """计算社区平均嵌入和节点嵌入列表"""
        embeds = self.train_g.nodesEmbed(pred_com)   # shape: (len(pred_com), embed_dim)
        avg_emb = np.mean(embeds, axis=0)
        return avg_emb, embeds

    def constructdata(self):
        """生成训练数据：粗糙社区、真实社区、边界节点"""
        true_comms = list(self.train_g.communities.values())
        seeds = []
        for com in true_comms:
            seed = random.choice(com)
            seeds.append(seed)

        # 1. 使用 expander 生成粗糙社区
        pred_coms, _ = self.expander.sample_bs_trajectories(seeds=seeds)

        # 2. 为每个粗糙社区提取边界节点（并剪枝）
        neighs = []
        for com in pred_coms:
            # 获取所有邻居（去重）
            neigh_single = self.train_g.getNodesNeigh(com)
            unique_neigh = list(set(neigh_single) - set(com))
            if len(unique_neigh) == 0:
                neighs.append([])
                continue

            # 计算社区平均嵌入和邻居节点嵌入
            avg_emb, com_embeds = self.getTraVec(com)
            # 提取邻居节点的嵌入（注意：getNodesNeigh 返回节点ID列表，需要对应嵌入）
            neigh_embeds = self.train_g.nodesEmbed(unique_neigh)   # shape: (len(unique_neigh), embed_dim)

            # 剪枝（保留与社区最相似的节点）
            # 注意：pruning 函数需要传入社区平均嵌入和邻居节点嵌入列表（或矩阵）
            # 假设 pruning 函数定义在 tool 中，返回剪枝后的节点列表（按相似度排序）
            pruned_nodes = pruning(
                community_pooled_embed=avg_emb,
                neigh_node_embed_list=neigh_embeds,
                neigh_nodes=unique_neigh,
            )
            neighs.append(pruned_nodes)

        return true_comms, pred_coms, neighs

    def trainRefiner(self):
        """
        使用构造的数据训练 XGBoost 分类器（仅保留/剔除，暂不处理加入）
        """
        print("开始构造训练数据...")
        true_comms, pred_coms, neighs = self.constructdata()

        X_keep = []
        y_keep = []

        # 为每个粗糙社区内的节点生成样本
        for pred, true_set, _ in zip(pred_coms, true_comms, neighs):
            if not pred:
                continue
            true_set = set(true_set)
            for node in pred:
                # 特征：节点嵌入 + 社区平均嵌入 + 结构特征（简单用邻居比例等）
                # 为了简单，我们先复用之前实验中的特征提取（但这里可以简化，只用嵌入）
                # 这里我们使用节点嵌入和社区平均嵌入的差值作为特征（更简单）
                node_emb = self.train_g.singleNodeEmbed(node)
                avg_emb, _ = self.getTraVec(pred)   # 重新计算当前社区的平均嵌入（因为 pred 就是当前粗糙社区）
                feat = np.concatenate([node_emb, avg_emb])   # 拼接两个嵌入
                X_keep.append(feat)
                y_keep.append(1 if node in true_set else 0)

        if len(X_keep) == 0:
            print("没有样本，无法训练")
            return

        X_keep = np.array(X_keep)
        y_keep = np.array(y_keep)

        print(f"样本数: {len(X_keep)}, 正样本比例: {np.mean(y_keep):.3f}")

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_keep)

        # 训练 XGBoost
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=(len(y_keep)-np.sum(y_keep))/np.sum(y_keep),
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        clf.fit(X_scaled, y_keep)

        # 保存到实例属性，以便后续预测使用
        self.clf = clf
        self.scaler = scaler

        # 打印训练集上的简单评估
        y_pred = clf.predict(X_scaled)
        from sklearn.metrics import classification_report
        print("\n【训练集上分类报告】")
        print(classification_report(y_keep, y_pred, target_names=['剔除', '保留']))

        print("Refiner 训练完成（未保存）")