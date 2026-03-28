import os
import random
import numpy as np
from collections import deque
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from dataProcess import DataLoader
from graph import Graph

# ---------------------------- 1. 加载数据 ---------------------------------
print("Loading data...")
loader = DataLoader(dataset_name="ibm_l_medium", train_ratio=0.8, min_com_size=3)
graph_data = loader.graph
train_comms = loader.train_comms
test_comms = loader.test_comms
all_nodes_set = set(loader.nodefeats.keys())

community_dict = {i: comm for i, comm in enumerate(train_comms + test_comms)}
graph = Graph(
    adj=graph_data['adj'],
    features=loader.nodefeats,
    communities=community_dict
)

# ---------------------------- 2. 1-Hop 粗糙社区构造 (保留+随机添加噪声) ---------------------------------
def make_rough_community_1hop(true_comm, all_nodes_set, noise_remove_ratio=0.2, noise_add_ratio=0.5):
    """
    1-hop 邻居 + 随机移除真实节点 + 随机添加外部节点
    """
    if len(true_comm) == 0:
        return []
    true_set = set(true_comm)
    
    # 随机移除一些真实节点
    n_remove = int(len(true_set) * noise_remove_ratio)
    n_remove = max(1, min(n_remove, len(true_set) - 1))  # 至少保留1个
    removed = random.sample(list(true_set), n_remove)
    rough = list(true_set - set(removed))
    
    # 随机添加外部节点（从1跳邻居中选择）
    # 获取所有1跳邻居（包括非社区节点）
    neighbors = set()
    for node in rough:
        neighbors.update(graph.getSingleNodeNeighbor(node))
    # 排除真实社区内的节点，只保留外部节点
    external = neighbors - true_set
    if external and noise_add_ratio > 0:
        n_add = int(len(true_set) * noise_add_ratio)
        n_add = min(n_add, len(external))
        added = random.sample(list(external), n_add)
        rough.extend(added)
    
    random.shuffle(rough)
    return rough

# ---------------------------- 3. 特征工程（增强版） ---------------------------------
def node_features(node, current_comm, graph):
    """
    返回包含结构特征和嵌入特征的向量
    """
    # 1. 结构特征
    neighbors = set(graph.getSingleNodeNeighbor(node))
    comm_set = set(current_comm)
    inter_neighbors = neighbors & comm_set
    inter_count = len(inter_neighbors)
    total_deg = len(neighbors) if neighbors else 1
    
    # 节点在社区内的邻居比例
    ratio_in_comm = inter_count / total_deg if total_deg > 0 else 0
    # 节点在社区内的邻居数量（原始）
    internal_deg = inter_count
    # 节点总度数
    degree = total_deg
    
    # 如果节点在社区内，额外特征：社区内邻居比例（同上），外部邻居比例
    if node in comm_set:
        internal_ratio = ratio_in_comm
        external_ratio = 1 - internal_ratio
    else:
        internal_ratio = ratio_in_comm
        external_ratio = 1 - internal_ratio
    
    # 2. 嵌入特征：节点自身嵌入，社区平均嵌入
    node_emb = graph.singleNodeEmbed(node)
    if len(current_comm) > 0:
        comm_emb = graph.nodesEmbed(current_comm).mean(axis=0)
    else:
        comm_emb = np.zeros_like(node_emb)
    
    # 计算嵌入差异
    diff = node_emb - comm_emb
    cos_sim = np.dot(node_emb, comm_emb) / (np.linalg.norm(node_emb) * np.linalg.norm(comm_emb) + 1e-8)
    euclidean_dist = np.linalg.norm(diff)
    
    # 3. 组合特征
    features = np.concatenate([
        [ratio_in_comm, internal_deg, degree, internal_ratio, external_ratio, cos_sim, euclidean_dist],
        node_emb,            # 原始节点嵌入
        comm_emb             # 社区平均嵌入
    ])
    return features.astype(np.float32)

# ---------------------------- 4. 构造训练样本（使用1-hop粗糙社区）---------------------------------
def build_training_data(train_comms, all_nodes_set, graph):
    X_keep = []
    y_keep = []
    X_add = []
    y_add = []

    for comm in train_comms:
        if len(comm) < 3:
            continue
        
        # 1-hop粗糙社区
        rough = make_rough_community_1hop(comm, all_nodes_set, noise_remove_ratio=0.2, noise_add_ratio=0.2)
        rough_set = set(rough)
        
        # 边界节点（粗糙社区的1-hop邻居中不在社区内的节点）
        boundary = set()
        for node in rough:
            for nb in graph.getSingleNodeNeighbor(node):
                if nb not in rough_set:
                    boundary.add(nb)
        
        true_set = set(comm)
        
        # 保留/剔除样本：粗糙社区内的每个节点
        for node in rough:
            feat = node_features(node, rough, graph)
            X_keep.append(feat)
            y_keep.append(1 if node in true_set else 0)
        
        # 加入/不加入样本：边界节点
        for node in boundary:
            feat = node_features(node, rough, graph)
            X_add.append(feat)
            y_add.append(1 if node in true_set else 0)
    
    return np.array(X_keep), np.array(y_keep), np.array(X_add), np.array(y_add)

# ---------------------------- 5. 训练分类器（支持多种模型）---------------------------------
def train_classifier(X, y, model_name='xgb'):
    if len(X) < 2:
        return None, None
    if len(np.unique(y)) < 2:
        print(f"Warning: 数据仅包含一类标签 {np.unique(y)}，跳过训练")
        return None, None
    
    # 标准化（某些模型需要，XGBoost不必须但可以提升稳定性）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_name == 'xgb':
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(len(y)-sum(y))/sum(y) if sum(y)>0 else 1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        clf.fit(X_scaled, y)
    elif model_name == 'rf':
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
        clf.fit(X_scaled, y)
    elif model_name == 'svm':
        clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
        clf.fit(X_scaled, y)
    else:
        return None, None
    
    return clf, scaler

# ---------------------------- 6. 社区精炼（使用保留模型，先只做剔除）---------------------------------
def refine_community(rough_comm, graph, clf_keep, scaler_keep, keep_threshold=0.5):
    """
    基于保留模型，移除概率低的节点
    """
    if clf_keep is None:
        return rough_comm
    
    current = list(rough_comm)
    # 计算每个节点的保留概率
    feats = np.array([node_features(node, current, graph) for node in current])
    feats_scaled = scaler_keep.transform(feats)
    probs = clf_keep.predict_proba(feats_scaled)[:, 1]  # 属于真实社区的概率
    
    # 保留概率大于阈值的节点
    new_comm = [node for node, prob in zip(current, probs) if prob > keep_threshold]
    if len(new_comm) == 0:
        return current  # 避免全删
    return new_comm

# ---------------------------- 7. 评估函数 ---------------------------------
def evaluate_communities(pred_comms, true_comms, graph):
    jaccards = []
    f1s = []
    for pred, true in zip(pred_comms, true_comms):
        if len(pred) == 0 and len(true) == 0:
            jaccards.append(1.0)
            f1s.append(1.0)
            continue
        pred_set = set(pred)
        true_set = set(true)
        inter = len(pred_set & true_set)
        union = len(pred_set | true_set)
        jacc = inter / union if union > 0 else 0.0
        prec = inter / len(pred_set) if len(pred_set) > 0 else 0.0
        rec = inter / len(true_set) if len(true_set) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        jaccards.append(jacc)
        f1s.append(f1)
    return np.mean(jaccards), np.mean(f1s)

# ---------------------------- 8. 主实验 ---------------------------------
def main():
    print("=== 构建训练数据 (1-Hop + 随机添加外部节点) ===")
    X_keep, y_keep, X_add, y_add = build_training_data(train_comms, all_nodes_set, graph)
    
    print(f"训练样本量: 保留/剔除 {len(X_keep)} 个, 加入/不加入 {len(X_add)} 个")
    print(f"特征维度: {X_keep.shape[1]}")
    print(f"Keep标签分布: 0(噪声)={np.sum(y_keep==0)}, 1(真实节点)={np.sum(y_keep==1)}")
    print(f"Add标签分布:  0(外部)={np.sum(y_add==0)}, 1(漏召回)={np.sum(y_add==1)}")
    
    print("\n=== 训练保留/剔除分类器 (XGBoost) ===")
    clf_keep, scaler_keep = train_classifier(X_keep, y_keep, model_name='xgb')
    
    # 可选：也训练加入分类器，但先不启用（样本不平衡）
    # clf_add, scaler_add = train_classifier(X_add, y_add, model_name='xgb')
    
    # 训练集评估
    if clf_keep is not None:
        X_keep_scaled = scaler_keep.transform(X_keep)
        y_pred_keep = clf_keep.predict(X_keep_scaled)
        print("\n【保留/剔除分类器 训练集效果】")
        print(classification_report(y_keep, y_pred_keep, target_names=['剔除', '保留']))
    
    print("\n=== 测试集评估 ===")
    # 生成测试集粗糙社区（1-hop，同样构造方式）
    rough_test = [make_rough_community_1hop(comm, all_nodes_set, noise_remove_ratio=0.2, noise_add_ratio=0.5) for comm in test_comms]
    
    # 精炼前效果
    jacc_before, f1_before = evaluate_communities(rough_test, test_comms, graph)
    print(f"精炼前 (1-Hop基线): 平均Jaccard = {jacc_before:.4f}, 平均F1 = {f1_before:.4f}")
    
    # 精炼：只剔除
    refined_test = []
    for rough in rough_test:
        refined = refine_community(rough, graph, clf_keep, scaler_keep, keep_threshold=0.5)
        refined_test.append(refined)
    
    jacc_after, f1_after = evaluate_communities(refined_test, test_comms, graph)
    print(f"精炼后 (纯剔除优化): 平均Jaccard = {jacc_after:.4f}, 平均F1 = {f1_after:.4f}")
    print(f"效果提升: Jaccard +{jacc_after - jacc_before:.4f}, F1 +{f1_after - f1_before:.4f}")
    
    # 可选：尝试不同的阈值，寻找最优
    print("\n=== 阈值调优 ===")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for th in thresholds:
        refined_test = [refine_community(rough, graph, clf_keep, scaler_keep, keep_threshold=th) for rough in rough_test]
        jacc, f1 = evaluate_communities(refined_test, test_comms, graph)
        print(f"阈值={th}: Jaccard={jacc:.4f}, F1={f1:.4f}")

if __name__ == "__main__":
    main()