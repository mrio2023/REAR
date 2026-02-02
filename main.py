import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, Dataset, Batch
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import components.config as Config


# 计算时序密度（TempASD核心指标：单位时间内节点的平均边数）
def calc_temporal_density(edge_count, node_count, snapshot_num):
    """
    edge_count: 子图总边数
    node_count: 子图节点数
    snapshot_num: 快照数量（时间窗口数）
    """
    return edge_count / (node_count * snapshot_num) if node_count * snapshot_num != 0 else 0.0

# 构建带快照的K-ego子图（每个快照包含对应时间区间的交易）
def build_temporal_k_ego_subgraph(central_node, adj_dict, time_snapshots, k=3):
    """
    central_node: 中心节点ID
    adj_dict: 带时间戳的邻接字典 {node: [(neighbor, timestamp, amount)]}
    time_snapshots: 时间快照区间列表 [(start1, end1), (start2, end2), ...]
    k: K阶邻居
    return: 子图节点列表、每个快照的邻接矩阵、节点交易频率权重、快照数
    """
    # 第一步：获取K-ego子图的所有节点（拓扑层面）
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
    snapshot_num = len(time_snapshots)
    
    # 第二步：为每个快照构建邻接矩阵（仅保留该时间区间的交易）
    snapshot_adjs = []  # [snapshot_num, node_count, node_count]
    # 统计每个节点的交易频率（用于GNN权重）
    node_trade_freq = np.zeros(node_count)
    
    for (t_start, t_end) in time_snapshots:
        adj = np.zeros((node_count, node_count))
        for i, u in enumerate(ego_nodes):
            # 遍历u的所有交易，筛选当前快照时间区间内的交易
            for (v, ts, amt) in adj_dict.get(u, []):
                if t_start <= ts <= t_end and v in node2idx:
                    j = node2idx[v]
                    adj[i, j] += 1  # 边权重=交易频率（TempASD核心：用交易频率替代简单连接）
                    node_trade_freq[i] += 1  # 节点交易频率累计
        snapshot_adjs.append(adj)
    
    # 归一化交易频率（作为节点初始权重）
    node_trade_freq = node_trade_freq / (np.max(node_trade_freq) + 1e-8)
    
    return ego_nodes, snapshot_adjs, node_trade_freq, node_count, snapshot_num

# 生成时间快照区间（将总时间窗口拆分为等间隔快照）
def generate_time_snapshots(total_start_ts, total_end_ts, interval=10):
    """
    total_start_ts: 总时间窗口起始时间戳
    total_end_ts: 总时间窗口结束时间戳
    interval: 快照间隔（分钟）
    return: 快照区间列表 [(start1, end1), ...]
    """
    interval_sec = interval * 60  # 转换为秒
    snapshots = []
    current_start = total_start_ts
    while current_start < total_end_ts:
        current_end = current_start + interval_sec
        if current_end > total_end_ts:
            current_end = total_end_ts
        snapshots.append((current_start, current_end))
        current_start = current_end
    return snapshots if snapshots else [(total_start_ts, total_end_ts)]

# ===================== 2. 时序GNN嵌入器（替换原多层嵌入，融入GConvLSTM） =====================
class TemporalGNNEmbedder(nn.Module):
    def __init__(self, in_dim, config):
        super(TemporalGNNEmbedder, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.snapshot_num = len(generate_time_snapshots(0, config.time_window*60, config.snapshot_interval))
        
        # 第一步：拓扑+交易频率嵌入（用SAGEConv，边权重=交易频率）
        self.sage_layer = SAGEConv(in_dim + 1, config.hidden_dim)  # +1：节点交易频率特征
        
        # 第二步：时序编码（TempASD用GConvLSTM，捕捉快照间的时序依赖）
        self.gconv_lstm = GConvLSTM(config.hidden_dim, config.hidden_dim, num_layers=1)
        
        # 第三步：融合时序密度的嵌入输出层
        self.fusion_layer = GCNConv(config.hidden_dim, config.embed_dim)
        
        # 时序预测头（预测下一个快照的交易分布，TempASD监督信号）
        self.time_pred_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 对比损失温度系数
        self.temperature = 0.07

    def forward(self, x, snapshot_adjs, node_freq, temporal_density):
        """
        x: 节点基础特征 [node_count, in_dim]
        snapshot_adjs: 快照邻接矩阵列表 [snapshot_num, node_count, node_count]
        node_freq: 节点交易频率 [node_count]
        temporal_density: 子图时序密度（标量）
        return: 最终嵌入、下快照预测
        """
        node_count = x.shape[0]
        
        # 融合节点基础特征+交易频率
        x = torch.cat([x, node_freq.unsqueeze(1)], dim=1)  # [node_count, in_dim+1]
        
        # 第一步：拓扑+交易频率嵌入（对每个快照的邻接矩阵计算）
        snapshot_embeds = []
        for adj in snapshot_adjs:
            # 转换邻接矩阵为PyG边索引格式
            edge_index = torch.nonzero(torch.tensor(adj, dtype=torch.float32)).t().contiguous()
            edge_weight = torch.tensor(adj[adj.nonzero()], dtype=torch.float32)
            
            # SAGEConv带边权重计算
            h = self.sage_layer(x, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
            snapshot_embeds.append(h)
        
        # 堆叠快照嵌入 [snapshot_num, node_count, hidden_dim]
        snapshot_embeds = torch.stack(snapshot_embeds, dim=0)
        
        # 第二步：GConvLSTM时序编码（捕捉快照间依赖）
        # GConvLSTM输入：[snapshot_num, node_count, hidden_dim]
        lstm_out, _ = self.gconv_lstm(snapshot_embeds)  # [snapshot_num, node_count, hidden_dim]
        # 取最后一个快照的输出作为时序聚合特征
        temporal_agg_embed = lstm_out[-1]  # [node_count, hidden_dim]
        
        # 第三步：融合时序密度（TempASD核心：时序密度作为重要特征）
        density_embed = temporal_density * torch.ones_like(temporal_agg_embed)  # [node_count, hidden_dim]
        fusion_embed = temporal_agg_embed + density_embed  # 简单加法融合，可替换为注意力融合
        
        # 最终嵌入（L2归一化）
        final_embed = self.fusion_layer(fusion_embed, edge_index)  # 用最后一个快照的边结构
        final_embed = F.normalize(final_embed, p=2, dim=-1)
        
        # 时序预测：预测下一个快照的嵌入（监督信号）
        next_snapshot_pred = self.time_pred_head(final_embed)
        
        return final_embed, next_snapshot_pred

    def contrastive_loss(self, embed, labels):
        """保持原对比损失逻辑（半监督标签利用）"""
        labeled_mask = labels != -1
        labeled_embed = embed[labeled_mask]
        labeled_labels = labels[labeled_mask]
        
        if len(labeled_embed) < 2:
            return torch.tensor(0.0, device=embed.device)
        
        sim_matrix = torch.mm(labeled_embed, labeled_embed.t()) / self.temperature
        sim_matrix = sim_matrix - torch.eye(len(sim_matrix), device=sim_matrix.device) * 1e9
        
        loss = 0.0
        for i in range(len(labeled_embed)):
            label = labeled_labels[i]
            pos_mask = (labeled_labels == label)
            pos_mask[i] = False
            neg_mask = ~pos_mask
            
            pos_sim = sim_matrix[i][pos_mask]
            neg_sim = sim_matrix[i][neg_mask]
            
            if len(pos_sim) == 0:
                continue
            
            pos_exp = torch.exp(pos_sim).sum()
            neg_exp = torch.exp(neg_sim).sum()
            loss += -torch.log(pos_exp / (pos_exp + neg_exp))
        
        return loss / len(labeled_embed)

    def time_pred_loss(self, pred, target):
        """时序预测损失（预测下一个快照的嵌入与真实嵌入的MSE）"""
        return F.mse_loss(pred, target)

# ===================== 3. 数据加载与预处理（适配时序快照） =====================
class TemporalFinancialDataset:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.adj_dict = {}  # 带时间戳的邻接字典 {node: [(neighbor, timestamp, amount)]}
        self.node_feat = {}  # 节点基础特征 {node_id: feat}
        self.node_label = {}  # 节点标签 {node_id: 0/1/-1}
        self.total_start_ts = None  # 全量数据起始时间戳
        self.total_end_ts = None  # 全量数据结束时间戳
        
        # 加载数据
        self.load_data()
        # 生成时间快照区间（全局）
        self.time_snapshots = generate_time_snapshots(
            self.total_start_ts, self.total_end_ts, config.snapshot_interval
        )
    
    def load_data(self):
        """加载带时间戳的金融交易数据，适配时序快照"""
        df = pd.read_csv(self.data_path)
        
        # 初始化时间范围
        self.total_start_ts = df['timestamp'].min()
        self.total_end_ts = df['timestamp'].max()
        
        for _, row in df.iterrows():
            src, tgt = row['source'], row['target']
            ts = row['timestamp']
            amt = row['amount']
            label = row.get('label', -1)
            
            # 构建带时间戳的邻接字典
            if src not in self.adj_dict:
                self.adj_dict[src] = []
            if tgt not in self.adj_dict:
                self.adj_dict[tgt] = []
            self.adj_dict[src].append((tgt, ts, amt))
            self.adj_dict[tgt].append((src, ts, amt))  # 无向图（交易双向）
            
            # 节点标签
            self.node_label[src] = label
            self.node_label[tgt] = self.node_label.get(tgt, -1)
            
            # 节点基础特征（金额统计+交易次数）
            src_feat = np.array([amt, len(self.adj_dict[src])])
            tgt_feat = np.array([amt, len(self.adj_dict[tgt])])
            self.node_feat[src] = src_feat
            self.node_feat[tgt] = tgt_feat
    
    def get_illegal_subgraphs(self):
        """获取非法节点的时序K-ego子图（带快照）"""
        illegal_nodes = [n for n, lbl in self.node_label.items() if lbl == 1]
        illegal_subgraphs = []
        
        for central_node in tqdm(illegal_nodes, desc="Building temporal illegal subgraphs"):
            # 构建带快照的K-ego子图
            ego_nodes, snapshot_adjs, node_freq, node_count, snapshot_num = build_temporal_k_ego_subgraph(
                central_node, self.adj_dict, self.time_snapshots, self.config.k_ego
            )
            
            # 计算子图时序密度（TempASD核心指标）
            total_edge_count = sum([adj.sum() for adj in snapshot_adjs])
            temporal_density = calc_temporal_density(total_edge_count, node_count, snapshot_num)
            
            # 构建特征矩阵、标签矩阵
            feat = np.zeros((node_count, len(self.node_feat[ego_nodes[0]])))
            labels = np.zeros(node_count)
            for i, node in enumerate(ego_nodes):
                feat[i] = self.node_feat.get(node, np.zeros_like(feat[0]))
                labels[i] = self.node_label.get(node, -1)
            
            # 转换为PyTorch张量
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            node_freq_tensor = torch.tensor(node_freq, dtype=torch.float32)
            temporal_density_tensor = torch.tensor(temporal_density, dtype=torch.float32)
            
            # 构建PyG数据对象（包含所有快照的邻接矩阵）
            data = Data(
                x=feat_tensor,
                snapshot_adjs=snapshot_adjs,  # 每个快照的邻接矩阵
                node_freq=node_freq_tensor,
                temporal_density=temporal_density_tensor,
                y=labels_tensor,
                node2idx={n: i for i, n in enumerate(ego_nodes)},
                ego_nodes=ego_nodes
            )
            illegal_subgraphs.append(data)
        
        return illegal_subgraphs

# ===================== 4. 模型训练（适配时序GNN嵌入器） =====================
def train_temporal_embedder(config, dataset):
    illegal_subgraphs = dataset.get_illegal_subgraphs()
    if not illegal_subgraphs:
        raise ValueError("No illegal subgraphs found!")
    
    # 初始化模型（in_dim=节点基础特征维度）
    in_dim = illegal_subgraphs[0].x.shape[1]
    model = TemporalGNNEmbedder(in_dim, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 训练循环
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        for subgraph in tqdm(illegal_subgraphs, desc=f"Epoch {epoch+1}/{config.epochs}"):
            subgraph = subgraph.to(device)
            optimizer.zero_grad()
            
            # 前向传播（输入快照邻接矩阵、交易频率、时序密度）
            embed, next_snap_pred = model(
                subgraph.x, subgraph.snapshot_adjs, subgraph.node_freq, subgraph.temporal_density
            )
            
            # 计算损失：对比损失（标签）+ 时序预测损失（快照）
            contrast_loss = model.contrastive_loss(embed, subgraph.y)
            # 时序预测目标：用最后一个快照的嵌入作为目标（简化，可替换为真实下快照）
            target_embed = embed  # 实际应用中需用真实下一个快照的嵌入
            time_loss = model.time_pred_loss(next_snap_pred, target_embed)
            loss = contrast_loss + time_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(illegal_subgraphs)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    return model

# ===================== 5. 聚类与重叠社区发现（保持原逻辑，适配新嵌入） =====================
def cluster_embeddings(embeds, config):
    if config.clustering_method == 'hierarchical':
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        clustering.fit(embeds)
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(embeds, method='ward')
        cluster_labels = fcluster(Z, t=config.dbscan_eps, criterion='distance')
    elif config.clustering_method == 'dbscan':
        clustering = DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples)
        cluster_labels = clustering.fit_predict(embeds)
    else:
        raise ValueError("Unsupported clustering method!")
    
    return cluster_labels

def find_overlapping_communities(embeds, cluster_labels, config):
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    cluster_centers = {}
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_centers[c] = embeds[mask].mean(axis=0)
    
    node_community_membership = {}
    for i in range(len(embeds)):
        node_embed = embeds[i].reshape(1, -1)
        membership = []
        for c, center in cluster_centers.items():
            center = center.reshape(1, -1)
            sim = cosine_similarity(node_embed, center)[0][0]
            if sim >= config.cosine_threshold:
                membership.append(c)
        node_community_membership[i] = membership
    
    return node_community_membership

# ===================== 6. 主流程（适配时序数据处理） =====================
def main(data_path):
    config = Config()
    
    # 加载时序金融数据
    print("Loading temporal dataset...")
    dataset = TemporalFinancialDataset(config, data_path)
    
    # 训练时序嵌入器
    print("Training temporal GNN embedder...")
    model = train_temporal_embedder(config, dataset)
    
    # 为所有节点生成嵌入
    print("Generating node embeddings...")
    all_nodes = list(dataset.node_feat.keys())
    all_embeds = []
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for node in tqdm(all_nodes, desc="Generating temporal embeds"):
            # 构建该节点的时序K-ego子图
            ego_nodes, snapshot_adjs, node_freq, node_count, snapshot_num = build_temporal_k_ego_subgraph(
                node, dataset.adj_dict, dataset.time_snapshots, config.k_ego
            )
            if node not in {n: i for i, n in enumerate(ego_nodes)}:
                all_embeds.append(np.zeros(config.embed_dim))
                continue
            
            # 构建子图特征
            feat = np.zeros((node_count, len(dataset.node_feat[ego_nodes[0]])))
            for i, n in enumerate(ego_nodes):
                feat[i] = dataset.node_feat.get(n, np.zeros_like(feat[0]))
            feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
            node_freq_tensor = torch.tensor(node_freq, dtype=torch.float32).to(device)
            
            # 计算时序密度
            total_edge_count = sum([adj.sum() for adj in snapshot_adjs])
            temporal_density = calc_temporal_density(total_edge_count, node_count, snapshot_num)
            temporal_density_tensor = torch.tensor(temporal_density, dtype=torch.float32).to(device)
            
            # 生成嵌入
            embed, _ = model(feat_tensor, snapshot_adjs, node_freq_tensor, temporal_density_tensor)
            
            # 获取当前节点的嵌入
            node_idx = {n: i for i, n in enumerate(ego_nodes)}[node]
            all_embeds.append(embed[node_idx].cpu().numpy())
    
    all_embeds = np.array(all_embeds)
    
    # 聚类与重叠社区发现
    print("Clustering embeddings...")
    cluster_labels = cluster_embeddings(all_embeds, config)
    
    print("Finding overlapping communities...")
    overlapping_membership = find_overlapping_communities(all_embeds, cluster_labels, config)
    
    # 输出结果
    print("\n=== Temporal Overlapping Community Results ===")
    for i, node in enumerate(all_nodes[:10]):
        communities = overlapping_membership[i]
        print(f"Node {node}: belongs to communities {communities}, temporal density {temporal_density:.2f}")
    
    return {
        'node_embeddings': dict(zip(all_nodes, all_embeds)),
        'cluster_labels': dict(zip(all_nodes, cluster_labels)),
        'overlapping_communities': dict(zip(all_nodes, overlapping_membership.values())),
        'temporal_density': {node: calc_temporal_density(
            sum([adj.sum() for adj in build_temporal_k_ego_subgraph(
                node, dataset.adj_dict, dataset.time_snapshots, config.k_ego)[1]]),
            len(build_temporal_k_ego_subgraph(
                node, dataset.adj_dict, dataset.time_snapshots, config.k_ego)[0]),
            len(dataset.time_snapshots)
        ) for node in all_nodes}
    }

# 运行主流程
if __name__ == "__main__":
    data_path = "transaction_data.csv"
    results = main(data_path)
    
    # 保存结果（包含时序密度）
    result_df = pd.DataFrame({
        'node_id': results['cluster_labels'].keys(),
        'cluster_label': results['cluster_labels'].values(),
        'overlapping_communities': results['overlapping_communities'].values(),
        'temporal_density': [results['temporal_density'][node] for node in results['cluster_labels'].keys()]
    })
    result_df.to_csv("temporal_community_results.csv", index=False)
    print("\nResults saved to temporal_community_results.csv")