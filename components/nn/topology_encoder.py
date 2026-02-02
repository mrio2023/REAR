import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TopologyEncoder(nn.Module):
    """无监督拓扑特征提取 - 基于GAT的图卷积层
    适配Graph类的邻接矩阵/节点特征输入
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index):
        """
        前向传播：拓扑特征提取
        :param x: 节点特征矩阵 [node_num, in_dim]
        :param edge_index: 边索引 [2, edge_num]（PyG格式）
        :return: 拓扑嵌入 [node_num, out_dim]
        """
        # 第一层GAT + 激活 + Dropout
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        
        # 第二层GAT + 层归一化
        h = self.gat2(h, edge_index)
        h = self.layer_norm(h)
        return h

    @staticmethod
    def graph2pyg_input(graph):
        """
        将自定义Graph类转换为PyG输入格式
        :param graph: Graph实例（K-ego子图）
        :return: 
            x: 节点特征 [node_count, feat_dim]
            edge_index: 边索引 [2, edge_num]
        """
        # 1. 构建节点特征（度数+交易频率）
        feat_list = []
        for node in graph.ego_nodes:
            # 获取节点度数（无则为0）
            degree = graph.degree_tab[graph.degree_tab['node'] == node]['degree'].values
            degree = degree[0] if len(degree) > 0 else 0
            # 获取交易频率（无则为0）
            freq = graph.degree_tab[graph.degree_tab['node'] == node]['normalized_trade_freq'].values
            freq = freq[0] if len(freq) > 0 else 0
            feat_list.append([degree, freq])
        x = torch.tensor(feat_list, dtype=torch.float32).to(DEVICE)

        # 2. 构建边索引（从邻接矩阵转换）
        edge_index = []
        # 取第一个快照的邻接矩阵（静态拓扑），也可叠加所有快照
        adj_matrix = graph.adjacency_ma[0] if len(graph.adjacency_ma) > 0 else []
        for i in range(graph.node_count):
            for j in range(graph.node_count):
                if adj_matrix[i][j] > 0:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T.to(DEVICE) if edge_index else torch.empty((2,0), dtype=torch.long).to(DEVICE)

        return x, edge_index