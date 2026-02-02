import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiFeatGConvLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim  # 输入特征维度：
        self.hidden_dim = hidden_dim
        
        # 图卷积层：多特征加权聚合（用1x1卷积实现逐特征权重学习）
        self.gconv = nn.Conv1d(input_dim, input_dim, kernel_size=1, groups=input_dim)  # 分组卷积=多特征独立权重
        # LSTM层：处理时序依赖
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 输出层：映射到异常评分特征
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, adj_matrix, feat_matrix, seq_len=22):
        """
        adj_matrix: 邻接矩阵 (batch_size, node_num, node_num)
        feat_matrix: 节点特征矩阵 (batch_size, node_num, input_dim)  # 每个节点的多特征向量
        seq_len: 快照数量（你的场景是22）
        """
        batch_size, node_num, _ = feat_matrix.shape
        hidden_states = []
        
        # 初始化LSTM隐藏状态
        h0 = torch.zeros(1, batch_size * node_num, self.hidden_dim).to(feat_matrix.device)
        c0 = torch.zeros(1, batch_size * node_num, self.hidden_dim).to(feat_matrix.device)
        
        for t in range(seq_len):
            # 1. 图卷积聚合：多特征加权
            # adj_matrix[t]: 第t个快照的邻接矩阵 (node_num, node_num)
            # 邻居特征聚合：A × X（矩阵乘法，多特征同时聚合）
            neighbor_feat = torch.matmul(adj_matrix[:, t, :, :], feat_matrix)  # (batch_size, node_num, input_dim)
            # 多特征独立加权（通过分组卷积学习每个特征的权重）
            weighted_feat = self.gconv(neighbor_feat.permute(0, 2, 1)).permute(0, 2, 1)  # (batch_size, node_num, input_dim)
            # 自身特征 + 邻居聚合特征
            gconv_out = F.relu(feat_matrix + weighted_feat)  # (batch_size, node_num, input_dim)
            
            # 2. LSTM处理时序依赖
            # reshape为(batch_size*node_num, input_dim)，适配LSTM输入
            lstm_in = gconv_out.reshape(batch_size * node_num, -1).unsqueeze(1)  # (batch_size*node_num, 1, input_dim)
            lstm_out, (h0, c0) = self.lstm(lstm_in, (h0, c0))  # (batch_size*node_num, 1, hidden_dim)
            
            # 3. 记录隐藏状态（用于后续异常评分）
            hidden_states.append(lstm_out.squeeze(1).reshape(batch_size, node_num, self.hidden_dim))
        
        # 4. 输出每个节点的时序特征嵌入
        final_feat = torch.stack(hidden_states, dim=1)  # (batch_size, seq_len, node_num, hidden_dim)
        anomaly_score = self.fc(final_feat).squeeze(-1)  # (batch_size, seq_len, node_num)：每个节点在每个快照的异常评分
        return final_feat, anomaly_score