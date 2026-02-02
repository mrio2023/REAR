import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalSupervisedLayer(nn.Module):
    """时序监督预测层 - 融合指数移动平均的时序信息
    核心：学习不同时间步的节点嵌入衰减，长时间无交互节点权重降低
    """
    def __init__(self, embed_dim, snapshot_num, decay_rate=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.snapshot_num = snapshot_num
        self.decay_rate = decay_rate  # 指数衰减率
        # 时序权重预测层
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embed_dim * snapshot_num, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 时序预测头（预测下一时间步的节点交互）
        self.predict_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, contrast_embed, graph):
        """
        前向传播：时序嵌入+预测损失
        :param contrast_embed: 对比学习后的嵌入 [node_num, embed_dim]
        :param graph: Graph实例（含多快照邻接矩阵）
        :return:
            final_embed: 最终融合嵌入 [node_num, embed_dim]
            temporal_loss: 时序预测损失
        """
        # 1. 生成多快照的时序嵌入（带指数衰减）
        snapshot_embeds = []
        for t in range(self.snapshot_num):
            # 衰减权重：越晚的快照权重越高（decay_rate^t）
            decay_weight = self.decay_rate ** (self.snapshot_num - t - 1)
            
            # 基于t时刻邻接矩阵的节点嵌入加权
            adj_t = torch.tensor(graph.adjacency_ma[t], dtype=torch.float32).to(DEVICE) if t < len(graph.adjacency_ma) else torch.zeros(graph.node_count, graph.node_count).to(DEVICE)
            # 邻接矩阵归一化
            adj_t = F.normalize(adj_t + torch.eye(graph.node_count).to(DEVICE), p=1, dim=1)
            # 时序加权嵌入
            temporal_embed = torch.mm(adj_t, contrast_embed) * decay_weight
            snapshot_embeds.append(temporal_embed)

        # 2. 拼接多快照嵌入 + MLP融合
        concat_embed = torch.cat(snapshot_embeds, dim=1)  # [N, D*T]
        temporal_embed = self.temporal_mlp(concat_embed)  # [N, D]

        # 3. 计算时序预测损失（预测下一时间步的节点交互）
        temporal_loss = self._temporal_prediction_loss(temporal_embed, graph)

        # 4. 最终融合嵌入（对比嵌入 + 时序嵌入）
        final_embed = F.normalize(contrast_embed + temporal_embed, p=2, dim=1)
        return final_embed, temporal_loss

    def _temporal_prediction_loss(self, temporal_embed, graph):
        """
        时序预测损失：预测下一时间步的节点交互
        :param temporal_embed: 时序嵌入 [N, D]
        :param graph: Graph实例
        :return: 预测损失
        """
        if self.snapshot_num < 2:
            return torch.tensor(0.0).to(DEVICE)
        
        # 取最后两个快照的交互作为预测目标
        t_last = self.snapshot_num - 1
        t_prev = self.snapshot_num - 2
        
        # 构建t_last时刻的交互标签（邻接矩阵）
        adj_last = torch.tensor(graph.adjacency_ma[t_last], dtype=torch.float32).to(DEVICE) if t_last < len(graph.adjacency_ma) else torch.zeros(graph.node_count, graph.node_count).to(DEVICE)
        # 基于t_prev时刻嵌入预测t_last时刻交互
        pred_embed = self.predict_head(temporal_embed)
        pred_adj = torch.mm(pred_embed, pred_embed.t())
        # MSE损失
        loss = F.mse_loss(pred_adj, adj_last)
        return loss

    @staticmethod
    def get_snapshot_num(graph):
        """从Graph实例获取快照数量"""
        return graph.snapshot_num if hasattr(graph, 'snapshot_num') else len(graph.adjacency_ma)