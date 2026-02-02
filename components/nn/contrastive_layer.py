import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SemiSupervisedContrastiveLayer(nn.Module):
    """半监督标签对比层 - 利用有标签/无标签节点的对比损失
    核心：同类节点拉近，异类节点推远，无标签节点利用拓扑相似性
    """
    def __init__(self, embed_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 归一化层
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)

    def forward(self, topology_embed, node_labels):
        """
        前向传播：对比损失计算
        :param topology_embed: 拓扑嵌入 [node_num, embed_dim]
        :param node_labels: 节点标签（-1=无标签，0=合法，1=非法）[node_num]
        :return: 
            contrast_embed: 对比学习后的嵌入 [node_num, embed_dim]
            loss: 对比损失值
        """
        # 1. 投影+归一化
        contrast_embed = self.projection(topology_embed)
        contrast_embed = self.l2_norm(contrast_embed)

        # 2. 计算对比损失
        loss = self._semi_supervised_contrastive_loss(contrast_embed, node_labels)
        return contrast_embed, loss

    def _semi_supervised_contrastive_loss(self, embed, labels):
        """
        半监督对比损失计算
        :param embed: 归一化后的嵌入 [N, D]
        :param labels: 标签 [-1,0,1] [N]
        :return: 损失值
        """
        N = embed.shape[0]
        # 计算相似度矩阵 [N, N]
        sim_matrix = torch.mm(embed, embed.t()) / self.temperature

        # 掩码：排除自身对比
        mask_self = torch.eye(N, dtype=torch.bool).to(DEVICE)
        sim_matrix = sim_matrix.masked_fill(mask_self, -1e9)

        # 1. 有标签节点损失
        labeled_mask = (labels != -1)
        if labeled_mask.sum() == 0:
            labeled_loss = torch.tensor(0.0).to(DEVICE)
        else:
            labeled_embed = embed[labeled_mask]
            labeled_labels = labels[labeled_mask]
            # 同类掩码
            pos_mask = (labeled_labels.unsqueeze(0) == labeled_labels.unsqueeze(1))
            pos_mask = pos_mask.masked_fill(torch.eye(len(labeled_labels), dtype=torch.bool).to(DEVICE), False)
            # 异类掩码
            neg_mask = ~pos_mask

            # 计算有标签对比损失
            pos_sim = sim_matrix[labeled_mask][:, labeled_mask][pos_mask].reshape(len(labeled_embed), -1)
            neg_sim = sim_matrix[labeled_mask][:, labeled_mask][neg_mask].reshape(len(labeled_embed), -1)
            
            if pos_sim.numel() == 0:
                labeled_loss = torch.tensor(0.0).to(DEVICE)
            else:
                pos_exp = torch.exp(pos_sim)
                neg_exp = torch.exp(neg_sim)
                labeled_loss = -torch.log(pos_exp.sum(dim=1) / (pos_exp.sum(dim=1) + neg_exp.sum(dim=1)))
                labeled_loss = labeled_loss.mean()

        # 2. 无标签节点损失（基于拓扑相似度）
        unlabeled_mask = (labels == -1)
        if unlabeled_mask.sum() == 0:
            unlabeled_loss = torch.tensor(0.0).to(DEVICE)
        else:
            unlabeled_embed = embed[unlabeled_mask]
            # 拓扑相似度掩码（前20%为正例）
            unlabeled_sim = sim_matrix[unlabeled_mask][:, unlabeled_mask]
            # 排除自身
            unlabeled_sim = unlabeled_sim.masked_fill(torch.eye(len(unlabeled_embed), dtype=torch.bool).to(DEVICE), -1e9)
            # 取每个节点的topk相似节点作为正例
            topk = max(1, int(len(unlabeled_embed) * 0.2))
            topk_indices = torch.topk(unlabeled_sim, k=topk, dim=1).indices
            
            # 构建无标签正例掩码
            unlabeled_pos_mask = torch.zeros_like(unlabeled_sim, dtype=torch.bool)
            for i in range(len(unlabeled_embed)):
                unlabeled_pos_mask[i, topk_indices[i]] = True

            # 计算无标签对比损失
            pos_sim = unlabeled_sim[unlabeled_pos_mask].reshape(len(unlabeled_embed), -1)
            neg_sim = unlabeled_sim[~unlabeled_pos_mask].reshape(len(unlabeled_embed), -1)
            
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)
            unlabeled_loss = -torch.log(pos_exp.sum(dim=1) / (pos_exp.sum(dim=1) + neg_exp.sum(dim=1)))
            unlabeled_loss = unlabeled_loss.mean()

        # 总损失：加权融合
        total_loss = 0.7 * labeled_loss + 0.3 * unlabeled_loss if labeled_mask.sum() > 0 and unlabeled_mask.sum() > 0 else labeled_loss + unlabeled_loss
        return total_loss

    @staticmethod
    def get_node_labels(graph, node_label_path):
        """
        从节点标签文件获取节点标签（适配Graph类）
        :param graph: Graph实例
        :param node_label_path: 节点标签文件路径（address, label）
        :return: 节点标签张量 [-1=无标签, 0=合法, 1=非法]
        """
        # 读取标签文件
        label_df = pd.read_csv(node_label_path)
        label_dict = dict(zip(label_df['address'], label_df['label']))
        
        # 为ego节点分配标签
        labels = []
        for node in graph.ego_nodes:
            if node in label_dict:
                labels.append(1 if label_dict[node] == 'illegal' else 0)
            else:
                labels.append(-1)
        return torch.tensor(labels, dtype=torch.long).to(DEVICE)