import torch
from torch import nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def make_linear_block(in_dim, out_dim, activation=Swish, norm_type=None):
    layers = [nn.Linear(in_dim, out_dim)]
    if norm_type == "batch":
        layers.append(nn.BatchNorm1d(out_dim))
    elif norm_type == "layer":
        layers.append(nn.LayerNorm(out_dim))
    if activation is not None:
        layers.append(activation())
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, hidden_size, input_size, norm_type=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.seed_embedding = nn.Linear(input_size, hidden_size, bias=True)
        self.node_embedding = nn.Linear(input_size, hidden_size, bias=True)

        self.input_mapping = nn.Sequential(
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
            make_linear_block(hidden_size, hidden_size, Swish, norm_type),
        )

        self.node_score_layer = nn.Linear(hidden_size, 1, bias=True)
        self.stopping_score_layer = nn.Linear(hidden_size, 2, bias=True)

        # 非零初始化参数
        nn.init.xavier_uniform_(self.node_score_layer.weight.data, gain=1.0)
        nn.init.xavier_uniform_(self.stopping_score_layer.weight.data, gain=1.0)
        if self.node_score_layer.bias is not None:
            nn.init.constant_(self.node_score_layer.bias.data, 0.01)
        if self.stopping_score_layer.bias is not None:
            nn.init.constant_(self.stopping_score_layer.bias.data, 0.01)

    def forward(self, x_seeds, x_nodes, indptr):
        # 检查输入
        if torch.isnan(x_seeds).any() or torch.isinf(x_seeds).any():
            raise ValueError(
                f"[DEBUG] NaN/Inf in x_seeds! shape={x_seeds.shape}, min={x_seeds.min()}, max={x_seeds.max()}"
            )
        if torch.isnan(x_nodes).any() or torch.isinf(x_nodes).any():
            raise ValueError(
                f"[DEBUG] NaN/Inf in x_nodes! shape={x_nodes.shape}, min={x_nodes.min()}, max={x_nodes.max()}"
            )

        h_seed = self.seed_embedding(x_seeds)
        h_node = self.node_embedding(x_nodes)
        h = h_seed + h_node

        h = self.input_mapping(h)

        node_scores = self.node_score_layer(h).squeeze(1)

        batch_logits = []
        for idx, (startpoint, endpoint, candidate_endpoint) in enumerate(indptr):
            if startpoint == endpoint:
                raise ValueError("Finished Episode!")

            candiLen = candidate_endpoint - startpoint

            # 停止节点特征计算
            stop_node = h[startpoint:endpoint].sum(dim=0, keepdim=True)
            stop_node = stop_node / (endpoint - startpoint)

            # 计算各类logits
            node_logits = node_scores[startpoint:candidate_endpoint]
            stopping_logits = self.stopping_score_layer(stop_node).squeeze(0)

            # 统一维度并拼接
            stop_action_logit = stopping_logits[1:].squeeze()
            action_logits = torch.cat(
                [node_logits + stopping_logits[0], stop_action_logit.unsqueeze(0)],
                dim=0,
            )

            batch_logits.append(action_logits)

        return batch_logits
