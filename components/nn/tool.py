import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Set, Tuple
import random
import os
def set_seed(seed: int):
        """固定所有随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"✅ 所有随机种子已固定为：{seed}")
def pruning(
    community_pooled_embed: Union[np.ndarray, torch.Tensor],
    neigh_node_embed_list: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ],
    neigh_nodes: List[str],
    top_k_max: int = 50,  # 最大保留50个
    top_p_ratio: float = 0.1,  # 最大保留10%
    min_neigh_threshold: int = 30,  # 邻居数≤30时不剪枝
) -> List[str]:
    """
    【静态函数】基于余弦相似度剪枝：
    1. 邻居数≤30 → 不剪枝（防止空点/过少节点）
    2. 邻居数>30 → 自动选择 10% 或 50个 中更小的数量保留
    :param community_pooled_embed: 社区池化向量（1维/2维均可）
    :param neigh_node_embed_list: 邻居节点嵌入（列表/数组，numpy/torch均可）
    :param neigh_nodes: 邻居节点ID列表（与嵌入一一对应）
    :param top_k_max: 硬上限（最多保留50个）
    :param top_p_ratio: 比例上限（最多保留10%）
    :param min_neigh_threshold: 不剪枝的最小阈值（≤30不剪枝）
    :return: 剪枝后的邻居节点ID列表
    """
    # ========== 1. 输入格式统一（兼容numpy/torch） ==========
    # 社区池化向量转numpy并统一为2维
    if isinstance(community_pooled_embed, torch.Tensor):
        pooled_embed = community_pooled_embed.detach().cpu().numpy()
    else:
        pooled_embed = community_pooled_embed
    pooled_embed = pooled_embed.reshape(1, -1)  # (1, embed_dim)

    # 邻居嵌入转numpy并统一为2维
    if isinstance(neigh_node_embed_list, list):
        if isinstance(neigh_node_embed_list[0], torch.Tensor):
            neigh_embeds = np.array(
                [e.detach().cpu().numpy() for e in neigh_node_embed_list]
            )
        else:
            neigh_embeds = np.array(neigh_node_embed_list)
    elif isinstance(neigh_node_embed_list, torch.Tensor):
        neigh_embeds = neigh_node_embed_list.detach().cpu().numpy()
    else:
        neigh_embeds = neigh_node_embed_list
    neigh_embeds = neigh_embeds.reshape(
        -1, pooled_embed.shape[1]
    )  # (num_neigh, embed_dim)

    # ========== 2. 核心规则：邻居数≤30时直接返回全部（不剪枝） ==========
    num_neigh = len(neigh_nodes)
    if num_neigh == 0:
        print("⚠️  无邻居节点，返回空列表")
        return []
    if num_neigh <= min_neigh_threshold:
        print(f"ℹ️  邻居数={num_neigh} ≤ {min_neigh_threshold}，不剪枝")
        return neigh_nodes

    # ========== 3. 邻居数>30时，计算剪枝数量（10%或50取更小） ==========
    keep_num_by_p = max(1, int(num_neigh * top_p_ratio))  # 10%的数量（至少1个）
    keep_num = min(keep_num_by_p, top_k_max)  # 选10%和50中更小的
    keep_num = max(
        min_neigh_threshold, keep_num
    )  # 兜底：至少保留30个（防止剪到<30）

    # ========== 4. 计算余弦相似度并排序 ==========
    sim_scores = cosine_similarity(pooled_embed, neigh_embeds)[
        0
    ]  # 每个邻居的相似度
    sorted_indices = np.argsort(sim_scores)[::-1]  # 降序排序

    # ========== 5. 筛选Top邻居 ==========
    top_indices = sorted_indices[:keep_num]
    pruned_neigh_nodes = [neigh_nodes[idx] for idx in top_indices]

    # 剪枝统计（便于调试）
    print(
        f"📌 剪枝统计：原始={num_neigh} | 10%={keep_num_by_p} | 最终保留={keep_num}"
    )

    return pruned_neigh_nodes


def eval_scores(
    pred_comm: Union[List, Set], true_comm: Union[List, Set]
) -> Tuple[float, float, float]:
    """
    【静态函数】计算精确率(P)、召回率(R)、F1分数
    :param pred_comm: 预测的社区节点列表/集合
    :param true_comm: 真实的社区节点列表/集合
    :return: (precision, recall, f1) 保留4位小数
    """
    pred_set = set(pred_comm) if isinstance(pred_comm, list) else pred_comm
    true_set = set(true_comm) if isinstance(true_comm, list) else true_comm
    
    intersect = true_set & pred_set
    p = len(intersect) / len(pred_set) if pred_set else 0.0
    r = len(intersect) / len(true_set) if true_set else 0.0
    f1 = 2 * p * r / (p + r + 1e-9)  # 加极小值防止除零
    return round(p, 4), round(r, 4), round(f1, 4)


def eval_f1(
    pred_comm: Union[List, Set], true_comm: Union[List, Set]
) -> float:
    """
    【静态函数】单独计算F1分数（奖励函数核心）
    :param pred_comm: 预测的社区节点列表/集合
    :param true_comm: 真实的社区节点列表/集合
    :return: F1分数（浮点型）
    """
    pred_set = set(pred_comm) if isinstance(pred_comm, list) else pred_comm
    true_set = set(true_comm) if isinstance(true_comm, list) else true_comm
    
    intersect = true_set & pred_set
    p = len(intersect) / len(pred_set) if pred_set else 0.0
    r = len(intersect) / len(true_set) if true_set else 0.0
    
    if (p + r) <= 0:
        return 0.0
    return 2 * p * r / (p + r)


# ------------------- 测试用例（可选） -------------------
if __name__ == "__main__":
    # 测试pruning函数
    pooled_embed = np.random.rand(128)
    neigh_embeds = np.random.rand(100, 128)
    neigh_nodes = [f"node_{i}" for i in range(100)]
    pruned_nodes = pruning(pooled_embed, neigh_embeds, neigh_nodes)
    print(f"剪枝后节点数：{len(pruned_nodes)}")  # 预期输出10（10%）
    
    # 测试eval_scores和eval_f1
    pred = [1,2,3,4]
    true = [3,4,5,6]
    p, r, f1 = eval_scores(pred, true)
    f1_single = eval_f1(pred, true)
    print(f"P={p}, R={r}, F1={f1}")  # P=0.5, R=0.5, F1=0.5
    print(f"单独计算F1：{f1_single}")   # 0.5