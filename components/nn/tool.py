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


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Optional


def pruning(
    community_pooled_embed: Union[np.ndarray, torch.Tensor],
    neigh_node_embed_list: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ],
    neigh_nodes: List[str],
    top_k_max: int = 50,  # 最大保留50个（与注释一致）
    top_p_ratio: float = 0.1,  # 最大保留10%
    min_neigh_threshold: int = 30,  # 邻居数≤30时不剪枝（与注释一致）
    debug: bool = False,  # 新增调试开关
) -> List[str]:
    """
    【静态函数】基于余弦相似度剪枝：
    1. 邻居数 ≤ min_neigh_threshold → 不剪枝（防止过少节点）
    2. 邻居数 > min_neigh_threshold → 自动选择 top_p_ratio 或 top_k_max 中更小的数量保留
    :param community_pooled_embed: 社区池化向量（1维/2维均可）
    :param neigh_node_embed_list: 邻居节点嵌入（列表/数组，numpy/torch均可）
    :param neigh_nodes: 邻居节点ID列表（与嵌入一一对应）
    :param top_k_max: 硬上限（最多保留多少个）
    :param top_p_ratio: 比例上限（最多保留百分之多少）
    :param min_neigh_threshold: 不剪枝的最小阈值（邻居数≤此值时不剪枝）
    :param debug: 是否打印调试信息
    :return: 剪枝后的邻居节点ID列表
    """
    # ========== 1. 输入格式统一（兼容numpy/torch） ==========
    if isinstance(community_pooled_embed, torch.Tensor):
        pooled_embed = community_pooled_embed.detach().cpu().numpy()
    else:
        pooled_embed = community_pooled_embed
    pooled_embed = pooled_embed.reshape(1, -1)  # (1, embed_dim)

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
    neigh_embeds = neigh_embeds.reshape(-1, pooled_embed.shape[1])

    num_neigh = len(neigh_nodes)
    if num_neigh == 0:
        if debug:
            print("[Pruning] 无邻居节点，返回空列表")
        return []

    # ========== 2. 邻居数 ≤ 阈值时不剪枝 ==========
    if num_neigh <= min_neigh_threshold:
        if debug:
            print(f"[Pruning] 邻居数={num_neigh} ≤ {min_neigh_threshold}，不剪枝")
        return neigh_nodes

    # ========== 3. 计算保留数量 ==========
    keep_num_by_p = max(1, int(num_neigh * top_p_ratio))  # 按比例至少1个
    keep_num = min(keep_num_by_p, top_k_max)  # 取比例上限和硬上限的较小值
    keep_num = max(
        min_neigh_threshold, keep_num
    )  # 确保不低于阈值（但此时num_neigh已>阈值，所以至少保留阈值个）
    # 注意：如果 keep_num 超过 num_neigh，取 num_neigh（但逻辑上不会，因为 keep_num_by_p ≤ num_neigh）
    keep_num = min(keep_num, num_neigh)

    # ========== 4. 计算余弦相似度并排序 ==========
    sim_scores = cosine_similarity(pooled_embed, neigh_embeds)[0]
    sorted_indices = np.argsort(sim_scores)[::-1]
    top_indices = sorted_indices[:keep_num]
    pruned_neigh_nodes = [neigh_nodes[idx] for idx in top_indices]

    # ========== 5. 调试输出 ==========
    if debug:
        min_sim = sim_scores[top_indices[-1]] if keep_num > 0 else 0.0
        print(
            f"[Pruning] 原始={num_neigh}, 保留={keep_num}, "
            f"比例={keep_num/num_neigh:.2f}, 最小相似度={min_sim:.4f}, "
            f"阈值设置: top_k_max={top_k_max}, top_p_ratio={top_p_ratio}, min_thresh={min_neigh_threshold}"
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


def eval_f1(pred_comm: Union[List, Set], true_comm: Union[List, Set]) -> float:
    """
    【静态函数】单独计算F1分数（临时偏向Recall）
    :param pred_comm: 预测的社区节点列表/集合
    :param true_comm: 真实的社区节点列表/集合
    :return: 偏向Recall的F1分数（浮点型）
    """
    pred_set = set(pred_comm) if isinstance(pred_comm, list) else pred_comm
    true_set = set(true_comm) if isinstance(true_comm, list) else true_comm

    intersect = true_set & pred_set
    p = len(intersect) / len(pred_set) if pred_set else 0.0
    r = len(intersect) / len(true_set) if true_set else 0.0

    # ========== 核心修改：给Recall加权重 ==========
    r_weight = 80  # Recall权重（可调：1.2-2.0，越大越偏向Recall）
    weighted_r = r * r_weight
    # ========== 替代原有F1计算 ==========

    if (p + weighted_r) <= 0:
        return 0.0
    return 2 * p * weighted_r / (p + weighted_r)  # 用加权后的Recall计算F1
