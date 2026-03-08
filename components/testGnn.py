import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union


class Tool:
    def __init__(self):
        pass

    def pruning(
        self,
        community_pooled_embed: Union[np.ndarray, torch.Tensor],
        neigh_node_embed_list: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
        neigh_nodes: List[str],
        safe_min: int = 10,          # 绝对保底：至少保留10个（防止剪到太少）
        mid_threshold: int = 100,    # 中等邻居数阈值（100以内少剪）
        big_top_k: int = 50,         # 大数量邻居：最多保留50个
        small_p: float = 0.5,        # 小数量邻居（<100）：保留50%
        big_p: float = 0.1           # 大数量邻居（≥100）：保留10%
    ) -> List[str]:
        """
        分层剪枝策略（稳妥版）：
        1. 邻居数 ≤ safe_min（10）→ 不剪枝（绝对保底）
        2. safe_min < 邻居数 < mid_threshold（100）→ 保留50%（最少10个，最多50个）
        3. 邻居数 ≥ mid_threshold（100）→ 保留10% 或 50个（取更小，且≥10个）
        彻底避免31个节点剪到3个的极端情况！
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
                neigh_embeds = np.array([e.detach().cpu().numpy() for e in neigh_node_embed_list])
            else:
                neigh_embeds = np.array(neigh_node_embed_list)
        elif isinstance(neigh_node_embed_list, torch.Tensor):
            neigh_embeds = neigh_node_embed_list.detach().cpu().numpy()
        else:
            neigh_embeds = neigh_node_embed_list
        neigh_embeds = neigh_embeds.reshape(-1, pooled_embed.shape[1])  # (num_neigh, embed_dim)

        # ========== 2. 分层剪枝核心逻辑（稳妥兜底） ==========
        num_neigh = len(neigh_nodes)
        if num_neigh == 0:
            print("⚠️  无邻居节点，返回空列表")
            return []
        
        # 档1：≤10个 → 不剪枝（绝对保底）
        if num_neigh <= safe_min:
            print(f"ℹ️  邻居数={num_neigh} ≤ {safe_min}，不剪枝")
            return neigh_nodes
        
        # 档2：10 < 邻居数 < 100 → 保留50%（最少10个，最多50个）
        elif num_neigh < mid_threshold:
            keep_num = max(safe_min, int(num_neigh * small_p))  # 50%且≥10
            keep_num = min(keep_num, big_top_k)                 # 最多50个（避免100以内剪太多）
            print(f"ℹ️  邻居数={num_neigh}（中等），保留50% → {keep_num}个")
        
        # 档3：≥100个 → 保留10% 或 50个（取更小，且≥10）
        else:
            keep_num_by_p = int(num_neigh * big_p)
            keep_num = min(keep_num_by_p, big_top_k)            # 10%或50取更小
            keep_num = max(keep_num, safe_min)                  # 兜底≥10
            print(f"ℹ️  邻居数={num_neigh}（大量），10%={keep_num_by_p} → 最终保留{keep_num}个")

        # 最终兜底：不超过实际邻居数
        keep_num = min(keep_num, num_neigh)

        # ========== 3. 计算相似度并筛选Top邻居 ==========
        sim_scores = cosine_similarity(pooled_embed, neigh_embeds)[0]
        sorted_indices = np.argsort(sim_scores)[::-1]  # 降序排序
        top_indices = sorted_indices[:keep_num]
        pruned_neigh_nodes = [neigh_nodes[idx] for idx in top_indices]

        return pruned_neigh_nodes