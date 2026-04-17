import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Set, Tuple
import random
import os
from typing import Dict, List, Tuple, Set, Any


def compute_metrics(
    true_coms: List[Tuple[str, List]], pred_coms: Dict[str, Set]
) -> Dict[str, float]:
    """
    Compute average precision, recall, F1, and Jaccard for a set of communities.

    Args:
        true_coms: List of (community_tag, list_of_nodes) for ground truth.
        pred_coms: Dictionary mapping community_tag -> set of predicted nodes.

    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'jaccard' containing averages.
    """
    METRIC_KEYS = ["precision", "recall", "f1", "jaccard"]
    metrics = {key: [] for key in METRIC_KEYS}
    for tag, true_nodes in true_coms:
        pred_nodes = list(pred_coms.get(tag, set()))
        compute_single_metrics(pred_nodes, true_nodes, metrics)
    return {key: safe_mean(metrics[key]) for key in METRIC_KEYS}


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    """Print a formatted metrics table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | Jaccard: {metrics['jaccard']:.4f}"
    )


def print_metrics_comparison(orig: Dict[str, float], refined: Dict[str, float]) -> None:

    METRIC_KEYS = ["precision", "recall", "f1", "jaccard"]
    """Print side-by-side comparison of original vs refined metrics."""
    print("\n" + "=" * 80)
    print("Metric Changes (Original → Refined)")
    print("=" * 80)
    for key in METRIC_KEYS:
        diff = refined[key] - orig[key]
        print(
            f"{key.capitalize():9}: {orig[key]:.4f} → {refined[key]:.4f} ({diff:+.4f})"
        )
    print("=" * 80)


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"All random seeds fixed to: {seed}")


def pruning(
    community_pooled_embed: Union[np.ndarray, torch.Tensor],
    neigh_node_embed_list: Union[
        List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor
    ],
    neigh_nodes: List[str],
    top_k_max: int = 50,
    top_p_ratio: float = 0.1,
    min_neigh_threshold: int = 30,
    debug: bool = False,
) -> List[str]:
    """
    Prune neighbors based on cosine similarity:
    1. If number of neighbors <= min_neigh_threshold -> no pruning.
    2. Otherwise keep min(top_p_ratio, top_k_max) neighbors with highest similarity.
    """
    if isinstance(community_pooled_embed, torch.Tensor):
        pooled_embed = community_pooled_embed.detach().cpu().numpy()
    else:
        pooled_embed = community_pooled_embed
    pooled_embed = pooled_embed.reshape(1, -1)

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
            print("[Pruning] No neighbors, returning empty list")
        return []

    if num_neigh <= min_neigh_threshold:
        if debug:
            print(
                f"[Pruning] Neighbors={num_neigh} <= {min_neigh_threshold}, no pruning"
            )
        return neigh_nodes

    keep_num_by_p = max(1, int(num_neigh * top_p_ratio))
    keep_num = min(keep_num_by_p, top_k_max)
    keep_num = max(min_neigh_threshold, keep_num)
    keep_num = min(keep_num, num_neigh)

    sim_scores = cosine_similarity(pooled_embed, neigh_embeds)[0]
    sorted_indices = np.argsort(sim_scores)[::-1]
    top_indices = sorted_indices[:keep_num]
    pruned_neigh_nodes = [neigh_nodes[idx] for idx in top_indices]

    return pruned_neigh_nodes


def eval_scores(
    pred_comm: Union[List, Set], true_comm: Union[List, Set]
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, F1 between predicted and true communities."""
    pred_set = set(pred_comm) if isinstance(pred_comm, list) else pred_comm
    true_set = set(true_comm) if isinstance(true_comm, list) else true_comm

    intersect = true_set & pred_set
    p = len(intersect) / len(pred_set) if pred_set else 0.0
    r = len(intersect) / len(true_set) if true_set else 0.0
    f1 = 2 * p * r / (p + r + 1e-9)
    return round(p, 4), round(r, 4), round(f1, 4)


def calculate_prfj(com1: List, com2: List) -> Tuple[float, float, float, float]:
    """Compute Precision, Recall, F1, and Jaccard for a single community pair."""
    p, r, f1 = eval_scores(com1, com2)

    set1, set2 = set(com1), set(com2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0.0

    return p, r, f1, jaccard


def safe_mean(values: List[float]) -> float:
    """Safely compute the mean of a list, returning 0.0 if empty."""
    return round(np.mean(values) if values else 0.0, 4)


def compute_single_metrics(
    pred_com: List, true_com: List, metrics_dict: Dict[str, List]
) -> None:
    """
    Compute metrics for a single prediction and append to the dictionary.
    Args:
        pred_com: Predicted community list
        true_com: Ground truth community list
        metrics_dict: Dictionary with keys 'precision', 'recall', 'f1', 'jaccard'
    """
    p, r, f1, j = calculate_prfj(pred_com, true_com)
    metrics_dict["precision"].append(p)
    metrics_dict["recall"].append(r)
    metrics_dict["f1"].append(f1)
    metrics_dict["jaccard"].append(j)


def aggregate_avg_metrics(
    metrics_before: Dict[str, List], metrics_after: Dict[str, List]
) -> Dict[str, float]:
    """
    Aggregate average metrics before and after refinement into a single dictionary.
    """
    return {
        "before_avg_precision": safe_mean(metrics_before["precision"]),
        "before_avg_recall": safe_mean(metrics_before["recall"]),
        "before_avg_f1": safe_mean(metrics_before["f1"]),
        "before_avg_jaccard": safe_mean(metrics_before["jaccard"]),
        "after_avg_precision": safe_mean(metrics_after["precision"]),
        "after_avg_recall": safe_mean(metrics_after["recall"]),
        "after_avg_f1": safe_mean(metrics_after["f1"]),
        "after_avg_jaccard": safe_mean(metrics_after["jaccard"]),
    }
