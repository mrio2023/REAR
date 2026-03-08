import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nn.dataProcess import dataProcess
from nn.graph import Graph
import torch
import os


# 设置工作目录（精简路径计算逻辑）
current_file = os.path.abspath(__file__)
sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
os.chdir(sci_dir)

# 固定随机种子
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)

# 仅保留数据加载必要参数（无Configure依赖）
DATASET_PARAMS = {
    "ibm": {"normal_node_ratio":3, "expand_hop":2, "min_community_size":1},
    "elliptic": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":2},
    "elliptic2": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":5}
}

# 核心：邻居Top相似节点同社区比例分析
def analyze_neighbor_similarity(graph, top_percent=0.5, top_k=50, sample_num=200):
    # 提取hacker节点和社区映射
    df_hacker = graph.df_hacker.dropna(subset=["address", "name_tag"])
    if len(df_hacker) == 0:
        return 0.0
    
    addr2community = dict(zip(df_hacker["address"], df_hacker["name_tag"]))
    sampled_addrs = random.sample(df_hacker["address"].tolist(), min(len(df_hacker), sample_num))
    same_ratios = []

    for addr in sampled_addrs:
        # 获取当前节点嵌入
        try:
            node_embed = graph.singleNodeEmbed(addr).reshape(1, -1)
        except Exception:
            continue

        # 获取邻居（修复df_edge属性名）
        edge_df = graph.df_edge
        neighbors_from = edge_df[edge_df["from"] == addr]["target"].tolist()
        neighbors_to = edge_df[edge_df["target"] == addr]["source"].tolist()
        neighbor_addrs = list(set(neighbors_from + neighbors_to))
        
        if len(neighbor_addrs) < 1:
            continue

        # 过滤有效邻居并计算相似度
        valid_neighbors, neighbor_embeds = [], []
        for n_addr in neighbor_addrs:
            if n_addr not in addr2community:
                continue
            try:
                neighbor_embeds.append(graph.singleNodeEmbed(n_addr))
                valid_neighbors.append(n_addr)
            except Exception:
                continue
        
        if len(valid_neighbors) < 1:
            continue

        # 计算Top相似邻居同社区比例
        sim_scores = cosine_similarity(node_embed, np.array(neighbor_embeds))[0]
        top_n = max(1, int(len(valid_neighbors)*top_percent/100), top_k)
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        top_neighbors = [valid_neighbors[i] for i in top_indices]

        current_comm = addr2community[addr]
        same_count = sum(1 for n in top_neighbors if addr2community[n] == current_comm)
        same_ratios.append(same_count / len(top_neighbors))

    return np.mean(same_ratios) if same_ratios else 0.0

# 核心测试函数（完全移除Configure、直接传参）
def test_embed(dfname, seed=2026):
    set_seed(seed)
    if dfname not in DATASET_PARAMS:
        print(f"❌ 无{dfname}配置")
        return
    
    # 直接传参给dataProcess（无Configure）
    params = DATASET_PARAMS[dfname]
    dp = dataProcess(
        dfname=dfname,
        normal_node_ratio=params["normal_node_ratio"],
        expand_hop=params["expand_hop"],
        min_community_size=params["min_community_size"]
    )

    # 构建图（直接传参）
    train_g = Graph(
        dfnode=dp.train_nodes,
        dffeature=dp.train_feature,
        dfhacker=dp.train_hacker,
        dfedge=dp.train_edge  # 匹配Graph类的df_edge属性
    )

    # 计算并输出结果
    ratio = analyze_neighbor_similarity(train_g)
    print(f"\n📊 {dfname} - Top0.5%/Top50相似邻居同社区率：{ratio:.4f}")
    # 剪枝可行性判断
    print(f"✅ 适合余弦剪枝" if ratio > 0.7 else "❌ 不适合余弦剪枝")

# 运行测试
if __name__ == "__main__":
    for dfname in ["ibm", "elliptic", "elliptic2"]:
        test_embed(dfname)