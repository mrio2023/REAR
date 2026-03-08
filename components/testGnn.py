import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nn.dataProcess import dataProcess
from nn.graph import Graph
import torch

# ==================== 1. 路径配置（核心：自动定位工作目录） ====================
def setup_workspace():
    """自动配置工作目录，确保路径正确"""
    # 获取当前脚本绝对路径
    current_file = os.path.abspath(__file__)
    # 向上三级目录（匹配你的 sci 目录层级）
    sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    # 设置为工作目录
    os.chdir(sci_dir)
    # 打印验证（可选，便于调试）
    print(f"✅ 工作目录已设置为：{sci_dir}")
    print(f"✅ 当前工作目录验证：{os.getcwd()}")
    return sci_dir

# 初始化路径
setup_workspace()

# ==================== 2. 基础配置 ====================
# 固定随机种子
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 如需使用torch，补充种子

# 数据集核心参数（仅保留必需项）
DATASET_PARAMS = {
    "ibm": {"normal_node_ratio":3, "expand_hop":2, "min_community_size":1},
    "elliptic": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":2},
    "elliptic2": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":5}
}

# ==================== 3. 核心分析函数（完全适配你的Graph类） ====================
def analyze_neighbor_similarity(graph, top_percent=0.5, top_k=50, sample_num=200):
    """
    分析邻居Top相似节点的同社区比例（适配你的Graph类）
    - 使用graph.getSingleNodeNeighbor获取邻居
    - 使用graph.singleNodeEmbed获取嵌入
    """
    # 提取hacker节点和社区映射
    df_hacker = graph.df_hacker.dropna(subset=["address", "name_tag"])
    if len(df_hacker) == 0:
        print("❌ 无有效hacker社区数据")
        return 0.0
    
    addr2community = dict(zip(df_hacker["address"], df_hacker["name_tag"]))
    sampled_addrs = random.sample(df_hacker["address"].tolist(), min(len(df_hacker), sample_num))
    
    same_ratios = []
    for idx, addr in enumerate(sampled_addrs):
        if idx % 20 == 0:
            print(f"   进度：{idx}/{len(sampled_addrs)} 节点")
        
        # 1. 获取当前节点嵌入（适配你的singleNodeEmbed方法）
        try:
            node_embed = graph.singleNodeEmbed(addr).reshape(1, -1)
        except ValueError:
            continue
        
        # 2. 获取邻居（适配你的getSingleNodeNeighbor方法）
        neighbor_addrs = graph.getSingleNodeNeighbor(addr)
        if len(neighbor_addrs) < 1:
            continue
        
        # 3. 过滤有效邻居（有社区+有嵌入）
        valid_neighbors, neighbor_embeds = [], []
        for n_addr in neighbor_addrs:
            if n_addr not in addr2community:
                continue
            try:
                neighbor_embeds.append(graph.singleNodeEmbed(n_addr))
                valid_neighbors.append(n_addr)
            except ValueError:
                continue
        
        if len(valid_neighbors) < 1:
            continue
        
        # 4. 计算余弦相似度并取Top
        sim_scores = cosine_similarity(node_embed, np.array(neighbor_embeds))[0]
        # 确定Top数量（0.5% 或 Top50，取较大值）
        top_n = max(1, int(len(valid_neighbors)*top_percent/100), top_k)
        top_n = min(top_n, len(valid_neighbors))  # 防止越界
        # 按相似度排序取Top
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        top_neighbors = [valid_neighbors[i] for i in top_indices]
        
        # 5. 统计同社区比例
        current_comm = addr2community[addr]
        same_count = sum(1 for n in top_neighbors if addr2community[n] == current_comm)
        same_ratios.append(same_count / len(top_neighbors))
    
    # 计算平均同社区率
    avg_ratio = np.mean(same_ratios) if same_ratios else 0.0
    print(f"\n📊 平均同社区比例：{avg_ratio:.4f}")
    print(f"✅ 适合余弦剪枝" if avg_ratio > 0.7 else "❌ 不适合余弦剪枝")
    return avg_ratio

# ==================== 4. 测试主函数 ====================
def test_dataset(dfname):
    """测试单个数据集"""
    set_seed(2026)
    if dfname not in DATASET_PARAMS:
        print(f"❌ 无{dfname}配置")
        return
    
    # 加载数据
    params = DATASET_PARAMS[dfname]
    dp = dataProcess(
        dfname=dfname,
        normal_node_ratio=params["normal_node_ratio"],
        expand_hop=params["expand_hop"],
        min_community_size=params["min_community_size"]
    )
    
    # 构建Graph（完全匹配你的Graph类入参）
    train_g = Graph(
        dfnode=dp.train_nodes,
        dffeature=dp.train_feature,
        dfhacker=dp.train_hacker,
        dfedge=dp.train_edge  # 你的Graph类入参是dfedge，实例内是df_edge
    )
    
    # 核心分析
    print(f"\n{'='*60}\n🚀 分析数据集：{dfname}")
    analyze_neighbor_similarity(train_g)

# ==================== 5. 运行测试 ====================
if __name__ == "__main__":
    # 测试所有数据集
    for dfname in ["ibm", "elliptic", "elliptic2"]:
        test_dataset(dfname)