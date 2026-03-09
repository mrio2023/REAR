import pandas as pd
import numpy as np
import networkx as nx
import random
from community import community_louvain  # 第三方库版（推荐）
# 如果用NetworkX内置版，注释上面一行，打开下面两行：
# import networkx as nx
# from networkx.algorithms.community import louvain

# ===================== 核心配置 =====================
# 数据集列表（想跑哪个加哪个）
DATASETS = ["ibm", "elliptic", "elliptic2"]
# 基础路径模板（自动替换数据集名称）
BASE_PATH = "/home/u2023312299/sci/codes/df/{dataset}/{dataset}_{file_type}.csv"

# ===================== 核心函数 =====================
def calc_f1(pred_nodes, true_nodes):
    """计算单个社区的Precision/Recall/F1"""
    pred = set(pred_nodes)
    true = set(true_nodes)
    intersect = len(pred & true)
    
    precision = intersect / len(pred) if len(pred) > 0 else 0.0
    recall = intersect / len(true) if len(true) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def run_louvain_for_dataset(dataset_name):
    """单数据集Louvain运行函数"""
    print(f"\n{'='*80}")
    print(f"开始处理数据集：{dataset_name}")
    print(f"{'='*80}")
    
    # 1. 拼接文件路径（自动替换数据集名称）
    edge_path = BASE_PATH.format(dataset=dataset_name, file_type="edgelist")
    hacker_path = BASE_PATH.format(dataset=dataset_name, file_type="hacker")
    node_path = BASE_PATH.format(dataset=dataset_name, file_type="node_classes")
    
    # 2. 加载数据（跳过空值/异常值）
    try:
        df_edge = pd.read_csv(edge_path).dropna(subset=["from", "to"])
        df_hacker = pd.read_csv(hacker_path).dropna(subset=["address", "name_tag"])
        df_node = pd.read_csv(node_path).dropna(subset=["address"])
        print(f"✅ 数据加载完成：")
        print(f"   - 边数：{len(df_edge)}")
        print(f"   - 真实洗钱社区记录数：{len(df_hacker)}")
        print(f"   - 节点数：{len(df_node)}")
    except FileNotFoundError as e:
        print(f"❌ 数据集 {dataset_name} 文件不存在：{e}")
        return None
    
    # 3. 构建NetworkX图（只保留测试集中的节点，避免内存溢出）
    # 过滤边：只保留节点表中存在的节点
    valid_nodes = set(df_node["address"].tolist())
    df_edge_filtered = df_edge[
        (df_edge["from"].isin(valid_nodes)) & 
        (df_edge["to"].isin(valid_nodes))
    ]
    
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(valid_nodes)
    # 添加边
    edges = list(zip(df_edge_filtered["from"], df_edge_filtered["to"]))
    G.add_edges_from(edges)
    print(f"✅ 图构建完成：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    
    # 4. 运行Louvain社区发现（优化模块化度）
    print("🚀 开始运行Louvain算法...")
    try:
        # 第三方库版（推荐）
        partition = community_louvain.best_partition(G, random_state=2026)
        
        # NetworkX内置版（如果用这个，注释上面一行，打开下面几行）
        # comms = louvain.louvain_communities(G, seed=2026)
        # partition = {}
        # for cid, comm in enumerate(comms):
        #     for node in comm:
        #         partition[node] = cid
        
        print(f"✅ Louvain完成：共发现 {len(set(partition.values()))} 个社区")
    except Exception as e:
        print(f"❌ Louvain运行失败：{e}")
        return None
    
    # 5. 整理真实洗钱社区（按name_tag分组）
    true_communities = df_hacker.groupby("name_tag")["address"].apply(list).to_dict()
    true_communities = {k: v for k, v in true_communities.items() if len(v) > 0}
    print(f"✅ 真实洗钱社区数：{len(true_communities)}")
    
    # 6. 匹配社区并计算指标
    metrics = {"precision": [], "recall": [], "f1": []}
    for tag, true_nodes in true_communities.items():
        # 随机选一个种子节点（模拟你的模型采样逻辑）
        seed_node = random.choice(true_nodes)
        if seed_node not in partition:
            continue
        
        # 找到种子所属的Louvain社区
        cid = partition[seed_node]
        pred_nodes = [n for n, comm_id in partition.items() if comm_id == cid]
        
        # 计算指标
        p, r, f1 = calc_f1(pred_nodes, true_nodes)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1)
    
    # 7. 汇总结果
    if metrics["f1"]:
        avg_p = round(np.mean(metrics["precision"]), 4)
        avg_r = round(np.mean(metrics["recall"]), 4)
        avg_f1 = round(np.mean(metrics["f1"]), 4)
        std_f1 = round(np.std(metrics["f1"]), 4)
        
        print(f"\n📊 {dataset_name} Louvain结果汇总：")
        print(f"   Precision: {avg_p}")
        print(f"   Recall:    {avg_r}")
        print(f"   F1-Score:  {avg_f1} (±{std_f1})")
        return {"dataset": dataset_name, "P": avg_p, "R": avg_r, "F1": avg_f1}
    else:
        print(f"❌ {dataset_name} 无有效社区匹配结果")
        return None

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 批量运行所有数据集
    all_results = []
    for ds in DATASETS:
        res = run_louvain_for_dataset(ds)
        if res:
            all_results.append(res)
    
    # 打印最终汇总表
    print(f"\n{'='*80}")
    print("📈 所有数据集Louvain结果汇总")
    print(f"{'='*80}")
    print(f"{'数据集':10s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s}")
    print(f"{'-'*80}")
    for res in all_results:
        print(f"{res['dataset']:10s} | {res['P']:10.4f} | {res['R']:10.4f} | {res['F1']:10.4f}")
    print(f"{'='*80}")