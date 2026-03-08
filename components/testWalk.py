import os
import random
import numpy as np
import pandas as pd
from nn.dataProcess import dataProcess
from nn.graph import Graph

import os


# 设置工作目录（精简路径计算逻辑）
current_file = os.path.abspath(__file__)
sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
os.chdir(sci_dir)

# 固定随机种子
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)

# ========== 核心：随机游走全社区覆盖测试 ==========
def test_walk_coverage(graph, max_walk_steps=1000, sample_communities=20):
    """
    测试随机游走是否能遍历全社区
    :param graph: 构建好的Graph实例
    :param max_walk_steps: 单条游走最大步数（防止无限循环）
    :param sample_communities: 抽样测试的社区数量
    :return: 各社区覆盖度统计、是否能遍历全社区的结论
    """
    # 1. 预处理社区数据
    df_hacker = graph.df_hacker.dropna(subset=["address", "name_tag"])
    if df_hacker.empty:
        print("❌ 无有效hacker社区数据")
        return {"coverage": 0.0, "can_cover_all": False, "details": []}
    
    # 按社区分组，过滤过小的社区（避免无意义测试）
    comm2addrs = df_hacker.groupby("name_tag")["address"].apply(set).to_dict()
    valid_comms = {k: v for k, v in comm2addrs.items() if len(v) >= 2}  # 至少2个节点的社区
    if not valid_comms:
        print("❌ 无有效社区（社区节点数≥2）")
        return {"coverage": 0.0, "can_cover_all": False, "details": []}
    
    # 2. 抽样测试社区
    sampled_comms = random.sample(list(valid_comms.keys()), min(sample_communities, len(valid_comms)))
    results = []
    all_cover_all = True  # 是否所有抽样社区都能遍历全

    for comm_name in sampled_comms:
        comm_nodes = valid_comms[comm_name]
        comm_size = len(comm_nodes)
        start_node = random.choice(list(comm_nodes))  # 随机选起始节点
        
        # 3. 随机游走核心逻辑
        walked_nodes = set([start_node])
        current_node = start_node
        step = 0
        cover_all = False
        
        while step < max_walk_steps and len(walked_nodes) < comm_size:
            # 获取当前节点的所有邻居
            neighbors = graph.getSingleNodeNeighbor(current_node)
                        # 过滤掉已走过的邻居，只选未访问的
            unvisited_neighbors = [n for n in neighbors if n not in walked_nodes and n in comm_nodes]
            if not unvisited_neighbors:
                break  # 无未访问邻居，终止游走
            
            # 随机选下一个节点（纯随机游走）
            next_node = random.choice(neighbors)
            if next_node in comm_nodes:  # 仅保留社区内节点
                walked_nodes.add(next_node)
                current_node = next_node
            
            step += 1
        
        # 4. 计算该社区的覆盖指标
        coverage = len(walked_nodes) / comm_size
        cover_all = (coverage == 1.0)
        all_cover_all = all_cover_all and cover_all
        
        result = {
            "community": comm_name,
            "community_size": comm_size,
            "walked_steps": step,
            "covered_nodes": len(walked_nodes),
            "coverage": round(coverage, 4),
            "can_cover_all": cover_all
        }
        results.append(result)
        
        # 打印单社区结果
        print(f"\n📌 社区 {comm_name}（大小：{comm_size}）")
        print(f"   游走步数：{step} | 覆盖节点数：{len(walked_nodes)} | 覆盖度：{coverage:.4f}")
        print(f"   是否遍历全社区：{'✅' if cover_all else '❌'}")

    # 5. 整体统计
    avg_coverage = np.mean([r["coverage"] for r in results])
    final_result = {
        "avg_coverage": round(avg_coverage, 4),
        "can_cover_all": all_cover_all,
        "details": results
    }

    # 输出最终结论
    print("\n==================== 随机游走全社区覆盖测试结论 ====================")
    print(f"平均覆盖度：{avg_coverage:.4f}")
    if all_cover_all:
        print("✅ 随机游走可以遍历全社区（抽样测试）")
    else:
        print("❌ 随机游走无法遍历全社区（抽样测试），建议更换方案")
    
    return final_result

# ========== 运行测试（适配你的数据流程） ==========
def run_walk_test(dfname, seed=2026):
    set_seed(seed)
    DATASET_PARAMS = {
        "ibm": {"normal_node_ratio":3, "expand_hop":2, "min_community_size":1},
        "elliptic": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":2},
        "elliptic2": {"normal_node_ratio":2, "expand_hop":2, "min_community_size":5}
    }
    
    if dfname not in DATASET_PARAMS:
        print(f"❌ 无{dfname}配置")
        return
    
    # 加载数据+构建图
    params = DATASET_PARAMS[dfname]
    dp = dataProcess(
        dfname=dfname,
        normal_node_ratio=params["normal_node_ratio"],
        expand_hop=params["expand_hop"],
        min_community_size=params["min_community_size"]
    )

    train_g = Graph(
        dfnode=dp.train_nodes,
        dffeature=dp.train_feature,
        dfhacker=dp.train_hacker,
        dfedge=dp.train_edge
    )

    # 执行随机游走覆盖测试
    test_walk_coverage(train_g, max_walk_steps=1000, sample_communities=5)

# 运行测试
if __name__ == "__main__":
    for dfname in ["ibm", "elliptic", "elliptic2"]:
        print(f"\n==================== 测试数据集：{dfname} ====================")
        run_walk_test(dfname)