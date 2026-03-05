from nn.dataProcess import dataProcess
from nn.graph import Graph
import random
import os

# 配置根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT_DIR)

# 初始化数据和图
dp = dataProcess(dfname="ibm", normal_node_ratio=2, expand_hop=2, min_community_size=3)
graph = Graph(dp.train_nodes, dp.train_feature, dp.train_hacker, dp.train_edge)

# 固定种子节点（你指定的两个）
seed = random.sample(graph.df_hacker["address"].values.tolist(),k=100)
tras = []  # 存储所有轨迹

# 定义精准率/召回率/F1计算函数（优化输出格式）
def getPre(true_nodes, pred_nodes, compare_type):
    """
    计算并打印P/R/F1，兼容空值
    :param true_nodes: 真实节点集合（对比基准）
    :param pred_nodes: 采样结果（(trajectory, is_weak) 元组）
    :param compare_type: 对比类型（用于打印标识）
    """
    # 提取轨迹列表（pred_nodes是 (trajectory, is_weak) 元组，取第一个元素）
    pred_traj = pred_nodes[0] if isinstance(pred_nodes, tuple) else pred_nodes
    # 转集合计算
    pred_set = set(pred_traj)
    true_set = set(true_nodes) if isinstance(true_nodes, (list, set)) else set()
    
    # 计算P/R/F1
    intersection = len(pred_set & true_set)
    p = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
    r = intersection / len(true_set) if len(true_set) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0.0
    
    print(f"  → 与{compare_type}比对：精准率(P): {p:.4f}, 召回率(R): {r:.4f}, F1值: {f1:.4f}")
    return p, r, f1

# 遍历种子节点执行采样和双数据集对比
for s in seed:
    try:
        print("="*60)
        # 执行采样（返回 (trajectory, is_weak)）
        tra = graph.sampleTrajectory(s, traj_length=100)
        tras.append(tra)  # 保存轨迹
        
        # 正确获取标签（字典取值用[]，不是()）
        tag = graph._addr2tag.get(s, None)  # 用get避免KeyError
        if tag is None:
            print(f"节点 {s} 无对应标签，跳过指标计算")
            continue
        
        # 1. 获取两个对比数据集
        steinT = graph.stanier_hacker.get(tag, set())  # 对比集1：完整斯坦纳树节点（含外部节点）
        hacker = graph.community_seeds.get(tag, set())  # 对比集2：社区原始黑客节点（纯目标节点）
        # 额外：斯坦纳树中的黑客节点（可选对比）
        stein_hacker = set([n for n in steinT if n in hacker])
        
        # 2. 打印基础信息
        print(f"节点：{s} | 所属社区：{tag}")
        print(f"  - 采样轨迹长度：{len(tra[0])}（期望100）")
        print(f"  - 社区原始黑客节点数：{len(hacker)}")
        print(f"  - 完整斯坦纳树节点数：{len(steinT)}（含{len(steinT)-len(stein_hacker)}个外部节点）")
        
        # 3. 双数据集指标对比
        print("\n【指标对比结果】")
        # 对比1：与社区原始黑客节点（纯目标）
        getPre(true_nodes=hacker, pred_nodes=tra, compare_type="社区原始黑客节点")
        # 对比2：与完整斯坦纳树节点（含外部节点）
        getPre(true_nodes=steinT, pred_nodes=tra, compare_type="完整斯坦纳树节点")
        # 可选：对比3：与斯坦纳树中的黑客节点
        getPre(true_nodes=stein_hacker, pred_nodes=tra, compare_type="斯坦纳树内黑客节点")
        
        # 4. 打印关键数据（方便直观分析）
        print("\n【关键数据展示】")
        print(f"  - 采样轨迹（前10个节点）：{tra[0][:10]} {'...' if len(tra[0])>10 else ''}")
        print(f"  - 完整斯坦纳树（前10个节点）：{list(steinT)[:10]} {'...' if len(steinT)>10 else ''}")
        print(f"  - 社区原始黑客节点（前10个）：{list(hacker)[:10]} {'...' if len(hacker)>10 else ''}")
        
    except Exception as e:
        print(f"处理节点 {s} 时出错：{e}")
        continue