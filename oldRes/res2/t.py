import pandas as pd
import os
import numpy as np
from ast import literal_eval
from scipy.spatial.distance import cosine
import igraph as ig

def calculate_community_cohesion(TQ, df_embed):
    """
    计算社区内聚性：社区节点间平均相似度
    """
    if len(TQ) < 2:
        return 0
    
    # 获取社区所有节点的嵌入
    embeddings = []
    for node in TQ:
        embed_row = df_embed[df_embed['node'].str.lower() == node.lower()]
        if not embed_row.empty:
            embeddings.append(embed_row['embed_array'].iloc[0])
    
    if len(embeddings) < 2:
        return 0
    
    # 计算所有节点对之间的余弦相似度
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            try:
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)
            except:
                continue
    
    return np.mean(similarities) if similarities else 0

def calculate_cohesion_gain(TQ, candidate, df_embed):
    """
    计算加入候选节点后的内聚性增益
    """
    current_cohesion = calculate_community_cohesion(TQ, df_embed)
    new_cohesion = calculate_community_cohesion(TQ + [candidate], df_embed)
    return new_cohesion - current_cohesion

def calculate_similarity_to_community(TQ, candidate, df_embed):
    """
    计算候选节点与社区的平均相似度
    """
    if not TQ:
        return 0
    
    # 获取候选节点嵌入
    candidate_embed = df_embed[df_embed['node'].str.lower() == candidate.lower()]
    if candidate_embed.empty:
        return 0
    candidate_embed = candidate_embed['embed_array'].iloc[0]
    
    # 计算与社区中每个节点的相似度
    similarities = []
    for node in TQ:
        node_embed = df_embed[df_embed['node'].str.lower() == node.lower()]
        if not node_embed.empty:
            try:
                sim = 1 - cosine(candidate_embed, node_embed['embed_array'].iloc[0])
                similarities.append(sim)
            except:
                continue
    
    return np.mean(similarities) if similarities else 0

def update_search_space_D(TQ, g, node_to_name, name_to_node):
    """
    动态更新搜索空间D：包含当前社区TQ的所有一阶邻居
    """
    new_D = set()
    
    # 对于TQ中的每个节点，添加其一阶邻居
    for node_name in TQ:
        node_lower = node_name.lower()
        if node_lower in name_to_node:
            node_id = name_to_node[node_lower]
            neighbor_ids = g.neighbors(node_id)
            neighbor_names = [node_to_name[nid] for nid in neighbor_ids]
            new_D.update(neighbor_names)
    
    # 移除已经在社区中的节点
    new_D.difference_update(TQ)
    
    return list(new_D)

def calculate_node_score(candidate, TQ, df_embed, method="hybrid"):
    """
    计算候选节点的得分，可以考虑多种策略
    method: "similarity" - 仅考虑与社区的相似度
            "cohesion" - 仅考虑内聚性增益
            "hybrid" - 综合考虑
    """
    if method == "similarity":
        return calculate_similarity_to_community(TQ, candidate, df_embed)
    elif method == "cohesion":
        return calculate_cohesion_gain(TQ, candidate, df_embed)
    else:  # hybrid
        similarity = calculate_similarity_to_community(TQ, candidate, df_embed)
        cohesion_gain = calculate_cohesion_gain(TQ, candidate, df_embed)
        # 权重可以调整，这里给相似度更高权重
        return 0.7 * similarity + 0.3 * cohesion_gain

def extractor_single_seed_community(dfname, similarity_threshold=0.6, min_community_size=3):
    rootPath = "df"
    dfPath = os.path.join(rootPath, dfname)
    seedPath = os.path.join(dfPath, f"{dfname}_seed.csv")
    embedPath = os.path.join("res2", f"{dfname}_node_embeddings.csv")
    edgePath = os.path.join(dfPath, f"{dfname}_edgelist.csv")

    # 读取数据
    df_seeds = pd.read_csv(seedPath)
    df_embed = pd.read_csv(embedPath)
    df_edge = pd.read_csv(edgePath)
    
    # 预处理：嵌入向量从字符串转numpy数组
    df_embed['embed_array'] = df_embed['embed'].apply(lambda x: np.array(literal_eval(x), dtype=np.float32))
    
    # 构建图
    g = ig.Graph.TupleList(df_edge[['from', 'to']].values, directed=False)
    node_to_name = {v.index: v['name'] for v in g.vs}
    name_to_node = {v['name'].lower(): v.index for v in g.vs}
    
    # 对每个种子节点独立执行社区拓展
    all_communities = []
    seed_addresses = df_seeds['address'].tolist()
    
    for seed_idx, seed_addr in enumerate(seed_addresses):
        print(f"\n===== 处理第 {seed_idx+1} 个种子节点：{seed_addr} =====")
        
        # 1. 初始化社区
        TQ = [seed_addr]
        seed_addr_lower = seed_addr.lower()
        
        # 检查种子是否有嵌入
        if df_embed[df_embed['node'].str.lower() == seed_addr_lower].empty:
            print(f"警告：种子 {seed_addr} 无对应嵌入，跳过")
            continue
        
        # 2. 初始搜索空间：种子的一阶邻居
        D = []
        if seed_addr_lower in name_to_node:
            seed_node_id = name_to_node[seed_addr_lower]
            neighbor_ids = g.neighbors(seed_node_id)
            neighbor_addrs = [node_to_name[nid] for nid in neighbor_ids]
            D = neighbor_addrs
        print(f"初始候选集 D 规模：{len(D)}（种子的一阶邻居）")
        
        # 3. 迭代拓展社区
        iteration = 0
        max_iterations = 20  # 防止无限循环
        previous_cohesion = 0
        
        while D and iteration < max_iterations:
            iteration += 1
            
            # 计算候选节点的得分
            candidate_scores = {}
            for candidate in D:
                # 检查候选节点是否有嵌入
                if df_embed[df_embed['node'].str.lower() == candidate.lower()].empty:
                    candidate_scores[candidate] = -1
                    continue
                    
                score = calculate_node_score(candidate, TQ, df_embed, method="hybrid")
                candidate_scores[candidate] = score
            
            # 选择得分最高的节点
            valid_candidates = {k: v for k, v in candidate_scores.items() if v >= 0}
            if not valid_candidates:
                print(f"迭代 {iteration}：无有效候选节点，停止")
                break
                
            best_candidate = max(valid_candidates.items(), key=lambda x: x[1])
            candidate_node, best_score = best_candidate
            
            # 检查是否满足相似度阈值
            similarity = calculate_similarity_to_community(TQ, candidate_node, df_embed)
            if similarity < similarity_threshold:
                print(f"迭代 {iteration}：最佳候选 {candidate_node} 相似度 {similarity:.3f} 低于阈值 {similarity_threshold}，停止")
                break
            
            # 计算加入节点后的内聚性
            new_TQ = TQ + [candidate_node]
            new_cohesion = calculate_community_cohesion(new_TQ, df_embed)
            
            # 检查内聚性是否下降太多（超过20%）
            if previous_cohesion > 0 and new_cohesion < previous_cohesion * 0.8:
                print(f"迭代 {iteration}：加入节点 {candidate_node} 后内聚性下降过多 ({previous_cohesion:.3f} → {new_cohesion:.3f})，停止")
                break
            
            # 加入节点到社区
            TQ.append(candidate_node)
            previous_cohesion = new_cohesion
            
            print(f"迭代 {iteration}：加入节点 {candidate_node}，得分：{best_score:.3f}，相似度：{similarity:.3f}，内聚性：{new_cohesion:.3f}")
            
            # 更新搜索空间：添加新加入节点的邻居
            D.remove(candidate_node)
            new_D = update_search_space_D([candidate_node], g, node_to_name, name_to_node)
            # 移除已经在社区中的节点
            new_D = [node for node in new_D if node not in TQ]
            D.extend(new_D)
            D = list(set(D))  # 去重
            
            print(f"更新后候选集规模：{len(D)}")
        
        # 4. 后处理：确保社区的最小规模
        if len(TQ) < min_community_size:
            print(f"社区规模 {len(TQ)} 小于最小要求 {min_community_size}，尝试补充")
            # 可以考虑放宽阈值或重新评估
            
        # 计算最终社区指标
        final_cohesion = calculate_community_cohesion(TQ, df_embed)
        avg_similarity = np.mean([
            calculate_similarity_to_community([seed_addr], node, df_embed) 
            for node in TQ if node != seed_addr
        ]) if len(TQ) > 1 else 0
        
        # 保存结果
        all_communities.append({
            'seed_node': seed_addr,
            'community': TQ,
            'community_size': len(TQ),
            'cohesion': final_cohesion,
            'avg_similarity_to_seed': avg_similarity,
            'iterations': iteration
        })
        
        print(f"种子 {seed_addr} 最终社区：")
        print(f"  规模：{len(TQ)}")
        print(f"  内聚性：{final_cohesion:.3f}")
        print(f"  平均与种子相似度：{avg_similarity:.3f}")
    
    # 汇总结果
    df_result = pd.DataFrame(all_communities)
    
    print(f"\n===== 社区挖掘完成 =====")
    print(f"共处理 {len(all_communities)} 个种子")
    print(f"平均社区规模：{df_result['community_size'].mean():.1f}")
    print(f"平均内聚性：{df_result['cohesion'].mean():.3f}")
    
    return df_result, df_embed, df_edge

# 调用函数
df_communities, df_embed, df_edge = extractor_single_seed_community("AscendEXHacker")