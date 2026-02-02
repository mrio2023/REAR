import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from graph import Graph
from nn.topology_encoder import TopologyEncoder
from nn.contrastive_layer import SemiSupervisedContrastiveLayer
from nn.temporal_layer import TemporalSupervisedLayer

# 全局配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
EMBED_DIM = 64
DECAY_RATE = 0.9  # 时序衰减率
TEMPERATURE = 0.07  # 对比损失温度系数

def main():
    # ===================== 1. 初始化Graph实例 =====================
    print("===== 初始化K-ego子图 =====")
    graph = Graph("AscendEXHacker")
    graph.init_graph(k=3)  # 构建3阶ego子图
    if graph.node_count == 0:
        print("Error: 子图节点数为0，退出程序")
        return

    # ===================== 2. 初始化三层嵌入器 =====================
    # 第一层：拓扑编码器
    feat_dim = 2  # 节点特征维度（度数+交易频率）
    topology_encoder = TopologyEncoder(
        in_dim=feat_dim,
        hidden_dim=HIDDEN_DIM,
        out_dim=EMBED_DIM
    ).to(DEVICE)

    # 第二层：半监督对比层
    contrast_layer = SemiSupervisedContrastiveLayer(
        embed_dim=EMBED_DIM,
        temperature=TEMPERATURE
    ).to(DEVICE)

    # 第三层：时序监督层
    snapshot_num = TemporalSupervisedLayer.get_snapshot_num(graph)
    temporal_layer = TemporalSupervisedLayer(
        embed_dim=EMBED_DIM,
        snapshot_num=snapshot_num,
        decay_rate=DECAY_RATE
    ).to(DEVICE)

    # ===================== 3. 准备训练数据 =====================
    # 拓扑编码器输入（Graph转PyG格式）
    x, edge_index = TopologyEncoder.graph2pyg_input(graph)
    # 节点标签（需替换为实际标签文件路径）
    node_labels = contrast_layer.get_node_labels(graph, node_label_path="df/AscendEXHacker/AscendEXHacker_node_classes.csv")

    # ===================== 4. 优化器配置 =====================
    optimizer = optim.AdamW(
        list(topology_encoder.parameters()) + 
        list(contrast_layer.parameters()) + 
        list(temporal_layer.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # ===================== 5. 训练流程 =====================
    print("===== 开始训练三层嵌入器 =====")
    topology_encoder.train()
    contrast_layer.train()
    temporal_layer.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # 第一层：拓扑嵌入
        topology_embed = topology_encoder(x, edge_index)

        # 第二层：对比学习 + 对比损失
        contrast_embed, contrast_loss = contrast_layer(topology_embed, node_labels)

        # 第三层：时序监督 + 时序损失
        final_embed, temporal_loss = temporal_layer(contrast_embed, graph)

        # 总损失
        total_loss = contrast_loss + temporal_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 打印训练日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Total Loss: {total_loss.item():.4f} | Contrast Loss: {contrast_loss.item():.4f} | Temporal Loss: {temporal_loss.item():.4f}")

    # ===================== 6. 推理：生成所有节点嵌入 =====================
    print("\n===== 训练完成，生成节点嵌入 =====")
    topology_encoder.eval()
    contrast_layer.eval()
    temporal_layer.eval()

    with torch.no_grad():
        # 推理流程
        topology_embed = topology_encoder(x, edge_index)
        contrast_embed, _ = contrast_layer(topology_embed, node_labels)
        final_embed, _ = temporal_layer(contrast_embed, graph)

    # 保存嵌入结果
    embed_df = pd.DataFrame({
        "node": graph.ego_nodes,
        "embed": final_embed.cpu().numpy().tolist()
    })
    embed_df.to_csv("node_embeddings.csv", index=False)
    print("节点嵌入已保存到 node_embeddings.csv")

    # ===================== 7. 重叠社区发现（层次聚类） =====================
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics.pairwise import cosine_similarity

    # 提取嵌入矩阵
    embed_matrix = np.array(embed_df['embed'].tolist())
    # 层次聚类
    Z = linkage(embed_matrix, method='ward')
    # 多阈值切割，生成重叠社区
    overlap_communities = {}
    thresholds = np.linspace(0.1, 1.0, 10)  # 不同切割阈值
    for node_idx, node in enumerate(graph.ego_nodes):
        node_embed = embed_matrix[node_idx].reshape(1, -1)
        # 计算与所有社区中心的相似度
        community_sims = []
        for thresh in thresholds:
            # 按阈值切割聚类
            clusters = fcluster(Z, t=thresh, criterion='distance')
            # 计算每个聚类的中心
            for cluster_id in np.unique(clusters):
                cluster_embeds = embed_matrix[clusters == cluster_id]
                cluster_center = cluster_embeds.mean(axis=0).reshape(1, -1)
                # 余弦相似度
                sim = cosine_similarity(node_embed, cluster_center)[0][0]
                if sim > 0.5:  # 相似度阈值
                    if node not in overlap_communities:
                        overlap_communities[node] = set()
                    overlap_communities[node].add(f"cluster_{cluster_id}_thresh_{thresh:.2f}")

    # 保存重叠社区结果
    community_df = pd.DataFrame({
        "node": list(overlap_communities.keys()),
        "communities": [list(v) for v in overlap_communities.values()]
    })
    community_df.to_csv("overlap_communities.csv", index=False)
    print("重叠社区结果已保存到 overlap_communities.csv")

if __name__ == "__main__":
    main()