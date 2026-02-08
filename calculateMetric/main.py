import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.cluster import normalized_mutual_info_score

# ---------------------- 1. 核心ONMI计算函数（保留论文同款逻辑，简化注释）----------------------
def calculate_onmi(true_labels, community_matrix):
    """
    计算重叠社区ONMI得分（0-1，越高表示社区划分与真实标签匹配度越好）
    """
    num_nodes = len(true_labels)
    num_communities = community_matrix.shape[1]
    
    # 构建社区虚拟标签（重叠节点取最后一个所属社区，论文常用处理）
    community_labels = np.full(num_nodes, -1)
    for comm_id in range(num_communities):
        comm_nodes = np.where(community_matrix[:, comm_id] == 1)[0]
        community_labels[comm_nodes] = comm_id
    
    # 计算归一化互信息
    return normalized_mutual_info_score(true_labels, community_labels, average_method='arithmetic')

# ---------------------- 2. 读取数据（直接套用你的真实路径）----------------------
# 重叠社区结果（你的真实路径）
df_overlapping = pd.read_csv(r"res1/PlusTokenPonzi_overlap_communities.csv")
# 真实标签数据（你的真实路径）
df_hacker = pd.read_csv(r"df/PlusTokenPonzi/PlusTokenPonzi_hacker.csv")

# ---------------------- 3. 数据预处理（简化核心步骤，保证节点匹配）----------------------
# 统一地址格式（消除大小写/空格差异，避免匹配失败）
df_overlapping["node"] = df_overlapping["node"].astype(str).str.lower().str.strip()
df_hacker["address"] = df_hacker["address"].astype(str).str.lower().str.strip()

# 筛选共同节点（只计算有真实标签的节点）
common_nodes = set(df_overlapping["node"]) & set(df_hacker["address"])
if not common_nodes:
    raise ValueError("两个数据集无共同区块链地址，请检查数据格式！")

# 过滤并排序（保证节点顺序严格一致）
df_overlap_filtered = df_overlapping[df_overlapping["node"].isin(common_nodes)].sort_values("node").reset_index(drop=True)
df_hacker_filtered = df_hacker[df_hacker["address"].isin(common_nodes)].sort_values("address").reset_index(drop=True)

# ===================== 新增：验证节点顺序是否一致 =====================
overlap_nodes_list = df_overlap_filtered["node"].tolist()
hacker_nodes_list = df_hacker_filtered["address"].tolist()

# 对比两个列表是否完全一致
if overlap_nodes_list != hacker_nodes_list:
    print("警告：节点顺序不一致！以下是前10个节点对比：")
    print("重叠社区节点前10：", overlap_nodes_list[:10])
    print("真实标签节点前10：", hacker_nodes_list[:10])
else:
    print("节点顺序完全一致，继续后续计算")
# ---------------------- 4. 真实标签编码 ----------------------
label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(df_hacker_filtered["label"].values)
print(f"真实标签分布：{dict(zip(label_encoder.classes_, np.bincount(true_labels)))}")

# ---------------------- 5. 构建二进制社区矩阵 ----------------------
# 提取所有唯一社区
all_communities = []
for comm_str in df_overlap_filtered["communities"]:
    comm_list = eval(comm_str)  # 解析字符串格式列表
    all_communities.extend(comm_list)
all_communities = list(set(all_communities))
num_communities = len(all_communities)
print(f"共检测到 {num_communities} 个重叠社区")

# 构建节点×社区的0/1矩阵
community_matrix = np.zeros((len(df_overlap_filtered), num_communities), dtype=int)
for idx, row in df_overlap_filtered.iterrows():
    comm_list = eval(row["communities"])
    for comm in comm_list:
        comm_idx = all_communities.index(comm)
        community_matrix[idx, comm_idx] = 1

# ---------------------- 6. 计算并输出ONMI ----------------------
onmi_score = calculate_onmi(true_labels, community_matrix)
print(f"\nONMI得分：{onmi_score:.4f}")
print("结果解读：越接近1越好，0.5+良好，0.7+优秀（参考论文标准）")