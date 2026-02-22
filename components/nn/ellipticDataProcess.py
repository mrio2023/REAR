import pandas as pd
import random
import os

class ellipticDataProcess():
    def __init__(self, dfname):
        self.dfname = dfname
        # 读取数据（适配dfname参数的通用路径）
        df_edge, df_nodes, df_feature, df_hacker = self.readData()
      
        # 社区过滤与训练/测试拆分（最小社区大小10）
        self.df_hacker_filtered, self.train_hacker, self.test_hacker = self.getTrainAndTestCom(df_hacker, min_community_size=10)
        
        # 拆分节点/特征数据
        self.train_nodes, self.test_nodes, self.train_feature, self.test_feature = self.splitData(df_nodes, df_feature)
        
        # 额外拆分边数据（可选，保持数据完整性）
        self.train_edge, self.test_edge = self.splitEdgeData(df_edge)
        
        # 清理临时变量释放内存
        del df_edge, df_nodes, df_feature, df_hacker

    def readData(self):
        base_path = os.path.join("codes", "df", self.dfname)
        edge_path = os.path.join(base_path, f"{self.dfname}_edgelist.csv")
        node_path = os.path.join(base_path, f"{self.dfname}_node_classes.csv")
        feature_path = os.path.join(base_path, f"{self.dfname}_features.csv")
        hacker_path = os.path.join(base_path, f"{self.dfname}_hacker.csv")
        
        try:
            df_edge = pd.read_csv(edge_path)
            df_nodes = pd.read_csv(node_path)
            df_feature = pd.read_csv(feature_path)
            df_hacker = pd.read_csv(hacker_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件不存在：{e.filename}")
        
        return df_edge, df_nodes, df_feature, df_hacker

    def getTrainAndTestCom(self, df_hacker, min_community_size:int=10):
        # 过滤空标签社区
        indices_to_drop = []
        for tag, group_df in df_hacker.groupby("name_tag"):
            if pd.isna(tag):
                indices_to_drop.extend(group_df.index.tolist())
        df_hacker_filtered = df_hacker.drop(indices_to_drop).copy() if indices_to_drop else df_hacker.copy()
        
        # 过滤小社区
        large_tags = []
        for tag in df_hacker_filtered["name_tag"].unique():
            if len(df_hacker_filtered[df_hacker_filtered["name_tag"] == tag]) >= min_community_size:
                large_tags.append(tag)
        df_hacker_filtered = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(large_tags)].copy()
        
        # 拆分训练/测试社区
        train_size = int(0.75 * len(large_tags))
        train_size = max(1, min(train_size, len(large_tags)-1))
        train_com = random.sample(large_tags, k=train_size)
        test_com = [c for c in large_tags if c not in train_com]
        
        # 生成训练/测试黑客数据
        train_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(train_com)].copy()
        test_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(test_com)].copy()
        
        return df_hacker_filtered, train_hacker, test_hacker

    def splitData(self, df_nodes, df_feature):
        # 提取测试节点ID
        test_node_ids = self.test_hacker["address"].unique().tolist()
        
        # 拆分节点数据
        test_nodes = df_nodes[df_nodes["address"].isin(test_node_ids)].copy()
        train_nodes = df_nodes.drop(test_nodes.index).copy() if not test_nodes.empty else df_nodes.copy()
        
        # 拆分特征数据
        test_feature = df_feature[df_feature["address"].isin(test_node_ids)].copy()
        train_feature = df_feature.drop(test_feature.index).copy() if not test_feature.empty else df_feature.copy()
        
        return train_nodes, test_nodes, train_feature, test_feature

    def splitEdgeData(self, df_edge):
        # 拆分边数据（可选，保持完整）
        test_node_ids = self.test_hacker["address"].unique().tolist()
        test_edge = df_edge[(df_edge["from"].isin(test_node_ids)) | (df_edge["to"].isin(test_node_ids))].copy()
        train_edge = df_edge.drop(test_edge.index).copy() if not test_edge.empty else df_edge.copy()
        return train_edge, test_edge

# 用法示例
if __name__ == "__main__":
    # 初始化处理器（传入数据集名称）
    processor = ellipticDataProcess("elliptic_txs")
    
    # 输出核心数据（和参考代码结构完全一致）
    print("=== 核心拆分结果 ===")
    print(f"过滤后黑客数据行数: {len(processor.df_hacker_filtered)}")
    print(f"训练黑客数据行数: {len(processor.train_hacker)}")
    print(f"测试黑客数据行数: {len(processor.test_hacker)}")
    print(f"训练节点数: {len(processor.train_nodes)}")
    print(f"测试节点数: {len(processor.test_nodes)}")
    print(f"训练特征数: {len(processor.train_feature)}")
    print(f"测试特征数: {len(processor.test_feature)}")