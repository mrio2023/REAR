import os
import pandas as pd
import random

class DataProcess:
    def __init__(self, dfname):
        self.dfname = dfname
      
        df_nodes, df_feature, df_hacker = self.readData()
      
        # 修改：不再过滤小社区，直接使用所有社区（注：实际过滤逻辑移到getTrainAndTestCom里了）
        self.df_hacker_filtered, self.train_hacker, self.test_hacker = self.getTrainAndTestCom(df_hacker, min_community_size=5)  # 可自定义小社区阈值
        
        self.train_nodes, self.test_nodes, self.train_feature, self.test_feature = self.splitData(df_nodes, df_feature)
        
        del df_nodes, df_feature, df_hacker

    def splitData(self, df_nodes, df_feature): 
        test_node_ids = self.test_hacker["address"].unique().tolist()
     
        test_nodes = df_nodes[df_nodes["address"].isin(test_node_ids)].copy()
        test_feature = df_feature[
            (df_feature["from"].isin(test_node_ids)) | 
            (df_feature["to"].isin(test_node_ids))
        ].copy()
        
        train_nodes = df_nodes.drop(test_nodes.index).copy() if not test_nodes.empty else df_nodes.copy()
        train_feature = df_feature.drop(test_feature.index).copy() if not test_feature.empty else df_feature.copy()
    
        return train_nodes, test_nodes, train_feature, test_feature

    def readData(self):
        rootPath = os.path.join("codes", "df")
        dfPath = os.path.join(rootPath, self.dfname)

        hackerPath = os.path.join(dfPath, f"{self.dfname}_hacker.csv")
        nodePath = os.path.join(dfPath, f"{self.dfname}_node_classes.csv")
        featuresPath = os.path.join(dfPath, f"{self.dfname}_features.csv")

        try:
            df_hacker = pd.read_csv(hackerPath)
            df_nodes = pd.read_csv(nodePath)
            df_feature = pd.read_csv(featuresPath)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件未找到：{e.filename}，请检查文件路径和名称")

        return df_nodes, df_feature, df_hacker

    def getTrainAndTestCom(self, df_hacker, min_community_size:int=2): 
        """
        修改点：
        1. 过滤空标签社区
        2. 过滤小社区（节点数 < min_community_size），直接移除不纳入计算
        3. 仅对剩余的大社区划分训练/测试集
        """
        # 第一步：删除标签为空的记录（name_tag为NaN的）
        indices_to_drop = []
        for tag, group_df in df_hacker.groupby("name_tag"):
            if pd.isna(tag):  # 只删除空标签
                indices_to_drop.extend(group_df.index.tolist())
                print(f"移除空标签社区: {len(group_df)}条记录")
        
        # 过滤掉空标签的记录
        df_hacker_filtered = df_hacker.drop(indices_to_drop).copy() if indices_to_drop else df_hacker.copy()
        
        if df_hacker_filtered.empty:
            raise ValueError("过滤空标签后无有效社区数据")
        
        # 第二步：过滤小社区（核心改动）
        small_community_tags = []  # 存储小社区的标签
        large_community_tags = []  # 存储大社区的标签
        for tag in df_hacker_filtered["name_tag"].unique():
            community_size = len(df_hacker_filtered[df_hacker_filtered["name_tag"] == tag])
            if community_size < min_community_size:
                small_community_tags.append((tag, community_size))
            else:
                large_community_tags.append(tag)
        
        # 移除小社区的所有记录
        df_hacker_filtered = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(large_community_tags)].copy()
        
        if df_hacker_filtered.empty:
            raise ValueError(f"过滤小社区（<{min_community_size}节点）后无有效社区数据")
        
        # 第三步：仅对大社区划分训练/测试集（75%/25%）
        train_size = int(0.75 * len(large_community_tags))
        trainCom = random.sample(large_community_tags, train_size)
        testCom = list(set(large_community_tags) - set(trainCom))
        
        # 生成train_hacker/test_hacker（仅包含大社区）
        train_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(trainCom)].copy()
        test_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(testCom)].copy()
        
        # 打印统计信息（更清晰的过滤报告）
        print(f"\n=== 社区过滤与划分统计 ===")
        print(f"原始社区总数（非空标签）: {len(df_hacker['name_tag'].dropna().unique())}")
        print(f"小社区数（<{min_community_size}节点）: {len(small_community_tags)} (已过滤)")
        print(f"大社区数（≥{min_community_size}节点）: {len(large_community_tags)} (用于划分)")
        print(f"  - 训练社区: {len(trainCom)} (75%)")
        print(f"  - 测试社区: {len(testCom)} (25%)")
        
        # 打印小社区详情（可选）
        if small_community_tags:
            print(f"\n被过滤的小社区列表（前10个）:")
            for tag, size in sorted(small_community_tags, key=lambda x: x[1])[:10]:
                print(f"  {tag}: {size}节点")
            if len(small_community_tags) > 10:
                print(f"  ... 还有{len(small_community_tags)-10}个小社区未显示")
        
        return df_hacker_filtered, train_hacker, test_hacker