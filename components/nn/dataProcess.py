import os
import pandas as pd
import random
import gc

class DataProcess:
    def __init__(self, dfname):
        self.dfname = dfname
      
        df_nodes, df_feature, df_hacker = self.readData()
      
        # 修改：不再过滤小社区，直接使用所有社区
        self.df_hacker_filtered, self.train_hacker, self.test_hacker = self.getTrainAndTestCom(df_hacker)
        
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

    def getTrainAndTestCom(self, df_hacker, min_size=3):
        """
        修改点：不再删除小社区，只处理空标签
        """
        # 只删除标签为空的记录（name_tag为NaN的）
        indices_to_drop = []
        for tag, group_df in df_hacker.groupby("name_tag"):
            if pd.isna(tag):  # 只删除空标签
                indices_to_drop.extend(group_df.index.tolist())
                print(f"移除空标签社区: {len(group_df)}条记录")
        
        # 过滤掉空标签的记录
        df_hacker_filtered = df_hacker.drop(indices_to_drop).copy() if indices_to_drop else df_hacker.copy()
        
        if df_hacker_filtered.empty:
            raise ValueError("过滤后无有效社区数据")
        
        # 随机划分训练/测试社区（75%/25%）
        tagList = df_hacker_filtered["name_tag"].unique().tolist()
        train_size = int(0.75 * len(tagList))
        
        trainCom = random.sample(tagList, train_size)
        testCom = list(set(tagList) - set(trainCom))
        
        # 生成train_hacker/test_hacker
        train_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(trainCom)].copy()
        test_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(testCom)].copy()
        
        # 打印统计信息
        print(f"\n社区统计:")
        print(f"  总社区数: {len(tagList)}")
        print(f"  训练社区: {len(trainCom)} (75%)")
        print(f"  测试社区: {len(testCom)} (25%)")
        
        # 打印小社区信息（方便观察）
        small_coms = []
        for tag in tagList:
            size = len(df_hacker_filtered[df_hacker_filtered["name_tag"] == tag])
            if size < 10:
                small_coms.append((tag, size))
        
        if small_coms:
            print(f"\n小社区统计 (<10节点):")
            for tag, size in sorted(small_coms, key=lambda x: x[1])[:10]:  # 只显示前10个
                print(f"  {tag}: {size}节点")
            if len(small_coms) > 10:
                print(f"  ... 还有{len(small_coms)-10}个小社区")
        
        return df_hacker_filtered, train_hacker, test_hacker