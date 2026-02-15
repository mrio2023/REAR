import os
import pandas as pd
import random
import gc  # 引入垃圾回收模块

#这个是测试用，之后测试完得删掉

class DataProcess:
    def __init__(self, dfname):
        self.dfname = dfname
      
        df_nodes, df_feature, df_hacker= self.readData()
      
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
        
        # 分割训练数据
        train_nodes = df_nodes.drop(test_nodes.index).copy() if not test_nodes.empty else df_nodes.copy()
        train_feature = df_feature.drop(test_feature.index).copy() if not test_feature.empty else df_feature.copy()

    
        return train_nodes, test_nodes, train_feature, test_feature

    def readData(self):

        rootPath = os.path.join("codes", "df")
        dfPath = os.path.join(rootPath, self.dfname)

        hackerPath = os.path.join(dfPath, f"{self.dfname}_hacker.csv")
        nodePath = os.path.join(dfPath, f"{self.dfname}_node_classes.csv")
        featuresPath = os.path.join(dfPath, f"{self.dfname}_features.csv")
        seedPath = os.path.join(dfPath, f"{self.dfname}_seed.csv")

        try:
            df_hacker = pd.read_csv(hackerPath)
            df_nodes = pd.read_csv(nodePath)
            df_feature = pd.read_csv(featuresPath)
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"文件未找到：{e.filename}，请检查文件路径和名称")

        return df_nodes, df_feature, df_hacker

    def getTrainAndTestCom(self, df_hacker, min_size=3):  # 修改：接收原始df_hacker作为参数
      
        indices_to_drop = []
        for tag, group_df in df_hacker.groupby("name_tag"):
            if pd.isna(tag) or len(group_df) < min_size:
                indices_to_drop.extend(group_df.index.tolist())
                if not pd.isna(tag):
                    print(f"移除社区 '{tag}': 大小={len(group_df)}")
        

        df_hacker_filtered = df_hacker.drop(indices_to_drop).copy() if indices_to_drop else df_hacker.copy()
        
   
        if df_hacker_filtered.empty:
            raise ValueError("过滤后无有效社区数据，请降低min_size参数")
     
    
        tagList = df_hacker_filtered["name_tag"].unique().tolist()
        train_size = int(0.75 * len(tagList))
        
        trainCom = random.sample(tagList, train_size)
        testCom = list(set(tagList) - set(trainCom))
        
        # 直接生成train_hacker/test_hacker（无需后续重复筛选）
        train_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(trainCom)].copy()
        test_hacker = df_hacker_filtered[df_hacker_filtered["name_tag"].isin(testCom)].copy()
        
        # 修改：返回过滤后的df_hacker + train_hacker + test_hacker
        return df_hacker_filtered, train_hacker, test_hacker

