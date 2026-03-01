
from typing import Union, Optional, List, Set
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os  # 新增：用于控制PyTorch CUDA随机性

# 记得转为相对引用
# starter.py 开头（修改后）
from .dataProcess import dataProcess  # 同目录
from .graph import Graph              # 同目录
from .Agent import Agent              # 同目录
from .expander import Expander        # 同目录
from .configure import Configure      # 同目录
import random


def set_seed(seed: int = 42):
    """固定所有随机种子，保证实验可复现"""
    # Python内置随机数
    random.seed(seed)
    # NumPy随机数
    np.random.seed(seed)
    # PyTorch CPU随机数
    torch.manual_seed(seed)
    # PyTorch GPU随机数（单卡）
    torch.cuda.manual_seed(seed)
    # PyTorch GPU随机数（多卡）
    torch.cuda.manual_seed_all(seed)
    # 禁用cuDNN的随机性（保证卷积/池化等操作可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 控制PyTorch数据加载的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 所有随机种子已固定为：{seed}")


def eval_model(expander: Expander, test_g: Graph, test_seeds: List,conf:Configure) -> dict:
    """
    评估模型在测试集上的表现（适配Expander的sample_bs_trajectories方法）
    Args:
        expander: 初始化好的Expander实例（已切换到test_g）
        test_g: 测试集的Graph实例
        test_seeds: 测试集的种子节点列表
    Returns:
        包含所有评估指标平均值的字典
    """
    # 确保模型处于评估模式（锁定参数，禁用dropout等）
    expander.model.eval()
    
    # 存储所有样本的指标
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": [],
        "f2": []
    }
    
    # 禁用梯度计算（加速评估，节省内存）
    with torch.no_grad():
        # 批量预测轨迹（复用Expander的sample_bs_trajectories方法）
        pred_coms, _ = expander.sample_bs_trajectories(test_seeds)
        
        # 遍历每个种子计算指标
        for idx, seed in enumerate(test_seeds):
            # 获取真实社区
            true_com = test_g.sampleTrajectory(seed,traj_length=conf.maxTraLen)
            # 清理预测社区（移除Stp标记）
            pred_com = [node for node in pred_coms[idx] if node != "Stp"]
            
            # 使用Expander内置的指标计算方法
            p, r, f1, j = expander.eval_scores(pred_com, true_com)
            f2 = expander.eval_fbeta(pred_com, true_com, beta=2.0)
            
            # 收集指标
            all_metrics["precision"].append(p)
            all_metrics["recall"].append(r)
            all_metrics["f1"].append(f1)
            all_metrics["jaccard"].append(j)
            all_metrics["f2"].append(f2)
    
    # 计算平均值
    avg_metrics = {
        "avg_precision": round(np.mean(all_metrics["precision"]), 4),
        "avg_recall": round(np.mean(all_metrics["recall"]), 4),
        "avg_f1": round(np.mean(all_metrics["f1"]), 4),
        "avg_jaccard": round(np.mean(all_metrics["jaccard"]), 4),
        "avg_f2": round(np.mean(all_metrics["f2"]), 4),
        "std_f1": round(np.std(all_metrics["f1"]), 4)  # F1标准差，评估稳定性
    }
    
    # 打印评估结果
    print("\n" + "="*60)
    print("📊 模型测试集评估结果")
    print("="*60)
    print(f"测试种子数量：{len(test_seeds)}")
    print(f"平均精度(P)：{avg_metrics['avg_precision']}")
    print(f"平均召回(R)：{avg_metrics['avg_recall']}")
    print(f"平均F1分数：{avg_metrics['avg_f1']} (±{avg_metrics['std_f1']})")
    print(f"平均F2分数：{avg_metrics['avg_f2']}")
    print(f"平均Jaccard系数：{avg_metrics['avg_jaccard']}")
    print("="*60)
    
    return avg_metrics


def run(dfname, conf: Configure, seed: int = 42):
    """
    测试Expander类核心功能（复用真实Graph+elliptic数据集）：
    1. 加载elliptic数据集并初始化Graph
    2. 初始化Expander（真实Graph+Agent+优化器）
    3. 测试trainReward方法（移除isweak后）
    4. 验证损失计算和指标输出
    5. 在测试集上评估模型表现
    """
    # 第一步：固定随机种子（核心修改）
    set_seed(seed)
    
    try:
        # ===================== 1. 加载真实数据并初始化Graph =====================
        # 初始化数据处理类（加载elliptic数据集）
        dp = dataProcess(
            dfname=conf.dfname,
            normal_node_ratio= conf.normal_node_ratio,
            expand_hop = conf.expand_hop,
            min_community_size=conf.min_community_size
        )
        print("✅ dataProcess初始化成功")
        
        print("tes_n",len(dp.test_nodes))
        print("tes_h",len(dp.test_hacker))
        print("tra_n",len(dp.train_nodes))
        print("tra_h",len(dp.train_hacker))

        # 初始化真实Graph类（使用训练集数据）
        g = Graph(
            dfnode=dp.train_nodes,
            dffeature=dp.train_feature,
            dfhacker=dp.train_hacker,
            dfedge=dp.train_edge
        )
        print(f"✅ Graph类初始化成功（基于{dfname}训练集）")

        # ===================== 2. 初始化Expander组件 =====================
        device = torch.device(conf.device) 
        # 初始化Agent模型（输入维度匹配Graph的embedsize）
        model = Agent(input_size=g.embedsize,hidden_size=128).to(device)
        # 初始化优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 初始化Expander
        expander = Expander(
            graph=g,
            model=model,
            optimizer=optimizer,
            device=device,
            maxLen=conf.maxTraLen,  
            gamma=conf.gamma,
            min_reward_threshold=conf.min_reward_threshold,
            invalid_penalty=conf.invalid_penalty
        )
        print("✅ Expander初始化成功")

        # ===================== 3. 训练过程 =====================
        # 从训练集黑客节点中选种子（避免空数据）
        if len(dp.train_hacker) < 2:
            raise ValueError("训练集黑客节点数量不足，无法测试")
        
        origin_seeds = dp.train_hacker["address"].values.tolist()  
        for i in range(conf.epoch):
            # 由于固定了种子，random.sample的结果每次都一致
            seeds=random.sample(origin_seeds,k=conf.seedNum)
            
            true_coms = [g.sampleTrajectory(s,traj_length=conf.maxTraLen) for s in seeds]
          
            print(f"✅ 构造真实社区完成，社区大小：{[len(c) for c in true_coms]}")

            # 测试trainReward方法
            loss, metrics = expander.trainReward(seeds=seeds, true_coms=true_coms)
            print(f"\n===================== 第{i}次训练结果 =====================")
            print(f"✅ trainReward执行成功")
            print(f"   本次训练损失值：{loss:.4f}")
            print(f"   平均精度(P)：{metrics['avg_precision']}")
            print(f"   平均召回(R)：{metrics['avg_recall']}")
            print(f"   平均F1分数：{metrics['avg_f1']}")
            print(f"   平均F2分数：{metrics['avg_f2']}")
            print(f"   平均Jaccard系数：{metrics['avg_jaccard']}")
        
        # ===================== 4. 测试集评估 =====================
        print("\n🔍 开始在测试集上评估模型...")
        # 初始化测试集Graph
        test_g  = Graph(
            dfnode=dp.test_nodes,
            dffeature=dp.test_feature,
            dfhacker=dp.test_hacker,
            dfedge=dp.test_edge
        )

        # 切换Expander到测试集Graph
        expander.graph = test_g
        
        # 准备测试集种子（避免数量不足）
        test_seeds = dp.test_hacker["address"].values.tolist()
        if len(test_seeds) == 0:
            raise ValueError("测试集无黑客节点，无法评估")
        # 限制测试种子数量（避免评估过久）
        test_seeds = test_seeds[:min(conf.seedNum * 2, len(test_seeds))]
        
        # 执行评估
        test_metrics = eval_model(expander, test_g, test_seeds,conf)
        
        return test_metrics

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return None


# # 执行测试
# def test_expander():
#     # 初始化配置（根据你的Configure类调整）
#     conf = Configure(dfname="elliptic")
#     conf.normal_node_ratio = 0.8
#     conf.expand_hop = 2
#     conf.min_community_size = 5
#     conf.device = "cuda" if torch.cuda.is_available() else "cpu"
#     conf.maxTraLen = 10
#     conf.gamma = 0.99
#     conf.min_reward_threshold = 0.001
#     conf.invalid_penalty = 0.01
#     conf.seedNum = 2  # 每轮训练种子数
#     conf.epoch = 5    # 训练轮数
    
#     # 运行测试和评估（指定固定种子）
#     run("elliptic", conf, seed=42)  # 可自定义种子值，比如100、2024等

# if __name__ == "__main__":
#     test_expander()