# 替换原来的 from dataProcess import DataProcess
import sys
import os
import numpy as np
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch import optim

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 将项目根目录加入Python路径（关键）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 导入自定义模块
from codes.components.nn.dataProcess import DataProcess
from codes.components.nn.ellipticDataProcess import ellipticDataProcess
from codes.components.nn.ellipticGraph import ellipticGraph
from codes.components.nn.graph import Graph
from codes.components.nn.gnn import GNN
from codes.components.nn.Agent import Agent
from codes.components.nn.expander import Expander

def plot_training_loss(history, dfname, save_path="./train_loss.png"):
    """
    可视化训练损失曲线
    :param history: 训练历史字典
    :param dfname: 数据集名称
    :param save_path: 图片保存路径
    """
    epochs = range(1, len(history['loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['loss'], 'b-', linewidth=2, label='训练损失')
    plt.axhline(y=np.mean(history['loss']), color='r', linestyle='--', label=f'平均损失: {np.mean(history["loss"]):.4f}')
    
    plt.title(f'{dfname} 数据集训练损失曲线', fontsize=14)
    plt.xlabel('训练轮数 (Epoch)', fontsize=12)
    plt.ylabel('损失值 (Loss)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 训练损失曲线已保存至: {save_path}")

def plot_test_metrics(pred_results, dfname, save_path="./test_metrics.png"):
    """
    可视化测试集评估指标（Precision/Recall/F1/Jaccard）
    :param pred_results: 测试结果字典
    :param dfname: 数据集名称
    :param save_path: 图片保存路径
    """
    # 提取指标数据
    seeds = [f"种子{i+1}" for i in range(len(pred_results['seed_node']))]
    precision = pred_results['precision']
    recall = pred_results['recall']
    f1 = pred_results['f1']
    jaccard = pred_results['jaccard']
    
    # 绘制子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dfname} 数据集测试集评估指标', fontsize=16)
    
    # 1. 精确率
    axes[0,0].bar(seeds, precision, color='skyblue', alpha=0.8)
    axes[0,0].axhline(y=np.mean(precision), color='r', linestyle='--', label=f'平均: {np.mean(precision):.4f}')
    axes[0,0].set_title('精确率 (Precision)', fontsize=12)
    axes[0,0].set_ylabel('值', fontsize=10)
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # 2. 召回率
    axes[0,1].bar(seeds, recall, color='lightgreen', alpha=0.8)
    axes[0,1].axhline(y=np.mean(recall), color='r', linestyle='--', label=f'平均: {np.mean(recall):.4f}')
    axes[0,1].set_title('召回率 (Recall)', fontsize=12)
    axes[0,1].set_ylabel('值', fontsize=10)
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # 3. F1分数
    axes[1,0].bar(seeds, f1, color='orange', alpha=0.8)
    axes[1,0].axhline(y=np.mean(f1), color='r', linestyle='--', label=f'平均: {np.mean(f1):.4f}')
    axes[1,0].set_title('F1分数 (F1-Score)', fontsize=12)
    axes[1,0].set_xlabel('测试种子节点', fontsize=10)
    axes[1,0].set_ylabel('值', fontsize=10)
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # 4. Jaccard相似度
    axes[1,1].bar(seeds, jaccard, color='purple', alpha=0.8)
    axes[1,1].axhline(y=np.mean(jaccard), color='r', linestyle='--', label=f'平均: {np.mean(jaccard):.4f}')
    axes[1,1].set_title('Jaccard相似度', fontsize=12)
    axes[1,1].set_xlabel('测试种子节点', fontsize=10)
    axes[1,1].set_ylabel('值', fontsize=10)
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 测试指标图已保存至: {save_path}")

def plot_community_size(pred_results, dfname, save_path="./community_size.png"):
    """
    可视化真实社区 vs 预测社区的节点数量对比
    :param pred_results: 测试结果字典
    :param dfname: 数据集名称
    :param save_path: 图片保存路径
    """
    # 提取节点数量
    seeds = [f"种子{i+1}" for i in range(len(pred_results['seed_node']))]
    true_size = [len(c) for c in pred_results['true_community']]
    pred_size = [len(c) for c in pred_results['pred_community']]
    
    # 绘制对比柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(seeds))
    width = 0.35
    
    plt.bar(x - width/2, true_size, width, label='真实社区节点数', color='royalblue', alpha=0.8)
    plt.bar(x + width/2, pred_size, width, label='预测社区节点数', color='tomato', alpha=0.8)
    
    plt.title(f'{dfname} 数据集社区节点数量对比', fontsize=14)
    plt.xlabel('测试种子节点', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.xticks(x, seeds)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 社区节点数量对比图已保存至: {save_path}")

def run(dfname: str, seedNum:int, epochs: int = 50, test_seed_num: int = 10):
    print(f"加载 {dfname} 训练集数据...")
    if dfname != "elliptic_txs":
        # 非椭圆数据集：加载训练集
        d_train = DataProcess(dfname=dfname)
        # 构建训练子图（仅训练集数据）
        train_g = Graph(
            dffeature=d_train.train_feature,
            dfhacker=d_train.train_hacker,
            dfnode=d_train.train_nodes
        )
    else:
        # 椭圆数据集：加载训练集
        d_train = ellipticDataProcess()
        # 构建训练子图（仅训练集数据）
        train_g = ellipticGraph(
            dfedge=d_train.train_edge,
            dffeature=d_train.train_feature,
            dfnode=d_train.train_nodes,
            dfhacker=d_train.train_hacker
        )

    # ===================== 2. 训练阶段（仅训练子图） =====================
    # 初始化模型和优化器（绑定训练子图）
    model = Agent(input_size=train_g.embedsize, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    e_train = Expander(graph=train_g, optimizer=optimizer, model=model)

    # 训练历史记录
    history = {'loss': [], 'train_f1': []}

    # 训练循环
    print(f"\n开始在训练子图上训练 {dfname} 数据集，共 {epochs} 轮...")
    for epoch in range(1, epochs + 1):
        # 仅从训练集采样种子节点
        train_seeds = random.sample(train_g.df_hacker["address"].values.tolist(), k=seedNum)
        isweak = []
        truecom = []
        
        # 在训练子图上生成轨迹
        for s in train_seeds:
            c, w = train_g.sampleTrajectory(s)
            truecom.append(c)
            isweak.append(w)
        
        # 训练并记录损失
        loss = e_train.trainReward(seeds=train_seeds, true_coms=truecom, isweak=isweak)
        history['loss'].append(loss)
        
        # 每10轮打印进度
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{epochs}] | Training Loss: {loss:.4f}")

    # ===================== 3. 加载全量数据，构建完整大图（训练+测试） =====================
    print(f"\n加载 {dfname} 全量数据（训练+测试），构建完整大图...")
    if dfname != "elliptic_txs":
        # 非椭圆数据集：重新加载全量数据（合并训练+测试）
        d_full = DataProcess(dfname=dfname)
        # 合并训练+测试数据构建完整大图
        full_feature = pd.concat([d_full.train_feature, d_full.test_feature], ignore_index=True).drop_duplicates()
        full_nodes = pd.concat([d_full.train_nodes, d_full.test_nodes], ignore_index=True).drop_duplicates()
        full_hacker = pd.concat([d_full.train_hacker, d_full.test_hacker], ignore_index=True).drop_duplicates()
        
        full_g = Graph(
            dffeature=full_feature,
            dfhacker=full_hacker,
            dfnode=full_nodes
        )
        # 提取测试集种子节点（仅地址）
        test_seeds_pool = d_full.test_hacker["address"].unique().tolist()
    else:
        # 椭圆数据集：重新加载全量数据
        d_full = ellipticDataProcess()
        # 合并训练+测试数据构建完整大图
        full_edge = pd.concat([d_full.train_edge, d_full.test_edge], ignore_index=True).drop_duplicates()
        full_feature = pd.concat([d_full.train_feature, d_full.test_feature], ignore_index=True).drop_duplicates()
        full_nodes = pd.concat([d_full.train_nodes, d_full.test_nodes], ignore_index=True).drop_duplicates()
        full_hacker = pd.concat([d_full.train_hacker, d_full.test_hacker], ignore_index=True).drop_duplicates()
        
        full_g = ellipticGraph(
            dfedge=full_edge,
            dffeature=full_feature,
            dfnode=full_nodes,
            dfhacker=full_hacker
        )
        # 提取测试集种子节点（仅地址）
        test_seeds_pool = d_full.test_hacker["address"].unique().tolist()

    # ===================== 4. 测试阶段（完整大图 + 测试集种子） =====================
    print("\n" + "="*80)
    print("测试阶段：完整大图 + 测试集种子节点")
    print("="*80)
    
    # 切换模型为评估模式（禁用Dropout/BatchNorm等）
    model.eval()
    
    # 初始化测试用Expander（绑定完整大图，复用训练好的模型）
    e_test = Expander(graph=full_g, optimizer=optimizer, model=model)
    
    # 采样测试集种子节点（保证数量足够）
    if len(test_seeds_pool) < test_seed_num:
        test_seed_num = len(test_seeds_pool)
        print(f"测试集种子节点不足，自动调整为 {test_seed_num} 个")
    
    test_seeds = random.sample(test_seeds_pool, k=test_seed_num)
    
    # 评估指标存储
    pred_results = {
        "seed_node": [],
        "pred_community": [],
        "true_community": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "jaccard": []
    }
    
    # 禁用梯度计算（加速测试，避免内存泄漏）
    with torch.no_grad():
        # 对每个测试种子节点评估
        for idx, seed in enumerate(test_seeds):
            # 1. 完整大图上生成真实社区轨迹
            true_com, _ = full_g.sampleTrajectory(seed)
            true_com_set = set(true_com)
            
            # 2. 模型预测社区（完整大图上扩展）
            pred_tra, _ = e_test.sample_bs_trajectories([seed])  # 单种子预测
            pred_com = [node for node in pred_tra[0] if node != "Stp" and node in full_g.deMap]
            pred_com_set = set(pred_com)
            
            # 3. 计算评估指标（避免除零错误）
            tp = len(true_com_set & pred_com_set)  # 真阳性
            fp = len(pred_com_set - true_com_set)  # 假阳性
            fn = len(true_com_set - pred_com_set)  # 假阴性
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            jaccard = tp / (len(true_com_set | pred_com_set)) if len(true_com_set | pred_com_set) > 0 else 0.0
            
            # 记录结果
            pred_results["seed_node"].append(seed)
            pred_results["pred_community"].append(pred_com)
            pred_results["true_community"].append(true_com)
            pred_results["precision"].append(precision)
            pred_results["recall"].append(recall)
            pred_results["f1"].append(f1)
            pred_results["jaccard"].append(jaccard)
    
    # ===================== 5. 输出测试结果 =====================
    # 计算平均指标
    avg_precision = np.mean(pred_results["precision"])
    avg_recall = np.mean(pred_results["recall"])
    avg_f1 = np.mean(pred_results["f1"])
    avg_jaccard = np.mean(pred_results["jaccard"])
    
    # 打印详细结果
    print("\n【测试集评估结果汇总】")
    print(f"测试种子数：{test_seed_num}")
    print(f"平均精确率(Precision)：{avg_precision:.4f}")
    print(f"平均召回率(Recall)：{avg_recall:.4f}")
    print(f"平均F1分数：{avg_f1:.4f}")
    print(f"平均Jaccard相似度：{avg_jaccard:.4f}")
    
    # 打印单个种子的示例（前3个）
    print("\n【前3个测试种子详情】")
    for i in range(min(3, len(test_seeds))):
        print(f"\n种子节点 {i+1}：{pred_results['seed_node'][i]}")
        print(f"  真实社区节点数：{len(pred_results['true_community'][i])}")
        print(f"  预测社区节点数：{len(pred_results['pred_community'][i])}")
        print(f"  F1分数：{pred_results['f1'][i]:.4f}")

    # 训练总结
    print("\n【训练总结】")
    avg_train_loss = np.mean(history['loss'])
    print(f"总训练轮数：{epochs}")
    print(f"平均训练损失：{avg_train_loss:.4f}")

    # ===================== 6. 可视化结果（新增核心部分） =====================
    print("\n" + "="*80)
    print("生成可视化结果...")
    print("="*80)
    # 1. 训练损失曲线
    plot_training_loss(history, dfname)
    # 2. 测试指标对比图
    plot_test_metrics(pred_results, dfname)
    # 3. 社区节点数量对比图
    plot_community_size(pred_results, dfname)

    return history, pred_results

