import torch
import os
from nn.starter import run
from nn.configure import Configure

# 自动跳转到sci目录
current_file = os.path.abspath(__file__)
components_dir = os.path.dirname(current_file)
codes_dir = os.path.dirname(components_dir)
sci_dir = os.path.dirname(codes_dir)
os.chdir(sci_dir)

def train_single_dataset(dfname: str, seed: int = 2026):
    """
    训练单个数据集的函数（IBM 纯精度导向：极致优先精度，放弃部分召回）
    """
    print(f"\n{'='*70}")
    print(f"🚀 开始训练数据集：{dfname} (种子={seed})")
    print(f"{'='*70}")
    
    conf = Configure(dfname=dfname)
    
    # ===================== IBM 纯精度导向参数（核心调整） =====================
    if dfname == "ibm":
        # 1. 数据层面：极致减少无关节点干扰
        conf.normal_node_ratio = 3        # 正常节点=黑客×3（比8更少，几乎只留黑客相关节点）
        conf.expand_hop = 0               # 0跳扩张：只保留种子黑客节点，不扩任何邻居（彻底避免无关节点）
        conf.min_community_size = 1       # 保留所有社区（哪怕只有1个节点，不过滤任何黑客）
        
        # 2. 奖励层面：纯精度惩罚，F1完全向精度倾斜
        conf.p_bias = 1.0                 # 精度倾斜拉满（奖励只看精度，召回权重≈0）
        conf.min_f1_threshold = 0.5       # F1底线提到0.5，精度不够直接扣光奖励
        conf.len_penalty_coeff = 0.5      # 长度惩罚拉满（扩张1步奖励折半，逼模型不扩张）
        
        # 3. 训练层面：限制扩张步数+充分训练
        conf.maxTraLen = 2                # 最多扩2步（几乎等于不扩）
        conf.gamma = 0.90                 # 只关注当前步的精度，完全忽略长期召回
        conf.seedNum = 60                 # 更多种子覆盖所有分散/重叠社区
        conf.epoch = 40                   # 更多轮数让模型收敛到“精准找节点”
        
    else:
        # elliptic/elliptic2 通用参数（保留原配置）
        conf.normal_node_ratio = 2
        conf.expand_hop = 2
        conf.min_community_size = 5
        conf.maxTraLen = 16
        conf.p_bias = 0.6
        conf.len_penalty_coeff = 0.8
        conf.min_f1_threshold = 0.35
        conf.gamma = 0.99
        conf.seedNum = 40
        conf.epoch = 30
    
    # 公共参数
    conf.device = "cuda" if torch.cuda.is_available() else "cpu"
    conf.f1_base_weight = 1.0            # F1基础权重不变，但p_bias=1.0时F1≈精度
    
    # 运行训练和评估
    try:
        test_metrics = run(dfname, conf, seed=seed)
        if test_metrics:
            print(f"\n✅ 数据集 {dfname} 训练完成！")
            print(f"   最终测试集F1：{test_metrics['avg_f1']:.4f}")
            print(f"   最终测试集P：{test_metrics['avg_precision']:.4f}")
            print(f"   最终测试集R：{test_metrics['avg_recall']:.4f}")
        else:
            print(f"\n✅ 数据集 {dfname} 训练完成！最终测试集F1：N/A")
        return test_metrics
    except Exception as e:
        print(f"\n❌ 运行失败：{e}")
        import traceback
        traceback.print_exc()
        print(f"✅ 数据集 {dfname} 训练完成！最终测试集F1：N/A")
        return None

def train_all_datasets(datasets: list, seed: int = 2026):
    all_results = {}
    for dfname in datasets:
        metrics = train_single_dataset(dfname, seed=seed)
        all_results[dfname] = metrics
    
    # 汇总打印
    print(f"\n{'='*70}")
    print("📊 所有数据集训练结果汇总")
    print(f"{'='*70}")
    print(f"{'数据集':10s} | P      | R      | F1     ")
    print(f"{'-'*70}")
    for dfname, metrics in all_results.items():
        if metrics:
            print(f"{dfname:10s} | {metrics['avg_precision']:.4f} | {metrics['avg_recall']:.4f} | {metrics['avg_f1']:.4f}")
        else:
            print(f"{dfname:10s} | 训练失败")
    
    return all_results

if __name__ == "__main__":
    dataset_list = ["ibm"]
    final_results = train_all_datasets(dataset_list, seed=2026)