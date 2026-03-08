import torch
import os
from nn.starter import Starter  # 注意：这里的Starter要使用之前纯字典透传的版本

# 设置工作目录（精简路径计算逻辑）
current_file = os.path.abspath(__file__)
sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
os.chdir(sci_dir)

# 数据集参数配置表（结构化管理，所有参数集中在这里）
DATASET_CONFIGS = {
    "ibm": {
        # 基础参数
        "dfname": "ibm",
        # 数据层面：减少无关节点干扰
        "normal_node_ratio": 3, 
        "expand_hop": 2, 
        "min_community_size": 1,
        # 奖励层面：纯精度惩罚
        "p_bias": 1.0, 
        "min_f1_threshold": 0, 
        "len_penalty_coeff": 0.9,
        # 训练层面：限制扩张+充分训练
        "maxTraLen": 20, 
        "gamma": 0.99, 
        "seedNum": 40, 
        "epoch": 30,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 20,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "elliptic": {
        # 基础参数
        "dfname": "elliptic",
        # Recall偏好配置
        "normal_node_ratio": 2, 
        "expand_hop": 2, 
        "min_community_size": 2,
        "maxTraLen": 100, 
        "p_bias": 0.8, 
        "len_penalty_coeff": 0.99,
        "min_f1_threshold": 0.8, 
        "gamma": 0.99, 
        "seedNum": 40, 
        "epoch": 30,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 20,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    "elliptic2": {
        # 基础参数
        "dfname": "elliptic2",
        # 通用参数
        "normal_node_ratio": 2, 
        "expand_hop": 2, 
        "min_community_size": 5,
        "maxTraLen": 16, 
        "p_bias": 0.3, 
        "len_penalty_coeff": 0.9,
        "min_f1_threshold": 0.2, 
        "gamma": 0.99, 
        "seedNum": 40, 
        "epoch": 30,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 20,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

def train_single_dataset(dfname: str, seed: int = 2026):
    """训练单个数据集，返回测试指标（纯字典传参）"""
    print(f"\n{'='*70}\n🚀 开始训练数据集：{dfname} (种子={seed})\n{'='*70}")
    
    # 检查配置是否存在
    if dfname not in DATASET_CONFIGS:
        print(f"❌ 数据集 {dfname} 无配置参数，终止训练")
        return None
    
    # 直接获取参数字典（核心：不再创建Configure对象）
    params = DATASET_CONFIGS[dfname]
    
    # 初始化Starter（直接传参数字典，不再传Configure）
    starter = Starter(params=params)
    try:
        # 调用run方法（参数完全解耦）
        test_metrics = starter.run(dfname=dfname, seed=seed)
        
        # 打印结果
        f1 = test_metrics['avg_f1'] if test_metrics else "N/A"
        p = test_metrics['avg_precision'] if test_metrics else "N/A"
        r = test_metrics['avg_recall'] if test_metrics else "N/A"
        print(f"\n✅ 数据集 {dfname} 训练完成！")
        print(f"   最终测试集F1：{f1} | P：{p} | R：{r}")
        return test_metrics
    
    except Exception as e:
        print(f"\n❌ 数据集 {dfname} 运行失败：{str(e)}")
        import traceback
        traceback.print_exc()
        print(f"✅ 数据集 {dfname} 训练完成！最终测试集F1：N/A")
        return None

def train_all_datasets(datasets: list, seed: int = 2026):
    """批量训练多个数据集，返回汇总结果"""
    all_results = {}
    for dfname in datasets:
        all_results[dfname] = train_single_dataset(dfname, seed=seed)
    
    # 汇总打印结果
    print(f"\n{'='*70}\n📊 所有数据集训练结果汇总\n{'='*70}")
    print(f"{'数据集':10s} | {'P':6s} | {'R':6s} | {'F1':6s}")
    print(f"{'-'*70}")
    for dfname, metrics in all_results.items():
        if metrics:
            print(f"{dfname:10s} | {metrics['avg_precision']:.4f} | {metrics['avg_recall']:.4f} | {metrics['avg_f1']:.4f}")
        else:
            print(f"{dfname:10s} | {'失败':6s} | {'失败':6s} | {'失败':6s}")
    
    return all_results

if __name__ == "__main__":
    # 可调整训练的数据集列表
    dataset_list = ["elliptic2"]
    final_results = train_all_datasets(dataset_list, seed=2026)