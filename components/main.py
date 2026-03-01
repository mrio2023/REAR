import torch
from nn.starter import run
from nn.configure import Configure

def train_single_dataset(dfname: str, seed: int = 2026):
    """
    训练单个数据集的函数（封装配置和运行逻辑）
    Args:
        dfname: 数据集名称（elliptic/elliptic2/ibm）
        seed: 固定随机种子
    """
    print(f"\n{'='*70}")
    print(f"🚀 开始训练数据集：{dfname} (种子={seed})")
    print(f"{'='*70}")
    
    # 初始化该数据集的配置
    conf = Configure(dfname=dfname)
    conf.normal_node_ratio = 1
    conf.expand_hop = 1
    conf.min_community_size = 5
    conf.device = "cuda" if torch.cuda.is_available() else "cpu"
    conf.maxTraLen = 16
    conf.gamma = 0.99
    conf.min_reward_threshold = 0.001
    conf.invalid_penalty = 0.02
    conf.seedNum = 40  # 每轮训练种子数
    conf.epoch = 30    # 训练轮数
    
    # 运行训练和评估
    try:
        test_metrics = run(dfname, conf, seed=seed)
        # 保存该数据集的评估结果（可选）
        print(f"\n✅ 数据集 {dfname} 训练完成！最终测试集F1：{test_metrics['avg_f1'] if test_metrics else 'N/A'}")
        return test_metrics
    except Exception as e:
        print(f"\n❌ 数据集 {dfname} 训练失败：{e}")
        return None

def train_all_datasets(datasets: list, seed: int = 2026):
    """
    总函数：批量训练多个数据集
    Args:
        datasets: 数据集名称列表
        seed: 全局固定种子
    Returns:
        所有数据集的评估结果字典
    """
    # 存储所有数据集的结果
    all_results = {}
    
    # 遍历每个数据集训练
    for dfname in datasets:
        metrics = train_single_dataset(dfname, seed=seed)
        all_results[dfname] = metrics
    
    # 打印汇总结果
    print(f"\n{'='*70}")
    print("📊 所有数据集训练结果汇总")
    print(f"{'='*70}")
    for dfname, metrics in all_results.items():
        if metrics:
            print(f"{dfname:10s} | P={metrics['avg_precision']:.4f} | R={metrics['avg_recall']:.4f} | F1={metrics['avg_f1']:.4f} | F2={metrics['avg_f2']:.4f}")
        else:
            print(f"{dfname:10s} | 训练失败")
    
    return all_results

if __name__ == "__main__":
    # 定义要训练的数据集列表
    dataset_list = ["elliptic", "elliptic2", "ibm"]
    # 调用总函数开始批量训练
    final_results = train_all_datasets(dataset_list, seed=2026)