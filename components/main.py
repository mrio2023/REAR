import torch
import os  # 新增：导入os模块
from nn.starter import run
from nn.configure import Configure

# ===================== 核心修改：自动识别并跳转到sci目录 =====================
current_file = os.path.abspath(__file__)
components_dir = os.path.dirname(current_file)
codes_dir = os.path.dirname(components_dir)
sci_dir = os.path.dirname(codes_dir)
os.chdir(sci_dir)

def train_single_dataset(dfname: str, seed: int = 2026):
    """
    训练单个数据集的函数（封装配置和运行逻辑）
    新增：打印AUC-PR、Recall@Top1%/Top5%指标
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
    conf.seedNum = 2  # 每轮训练种子数
    conf.epoch = 3    # 训练轮数
    
    # 运行训练和评估
    try:
        test_metrics = run(dfname, conf, seed=seed)
        # 核心修改：新增AUC-PR/Recall@TopK的打印
        if test_metrics:
            print(f"\n✅ 数据集 {dfname} 训练完成！")
            print(f"   最终测试集F1：{test_metrics['avg_f1']:.4f}")
            print(f"   最终测试集AUC-PR：{test_metrics['avg_auc_pr']:.4f}")
            print(f"   最终测试集Recall@Top1%：{test_metrics['avg_recall_top1']:.4f}")
            print(f"   最终测试集Recall@Top5%：{test_metrics['avg_recall_top5']:.4f}")
        else:
            print(f"\n✅ 数据集 {dfname} 训练完成！最终测试集F1：N/A")
        return test_metrics
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        print(f"✅ 数据集 {dfname} 训练完成！最终测试集F1：N/A")
        return None

def train_all_datasets(datasets: list, seed: int = 2026):
    """
    总函数：批量训练多个数据集
    新增：汇总AUC-PR、Recall@Top1%/Top5%指标
    """
    # 存储所有数据集的结果
    all_results = {}
    
    # 遍历每个数据集训练
    for dfname in datasets:
        metrics = train_single_dataset(dfname, seed=seed)
        all_results[dfname] = metrics
    
    # 核心修改：新增AUC-PR/Recall@TopK的汇总打印
    print(f"\n{'='*70}")
    print("📊 所有数据集训练结果汇总")
    print(f"{'='*70}")
    # 打印表头（新增列）
    print(f"{'数据集':10s} | P      | R      | F1     | AUC-PR | Top1%  | Top5%")
    print(f"{'-'*70}")
    for dfname, metrics in all_results.items():
        if metrics:
            print(f"{dfname:10s} | {metrics['avg_precision']:.4f} | {metrics['avg_recall']:.4f} | {metrics['avg_f1']:.4f} | {metrics['avg_auc_pr']:.4f} | {metrics['avg_recall_top1']:.4f} | {metrics['avg_recall_top5']:.4f}")
        else:
            print(f"{dfname:10s} | 训练失败")
    
    return all_results

if __name__ == "__main__":
    # 定义要训练的数据集列表
    dataset_list = ["elliptic"]
    # 调用总函数开始批量训练
    final_results = train_all_datasets(dataset_list, seed=2026)