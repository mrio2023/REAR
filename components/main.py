import torch
import os
import itertools
import pandas as pd
from nn.starter import run
from nn.configure import Configure

# 自动跳转到sci目录
current_file = os.path.abspath(__file__)
components_dir = os.path.dirname(current_file)
codes_dir = os.path.dirname(components_dir)
sci_dir = os.path.dirname(codes_dir)
os.chdir(sci_dir)

def train_single_param_set(dfname: str, params: dict, seed: int = 2026):
    """
    单参数组合训练：聚焦p_bias，统一其他次要参数
    """
    print(f"\n{'='*50}")
    print(f"📌 训练 {dfname} | 核心参数：p_bias={params['p_bias']}")
    print(f"{'='*50}")
    
    conf = Configure(dfname=dfname)
    
    # ===================== 统一参数（所有数据集共用） =====================
    conf.device = "cuda" if torch.cuda.is_available() else "cpu"
    conf.f1_base_weight = 1.0
    conf.gamma = 0.90
    conf.seedNum = 50
    conf.epoch = 35
    conf.maxTraLen = 16                # 统一最大步数（各数据集差异小）
    conf.min_community_size = 3       # 统一社区大小（适配所有数据集的小社区）
    
    # ===================== 聚焦核心搜索参数 =====================
    conf.p_bias = params["p_bias"]                # 核心：精度权重（重点搜）
    conf.len_penalty_coeff = params["len_coeff"]  # 辅助：长度惩罚（配合p_bias）
    conf.min_f1_threshold = params["f1_thresh"]   # 辅助：F1底线（配合p_bias）
    
    # ===================== 分数据集微调（仅3个关键参数） =====================
    if dfname == "ibm":
        conf.normal_node_ratio = 5    # IBM：少无关节点干扰
        conf.expand_hop = 1           # IBM：少扩张
    else:
        conf.normal_node_ratio = 3    # elliptic系列：平衡比例
        conf.expand_hop = 2           # elliptic系列：正常扩张
    
    # 运行训练并返回指标
    try:
        test_metrics = run(dfname, conf, seed=seed)
        if test_metrics:
            metrics = {
                "P": round(test_metrics["avg_precision"], 4),
                "R": round(test_metrics["avg_recall"], 4),
                "F1": round(test_metrics["avg_f1"], 4),
                "p_bias": params["p_bias"],
                "len_coeff": params["len_coeff"],
                "f1_thresh": params["f1_thresh"]
            }
            print(f"✅ {dfname} | P={metrics['P']}, R={metrics['R']}, F1={metrics['F1']}")
        else:
            metrics = {"P": None, "R": None, "F1": None, "p_bias": params["p_bias"], "len_coeff": params["len_coeff"], "f1_thresh": params["f1_thresh"]}
            print(f"❌ {dfname} | 训练失败")
        return metrics
    except Exception as e:
        print(f"❌ {dfname} | 报错：{str(e)[:50]}")
        return {"P": None, "R": None, "F1": None, "p_bias": params["p_bias"], "len_coeff": params["len_coeff"], "f1_thresh": params["f1_thresh"]}

def grid_search_focus_pbias(datasets: list, param_grid: dict, seed: int = 2026):
    """
    聚焦p_bias的网格搜索：大幅精简组合数，精准找最优精度权重
    """
    all_results = {dfname: [] for dfname in datasets}
    
    # 生成参数组合（仅聚焦p_bias+2个辅助参数）
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    total_combs = len(param_combinations)
    print(f"\n🚀 聚焦精度权重搜索 | 总组合数：{total_combs}（大幅精简）")
    
    # 遍历每个数据集
    for dfname in datasets:
        print(f"\n{'='*60}")
        print(f"🔍 正在搜索 {dfname} 的最优精度权重...")
        print(f"{'='*60}")
        
        # 遍历所有参数组合
        for idx, comb in enumerate(param_combinations):
            params = dict(zip(param_keys, comb))
            print(f"\n[{idx+1}/{total_combs}] 测试 p_bias={params['p_bias']}")
            metrics = train_single_param_set(dfname, params, seed)
            if metrics["F1"] is not None:
                all_results[dfname].append(metrics)
        
        # 找到当前数据集的最优参数（按F1排序）
        if all_results[dfname]:
            sorted_results = sorted(all_results[dfname], key=lambda x: x["F1"], reverse=True)
            best = sorted_results[0]
            print(f"\n🏆 {dfname} 最优精度配置：")
            print(f"   最优F1={best['F1']} | P={best['P']}, R={best['R']}")
            print(f"   核心参数：p_bias={best['p_bias']}, len_coeff={best['len_coeff']}, f1_thresh={best['f1_thresh']}")
        else:
            print(f"\n❌ {dfname} 无有效训练结果")
    
    # 汇总+保存结果
    print(f"\n{'='*70}")
    print("📊 全数据集最优精度配置汇总")
    print(f"{'='*70}")
    print(f"{'数据集':10s} | 最优P    | 最优R    | 最优F1   | 最优p_bias")
    print(f"{'-'*70}")
    for dfname in datasets:
        if all_results[dfname]:
            best = sorted(all_results[dfname], key=lambda x: x["F1"], reverse=True)[0]
            print(f"{dfname:10s} | {best['P']:.4f} | {best['R']:.4f} | {best['F1']:.4f} | {best['p_bias']}")
        else:
            print(f"{dfname:10s} | 无结果   | 无结果   | 无结果   | 无")
    
    # 保存结果到CSV
    save_results(all_results)
    return all_results

def save_results(all_results: dict):
    """保存聚焦p_bias的搜索结果"""
    save_path = "p_bias_grid_search_results.csv"
    rows = []
    for dfname, results in all_results.items():
        for res in results:
            rows.append({
                "dataset": dfname,
                "p_bias": res["p_bias"],
                "len_coeff": res["len_coeff"],
                "f1_thresh": res["f1_thresh"],
                "P": res["P"],
                "R": res["R"],
                "F1": res["F1"]
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\n💾 精度权重搜索结果已保存至：{save_path}")

if __name__ == "__main__":
    # 1. 定义要搜索的数据集
    dataset_list = ["elliptic", "elliptic2", "ibm"]
    
    # 2. 聚焦p_bias的精简参数网格（仅3个参数，组合数=7*3*3=63组，大幅减少）
    param_grid = {
        "p_bias": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],  # 核心：精度权重（全覆盖）
        "len_coeff": [0.6, 0.7, 0.8],                     # 辅助：长度惩罚（配合p_bias）
        "f1_thresh": [0.35, 0.4, 0.45]                    # 辅助：F1底线（配合p_bias）
    }
    
    # 3. 执行聚焦精度权重的网格搜索
    final_results = grid_search_focus_pbias(dataset_list, param_grid, seed=2026)