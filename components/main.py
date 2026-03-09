import torch
import os
from nn.starter import Starter  # 注意：这里的Starter要使用之前纯字典透传的版本
from nn.tool import set_seed

# 设置工作目录（精简路径计算逻辑）
current_file = os.path.abspath(__file__)
sci_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
os.chdir(sci_dir)


# # ===================== 基础RL参数 =====================
# # gamma（折扣因子）
# # 公式：G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(T-t-1)*r_{T-1}
# # 代码位置：Expander.__init__ / trainReward折扣奖励计算
# # 调节：γ↑（0.99）→ 重视长期奖励；γ↓（0.9）→ 只看近期
# self.gamma = gamma

# # maxLen（最大轨迹长度）
# # 公式：len(tra_nodes) ≤ maxLen
# # 代码位置：Expander.__init__ / add_node长度判断
# # 调节：maxLen↓→P↑R↓；maxLen↑→P↓R↑
# self.maxLen = maxLen

# # ===================== F1奖励核心参数 =====================
# # f1_base_weight（F1基础权重）
# # 公式：base_reward = biased_f1 * w_base  或  base_reward = -(1-curr_f1)*w_base
# # 代码位置：trainReward奖励计算
# # 调节：w_base↑→奖励放大；w_base↓→奖励缩小
# self.f1_base_weight = f1_base_weight

# # p_bias（精度倾斜系数）
# # 公式：biased_f1 = curr_f1*(1+p_bias) （当curr_p < curr_r时）
# #       biased_f1 = curr_f1            （当curr_p ≥ curr_r时）
# # 代码位置：trainReward奖励计算
# # 调节：p_bias↑→P↑R↓；p_bias↓→P↓R↑；p_bias<0→偏向Recall
# self.p_bias = p_bias

# # min_f1_threshold（最小F1阈值）
# # 公式：if curr_f1 > pre_f1 and curr_f1 > θ → 正奖励；else → 负奖励
# # 代码位置：trainReward奖励判断
# # 调节：θ↑→模型更挑剔（P↑R↓）；θ↓→模型更宽松（P↓R↑）
# self.min_f1_threshold = min_f1_threshold

# # len_penalty_coeff（长度惩罚系数）
# # 公式：base_reward *= α^(curr_len - true_len) （curr_len > true_len时）
# # 代码位置：trainReward长度惩罚计算
# # 调节：α↓（0.7）→ 惩罚重（P↑R↓）；α↑（0.99）→ 惩罚轻（P↓R↑）
# self.len_penalty_coeff = len_penalty_coeff

# # ===================== 损失函数 =====================
# # loss（策略梯度损失）
# # 公式：loss = -Σ(rewards_detach * logps * mask)
# # 代码位置：trainReward梯度计算
# # 逻辑：奖励正→loss小；奖励负→loss大（模型追求正奖励）
# loss = -(rewards_detach * logps * mask).sum()
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
        "p_bias": 0,
        "min_f1_threshold": 0,
        "len_penalty_coeff": 0.9,
        # 训练层面：限制扩张+充分训练
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 50,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 10,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "elliptic": {
        # 基础参数
        "dfname": "elliptic",
        # Recall偏好配置
        "normal_node_ratio": 2,
        "expand_hop": 2,
        "min_community_size": 2,
        "p_bias": 20,
        "len_penalty_coeff": 0.99,
        "min_f1_threshold": 0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 40,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 20,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "elliptic2": {
        # 基础参数
        "dfname": "elliptic2",
        # 通用参数
        "normal_node_ratio": 2,
        "expand_hop": 2,
        "min_community_size": 5,
        "p_bias": 0.3,
        "len_penalty_coeff": 0.9,
        "min_f1_threshold": 0,
        "gamma": 0.99,
        "seedNum": 40,
        "epoch": 30,
        # 其他参数
        "f1_base_weight": 1.0,
        "lr": 1e-4,
        "maxLen": 20,
        "hidden_size": 128,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
}


def train_single_dataset(dfname: str, seed: int = 2026):
    """训练单个数据集，返回测试指标（纯字典传参）"""
    print(f"\n{'='*70}\n🚀 开始训练数据集：{dfname} (种子={seed})\n{'='*70}")
    set_seed(seed=seed)

    print("种子是", seed)
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


    return all_results


if __name__ == "__main__":
    # 可调整训练的数据集列表
    dataset_list = ["elliptic"]
    final_results = train_all_datasets(dataset_list, seed=2026)
