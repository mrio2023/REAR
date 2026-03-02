import torch

class Configure:
    def __init__(
        self,
        dfname,
        normal_node_ratio: float = 0.2,  # 生成训练集混景区的点的比例
        expand_hop: int = 2,  # 生成训练集初始节点向外扩展几条
        min_community_size: int = 10,  # 过滤社区的大小
        maxTraLen: int = 16,  # 扩展社区的上限
        gamma: float = 0.99,  # RL的折扣数
        # F1奖励核心参数（替换原AUC-PR相关参数）
        f1_base_weight: float = 1.0,       # F1基础权重
        p_bias: float = 0.2,               # 精度倾斜系数（P<R时放大F1）
        min_f1_threshold: float = 0.1,     # 最小F1阈值（低于则无奖励）
        len_penalty_coeff: float = 0.95,   # 长度惩罚系数（超过真实社区长度衰减）
        # 可选：允许手动指定设备，默认自动检测
        device: str | torch.device = None,
        seedNum: int = 50  # 一次采样多少种子
    ):
        # 基础数据集/训练参数
        self.dfname = dfname
        self.normal_node_ratio = normal_node_ratio
        self.expand_hop = expand_hop
        self.min_community_size = min_community_size
        self.maxTraLen = maxTraLen
        self.gamma = gamma
        self.seedNum = seedNum

        # F1奖励核心参数（和Expander类完全对应）
        self.f1_base_weight = f1_base_weight
        self.p_bias = p_bias
        self.min_f1_threshold = min_f1_threshold
        self.len_penalty_coeff = len_penalty_coeff

        # 设备自动检测逻辑
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(f"自动检测到 CUDA 可用，使用设备: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("未检测到 CUDA，使用 CPU 设备")
        else:
            self.device = torch.device(device)
            print(f"使用手动指定的设备: {self.device}")