import torch


class Configure:
    def __init__(
        self,
        dfname,
        normal_node_ratio: float = 0.2,#生成训练集混景区的点的比例
        expand_hop: int = 2,#生成训练集初始节点向外扩展几条
        min_community_size: int = 10,#过滤社区的大小
        maxTraLen:int =16,#扩展社区的上限
        gamma: float = 0.99,#RL的折扣数
        min_reward_threshold: float = 0.001,  # 【新增】最小有效奖励阈值
        invalid_penalty: float = 0.01,  # 【新增】无效节点惩罚值
        reward_weight_abs: float = 0.7,  # 【新增】F2绝对值权重
        reward_weight_delta: float = 0.3,  # 【新增】F2增量权重
        len_penalty_base: float = 0.99,
        # 可选：允许手动指定设备，默认自动检测
        device: str | torch.device = None,
        seedNum:int=50#一次采样多少种子
    ):
        # 原有参数保存
        self.dfname = dfname
        self.normal_node_ratio = normal_node_ratio
        self.expand_hop = expand_hop
        self.min_community_size = min_community_size
        self.gamma = gamma
        self.min_reward_threshold = min_reward_threshold
        self.invalid_penalty = invalid_penalty
        self.reward_weight_abs = reward_weight_abs
        self.reward_weight_delta = reward_weight_delta
        self.maxTraLen=maxTraLen
        self.seedNum=seedNum
        self. len_penalty_base=len_penalty_base

        # 新增：自动检测 CUDA 设备（核心逻辑）
        if device is None:  # 如果未手动指定，自动检测
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")  # 优先用第0块GPU
                # 可选：打印GPU信息，方便验证
                print(
                    f"自动检测到 CUDA 可用，使用设备: {torch.cuda.get_device_name(0)}"
                )
            else:
                self.device = torch.device("cpu")
                print("未检测到 CUDA，使用 CPU 设备")
        else:
            # 如果手动指定了设备（如 "cuda:1"、"cpu"），则使用指定的
            self.device = torch.device(device)
            print(f"使用手动指定的设备: {self.device}")
