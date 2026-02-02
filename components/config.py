class Config:
    # 模型参数
    hidden_dim = 128
    embed_dim = 64
    num_layers = 3
    dropout = 0.2
    lr = 1e-3
    epochs = 100
    weight_decay = 1e-5
    
    # 数据参数
    k_ego = 3  # K-ego子图的K值
    time_window = 30  # 总时间窗口（单位：分钟，可调整）
    snapshot_interval = 10  # 快照时间间隔（论文推荐10分钟，拆分总窗口为多个快照）
    cosine_threshold = 0.7  # 重叠社区余弦相似度阈值
    
    # 聚类参数
    clustering_method = 'hierarchical'  # hierarchical / dbscan
    dbscan_eps = 0.5
    dbscan_min_samples = 5