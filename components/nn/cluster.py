import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import squareform, pdist
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# ========== 核心：纯Python实现KMedoids（适配numpy2.0+sklearn1.7，无Cython依赖） ==========
class PurePythonKMedoids:
    def __init__(self, n_clusters=2, metric="precomputed", random_state=0):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.labels_ = None  # 兼容原有代码的labels_属性
        self.medoid_indices_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        
        # 1. 处理距离矩阵（和原有逻辑一致，支持precomputed）
        if self.metric == "precomputed":
            distance_matrix = X  # 输入是距离矩阵（1-simi）
        else:
            raise ValueError("仅支持precomputed距离矩阵（和原有KMedoids逻辑一致）")
        
        n_samples = distance_matrix.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(f"样本数{n_samples}小于聚类数{self.n_clusters}")
        
        # 2. 随机初始化聚类中心
        self.medoid_indices_ = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # 3. 迭代更新（和原有KMedoids逻辑对齐）
        max_iter = 300
        for _ in range(max_iter):
            # 分配样本到最近的聚类中心
            self.labels_ = np.argmin(distance_matrix[:, self.medoid_indices_], axis=1)
            
            # 更新聚类中心（选簇内总距离最小的点）
            new_medoids = []
            for cluster in range(self.n_clusters):
                cluster_points = np.where(self.labels_ == cluster)[0]
                if len(cluster_points) == 0:
                    # 空簇：随机选一个新中心（兼容边界情况）
                    new_medoids.append(np.random.choice(n_samples))
                    continue
                # 计算簇内总距离，选最小的作为medoid
                cluster_dist = distance_matrix[cluster_points][:, cluster_points]
                total_dist = cluster_dist.sum(axis=1)
                best_medoid = cluster_points[np.argmin(total_dist)]
                new_medoids.append(best_medoid)
            
            # 收敛判断
            if set(self.medoid_indices_) == set(new_medoids):
                break
            self.medoid_indices_ = new_medoids
        
        return self

    def fit_predict(self, X):
        # 兼容原有代码的fit_predict调用方式
        self.fit(X)
        return self.labels_

# ========== 保留你原有Cluster类的所有逻辑，仅替换KMedoids导入 ==========
class Cluster:
    def UsingScSelectCom(self, simi, communities, K=2):
        """
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        """
        # 选在其中的一类
        spectral_clustering = SpectralClustering(n_clusters=K, affinity="precomputed")
        labels = spectral_clustering.fit_predict(simi)
        # 输出聚类结果
        simis, traincom = [], []
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                simis.append(simi[0][i])
                traincom.append(communities[i])
        return traincom

    def UsingKMedoidsSelectCom(self, simi, communities, K=2):
        """
        聚类，并选择包含局部结构的簇（适配numpy2.0，替换sklearn_extra）
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        """
        # 将相似性矩阵转换为距离矩阵（原有逻辑不变）
        distance_matrix = 1 - simi
        # 关键：用纯Python版KMedoids替代sklearn_extra版
        kmedoids = PurePythonKMedoids(n_clusters=K, metric="precomputed", random_state=0)

        # 训练模型（原有逻辑完全不变）
        kmedoids.fit(distance_matrix)

        # 获取聚类标签（原有逻辑完全不变）
        labels = kmedoids.labels_
        print("Cluster labels:", labels)

        # 输出聚类结果（原有逻辑完全不变）
        traincom = []
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom

    def UsingGmmSelectCom(self, simi, communities, K=2):
        """
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        """
        # 将相似性矩阵转换为距离矩阵（原有逻辑不变）
        distance_matrix = squareform(pdist(simi, "euclidean"))

        # 进行高斯混合模型聚类（原有逻辑不变）
        gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0)
        gmm.fit(distance_matrix)
        labels = gmm.predict(distance_matrix)
        print("Cluster labels:", labels)

        # 输出聚类结果（原有逻辑不变）
        traincom = []
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])
        return traincom

    def UsingCengCiSelectCom(self, simi, communities, K=2):
        """
        聚类，并选择包含局部结构的簇
        @param simi: 相似性矩阵
        @param communities: 已知社区+局部结构
        @param K: 聚类系数
        """
        # 将相似性矩阵转换为距离矩阵（原有逻辑不变）
        dist_matrix = 1 - simi
        np.fill_diagonal(dist_matrix, 0)

        linked = linkage(squareform(dist_matrix), "complete")

        # 使用K指定聚类数量（原有逻辑不变）
        labels = fcluster(linked, K, criterion="maxclust")
        print("Cluster labels:", labels)

        # 输出聚类结果（原有逻辑不变）
        traincom = []
        for i in range(1, len(labels)):
            if labels[i] == labels[0]:
                traincom.append(communities[i])

        return traincom