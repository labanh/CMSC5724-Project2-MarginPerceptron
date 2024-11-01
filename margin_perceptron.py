import numpy as np

class MarginPerceptron:
    def __init__(self, d, eta=0.1, gamma=1.0):
        """
        初始化 Margin Perceptron 参数
        :param d: 特征空间的维度
        :param eta: 学习率
        :param gamma: margin 参数
        """
        self.w = np.zeros(d)  # 权重向量初始化为零向量
        self.b = 0.0          # 偏置初始化为 0
        self.eta = eta        # 学习率
        self.gamma = gamma    # margin 参数

    def train(self, X, y, max_iter=1000):
        """
        训练 Margin Perceptron 模型
        :param X: 输入特征矩阵，shape 为 (n_samples, d)
        :param y: 标签数组，shape 为 (n_samples,)
        :param max_iter: 最大迭代次数
        """
        for _ in range(max_iter):
            for xi, yi in zip(X, y):
                # 计算得分
                score = yi * (np.dot(self.w, xi) + self.b)
                if score < self.gamma:
                    # 更新权重和偏置
                    self.w += self.eta * yi * xi
                    self.b += self.eta * yi

    def predict(self, X):
        """
        使用模型进行预测
        :param X: 输入特征矩阵
        :return: 预测标签数组
        """
        return np.sign(np.dot(X, self.w) + self.b)

    def calculate_margin(self):
        """
        计算当前模型的 margin
        :return: margin 值
        """
        return self.gamma / np.linalg.norm(self.w)
