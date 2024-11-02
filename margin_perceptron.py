import math
import os
import logging
from utils.logger import setup_logging
from utils.argparser import argparser
from utils.dataset_loader import read_file, parse_dataset

class MarginPerceptron:
    def __init__(self, d: int, gamma: float = 100.0):
        # y = w * x + b
        self.w = [0.0] * d  # Initialize weight vector -> 0 vector
        self.b = 0.0        # Initialize bias = 0
        self.gamma = gamma  # Margin parameter

    def dot_product(self, v1: list, v2: list) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    def vector_add(self, v1: list, v2: list, scalar: float = 1.0) -> list:
        return [x + scalar * y for x, y in zip(v1, v2)]

    def norm(self, v: list) -> float:
        return math.sqrt(sum(x * x for x in v))

    def train(self, X, y, max_iter=1000):
        R_max = self.find_max_distance_to_origin(X)
        
        for iteration in range(max_iter):
            logging.info(f"Iteration {iteration + 1}:\n")
            violation_found = False

            for xi, yi in zip(X, y):
                # 计算到分离平面的距离 d = w * x + b
                distance = self.dot_product(self.w, xi) + self.b
                
                # 判断是否存在违规点: 距离 + 分类错误
                if abs(distance) < self.gamma / 2 or (yi == 1 and distance < 0) or (yi == -1 and distance > 0):
                    violation_found = True
                    logging.info(f"\t\u2716Violation Point Found: {xi}, Label: {yi}, Distance: {distance:.4f}")

                    # 调整权重
                    if yi == 1:
                        self.w = self.vector_add(self.w, xi)  # w += p
                    else:
                        self.w = self.vector_add(self.w, xi, scalar=-1)  # w -= p

                    # 输出当前的权重和偏置
                    logging.info(f"\t\t Updated: w={self.w}, b={self.b:.4f}, Distance: {distance:.4f}\n")

            if not violation_found:
                logging.info("\t\u2714 No violation points found, training finished.\n")
                break  # 结束训练，如果没有违规点
    
    def find_max_distance_to_origin(self, X) -> float:
        max_distance = 0
        for xi in X:
            # 计算 xi 到原点的欧几里得距离
            distance = sum(x ** 2 for x in xi) ** 0.5
            if distance > max_distance:
                max_distance = distance
        return max_distance

    # 计算最小的 margin: gamma = min{ |wx + b|/||w| }       
    def calculate_margin(self, X, y) -> float:
        if self.norm(self.w) == 0:
            return float('inf')
        margins = [abs(self.dot_product(self.w, xi) + self.b) / self.norm(self.w) for xi in X]
        return min(margins)



if __name__ == '__main__':

    dataset_path = argparser()

    # set logger
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    setup_logging(dataset_name)
    
    logging.info(f"Dataset Path: {dataset_path}")

    # Load Dataset
    lines = read_file(dataset_path)
    instance_dim, num_points, r, data_points = parse_dataset(lines)

    logging.info(f"Number of Points: {num_points} \nDimension of Instance Space: {instance_dim} \nRadius: {r}\n")

    # 定义 gamma 的候选值范围
    gamma_values = [0.1, 1.0, 10.0, 100.0, 1900.0, 2000.0, 2100.0, 2200.0, 2300.0]
    best_gamma = None
    best_margin = -1

    for gamma in gamma_values:
        perceptron = MarginPerceptron(instance_dim, gamma=gamma)
        x, y = [], []
        for point in data_points:
            x.append(point[0])
            y.append(point[1])

        perceptron.train(x, y)
        margin = perceptron.calculate_margin(x, y)
        logging.info(f"Gamma: {gamma}, Margin: {margin:.4f}")

        if margin > best_margin:
            best_margin = margin
            best_gamma = gamma

    logging.info(f"Optimal Gamma: {best_gamma}, Optimal Margin: {best_margin:.4f}")