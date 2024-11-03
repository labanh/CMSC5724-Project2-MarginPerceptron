import math
import os
import logging
import matplotlib.pyplot as plt
from utils.logger import setup_logging
from utils.argparser import argparser
from utils.dataset_loader import read_file, parse_dataset

class MarginPerceptron:
    def __init__(self, d: int, r: float):
        # y = w * x + b
        self.w = [0.0] * d  # Initialize weight vector -> 0 vector
        self.b = 0.0        # Initialize bias = 0
        self.eta = 0.1      # Learning rate
        self.R_max = r      # Maximum distance to origin
        self.gamma_guess = self.R_max  # R_max -> γ_guess

    def dot_product(self, v1: list, v2: list) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    def vector_add(self, v1: list, v2: list, scalar: float = 1.0) -> list:
        return [x + scalar * y for x, y in zip(v1, v2)]

    def norm(self, v: list) -> float:
        return math.sqrt(sum(x * x for x in v))

    def train(self, X, y):

        terminate_iter = self.terminate_iter()
        logging.info(f"Terminate Iteration: {terminate_iter}\n")

        for iteration in range(terminate_iter):
            logging.info(f"Iteration {iteration + 1}:\n")
            violation_found = False

            for xi, yi in zip(X, y):
                # 计算到分离平面的距离 d = w * x + b
                distance = self.dot_product(self.w, xi) + self.b
                
                # 判断是否存在违规点: 距离 + 分类错误
                if abs(distance) < self.gamma_guess / 2 or (yi == 1 and distance < 0) or (yi == -1 and distance > 0):
                    violation_found = True
                    logging.info(f"\t\u2716 Violation Point Found: {xi}, Label: {yi}, Distance: {distance:.4f}")

                    # 调整权重和偏置
                    if yi == 1:
                        self.w = self.vector_add(self.w, xi, scalar=self.eta)  # w += η * p
                        self.b += self.eta  # b += η
                    else:
                        self.w = self.vector_add(self.w, xi, scalar=-self.eta)  # w -= η * p
                        self.b -= self.eta  # b -= η

                    # 输出当前的权重和偏置
                    logging.info(f"\t\t Updated: w={self.w}, b={self.b:.4f}, Distance: {distance:.4f}\n")
            
            # Self-Termination
            if not violation_found:
                logging.info("\t\u2714 No violation points found, training finished.\n")
                return  # 结束训练，如果没有违规点

            # Forced-Termination
            if iteration == terminate_iter - 1:
                logging.info("\t\u26A0 Training reached maximum iterations.\n")
                self.gamma_guess /= 2
                logging.info(f"\t\t Updated: γ_guess={self.gamma_guess:.4f}\n")
                iteration = 0  # 重新开始训练
    
    # 计算数据点到原点的最大距离，等于数据中给出的 R
    def find_max_distance_to_origin(self, X) -> float:
        max_distance = 0
        for xi in X:
            # 计算 xi 到原点的欧几里得距离
            distance = sum(x ** 2 for x in xi) ** 0.5
            if distance > max_distance:
                max_distance = distance
        return max_distance

    # Calculate the Terminate Iteration: 12R^2/γ^2
    def terminate_iter(self) -> int:
        return int(12.0 * (self.R_max **2)/(gamma_guess ** 2))

    # 计算最小的 margin: gamma_opt = min{ |wx + b|/||w| }       
    def calculate_margin(self, X, y) -> float:
        if self.norm(self.w) == 0:
            return float('inf')
        margins = [abs(self.dot_product(self.w, xi) + self.b) / self.norm(self.w) for xi in X]
        return min(margins)
    
    def return_w(self):
        return self.w
    
    def return_b(self):
        return self.b

def plot_2d_points_and_decision_boundary(X, y, w, b):
    # 绘制数据点，不同类别使用不同颜色
    for xi, yi in zip(X, y):
        if yi == 1:
            plt.scatter(xi[0], xi[1], color='blue', s=0.1, label='Class 1' if 'Class 1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(xi[0], xi[1], color='red', s=0.1, label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # 绘制分类线
    x_min = min(point[0] for point in X) - 1
    x_max = max(point[0] for point in X) + 1
    x_values = [x_min + i * (x_max - x_min) / 100 for i in range(101)]

    # 计算 y_values
    y_values = [-(w[0] * x + b) / w[1] for x in x_values]

    plt.plot(x_values, y_values, color='green', label='Decision Boundary')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('2D Points and Decision Boundary')
    plt.legend()
    plt.grid(True)
    # 保存图像到指定路径
    plt.savefig('2d_points_and_decision_boundary.png')
    plt.close()  # 关闭图像，防止在后续绘图时重叠

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

    # 定义 gamma_guess 的候选值范围
    # gamma_guess_values = [0.1, 1.0, 10.0, 100.0, 1900.0, 2000.0, 2100.0, 2200.0, 2300.0]
    gamma_guess_values = [1.0]
    best_gamma_guess = None
    best_margin = -1

    for gamma_guess in gamma_guess_values:
        perceptron = MarginPerceptron(instance_dim, r)
        x, y = [], []
        for point in data_points:
            x.append(point[0])
            y.append(point[1])

        perceptron.train(x, y)
        margin = perceptron.calculate_margin(x, y)
        logging.info(f"gamma_guess: {gamma_guess}, Margin: {margin:.4f}\n")

        if margin > best_margin:
            best_margin = margin
            best_gamma_guess = gamma_guess

    w = perceptron.return_w()
    b = perceptron.return_b()
    logging.info(f"Optimal gamma_guess: {best_gamma_guess}, Optimal Margin: {best_margin:.4f}\n")
    logging.info(f"w: {w}, b: {b}")

    # plot_2d_points_and_decision_boundary(x, y, w, b)