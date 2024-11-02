import math
import logging
import os
from utils.argparser import argparser
from utils.dataset_loader import *

class MarginPerceptron:
    def __init__(self, d: int, eta: float = 0.1, gamma: float = 1.0):
        # y = w * x + b
        self.w = [0.0] * d  # Initialize weight vector -> 0 vector
        self.b = 0.0        # Initialize bias = 0
        self.eta = eta      # Learning rate
        self.gamma = gamma  # Margin parameter

    def dot_product(self, v1: list, v2: list) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    def vector_add(self, v1: list, v2: list, scalar: float = 1.0) -> list:
        return [x + scalar * y for x, y in zip(v1, v2)]

    def norm(self, v: list) -> float:
        return math.sqrt(sum(x * x for x in v))

    def train(self, X, y, max_iter=1000):
        for iteration in range(max_iter):
            all_points_correct = True
            logging.info(f"> Iteration {iteration + 1}:")
            for xi, yi in zip(X, y):
                score = yi * (self.dot_product(self.w, xi) + self.b)
                logging.info(f"> > Current Point: {xi}, Label: {yi}, Score: {score:.4f}")
                if score < self.gamma:
                    self.w = self.vector_add(self.w, xi, scalar=self.eta * yi)
                    self.b += self.eta * yi
                    all_points_correct = False
                    # 输出当前的权重和偏置
                    logging.info(f"\t Updated: w={self.w}, b={self.b:.4f}, score={score:.4f}")
                else:
                    logging.info(f"\t No update: w={self.w}, b={self.b:.4f}, score={score:.4f}")
            if all_points_correct:
                logging.info("All points classified correctly.")
                break  # 当所有点都满足条件时，停止训练


    def predict(self, X: list) -> list:
        """
        Make predictions using the model
        :param X: Input feature matrix (list of lists)
        :return: List of predicted labels
        """
        return [1 if self.dot_product(self.w, xi) + self.b >= 0 else -1 for xi in X]

    def calculate_margin(self) -> float:
        """
        Calculate the current margin of the model
        :return: Margin value
        """
        return self.gamma / self.norm(self.w) if self.norm(self.w) != 0 else float('inf')


def setup_logging(dataset_name):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = f"{dataset_name}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='w'),
            logging.StreamHandler()
        ]
    )
    print(f"Logging to {log_filepath}")

    

if __name__ == '__main__':

    dataset_path = argparser()

    # set logger
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    setup_logging(dataset_name)
    
    logging.info(f"Dataset Path: {dataset_path}")

    lines = read_file(dataset_path)
    num_points, instance_dim, r, data_points = parse_dataset(lines)

    logging.info(f"n: {num_points}, d: {instance_dim}, r: {r}")
    perceptron = MarginPerceptron(instance_dim, eta=0.1, gamma=1.0)

    x, y = [], []
    for point in data_points:
        x.append(point[0])
        y.append(point[1])

    perceptron.train(x, y)