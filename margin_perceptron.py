import math
import os
import logging
import matplotlib.pyplot as plt
from utils.logger import setup_logging
from utils.argparser import argparser
from utils.dataset_loader import read_file, parse_dataset
from utils.visualize_2d import plot_2d_points_and_decision_boundary

class MarginPerceptron:
    def __init__(self, d: int, r: float):
        # y = w * x + b
        self.w = [0.0] * d  # Initialize weight vector -> 0 vector
        self.b = 0.0        # Initialize bias = 0
        self.eta = 0.01     # Learning rate
        self.R_max = r      # Maximum distance to origin
        self.gamma_guess = self.R_max  # R_max -> γ_guess ?

    # calculate the dot product of two vectors
    def dot_product(self, v1: list[float], v2: list[float]) -> float:
        return sum(x * y for x, y in zip(v1, v2))
    
    # calculate the sum of two vectors
    def vector_add(self, v1: list[float], v2: list[float], scalar: float = 1.0) -> list[float]:
        return [x + scalar * y for x, y in zip(v1, v2)]

    # calculate the norm of a vector
    def norm(self, v: list[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def train(self, X: list[list[float]], y: list[int]) -> None:
        terminate_iter = self.terminate_iter()

        for iteration in range(10000):
            logging.info(f"Iteration {iteration + 1}:\n")
            violation_found = False

            for xi, yi in zip(X, y):
                # Calculate the distance to the separating plane d = w * x + b
                distance = self.dot_product(self.w, xi) + self.b
                
                # Check if there is a violation point: distance + classification error
                if abs(distance) < self.gamma_guess / 2 or (yi == 1 and distance < 0) or (yi == -1 and distance > 0):
                    violation_found = True
                    logging.info(f"\t\u2716 Violation Point Found: {xi}, Label: {yi}, Distance: {distance:.4f}")

                    # Adjust weights and bias
                    if yi == 1:
                        self.w = self.vector_add(self.w, xi, scalar=self.eta)  # w += η * p
                        self.b += self.eta  # b += η
                    else:
                        self.w = self.vector_add(self.w, xi, scalar=-self.eta)  # w -= η * p
                        self.b -= self.eta  # b -= η

                    # Output the current weights and bias
                    logging.info(f"\t\t Updated: w={self.w}, b={self.b:.4f}, Distance: {distance:.4f}\n")
            
            # Self-Termination
            if not violation_found:
                logging.info("\t\u2714 No violation points found, training finished.\n")
                return  # End training if no violation points

            # Forced-Termination
            if iteration == terminate_iter - 1:
                logging.info("\t\u26A0 Training reached maximum iterations.\n")
                self.gamma_guess /= 2
                terminate_iter = self.terminate_iter()
                logging.info(f"\t\t Updated: gamma_guess={self.gamma_guess:.4f}\n")
                iteration = 0  # Restart training

    # calculate the termination condition: 12 * R_max^2 / γ_guess^2
    def terminate_iter(self) -> int:
        return int(12.0 * (self.R_max **2)/(self.gamma_guess **2))

    # calculate the minimum margin of the classifier
    def calculate_margin(self, X: list[list[float]], y: list[int]) -> float:
        if self.norm(self.w) == 0:
            return float('inf')
        margins = [abs(self.dot_product(self.w, xi) + self.b) / self.norm(self.w) for xi in X]
        return min(margins)
    
    def return_w(self) -> list[float]:
        return self.w
    
    def return_b(self) -> float:
        return self.b



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

    perceptron = MarginPerceptron(instance_dim, r)
    X, y = [], []
    for point in data_points:
        X.append(point[0])
        y.append(point[1])
    perceptron.train(X, y)
    margin = perceptron.calculate_margin(X, y)
    logging.info(f"gamma_guess: {perceptron.gamma_guess}, Margin: {margin:.4f}\n")
    w = perceptron.return_w()
    b = perceptron.return_b()
    logging.info(f"w: {w}, b: {b:.4f}\n")

    # plot_2d_points_and_decision_boundary(x, y, w, b)
