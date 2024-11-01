import math

class MarginPerceptron:
    def __init__(self, d: int, eta: float = 0.1, gamma: float = 1.0):

        self.w = [0.0] * d  # Initialize weight vector to zero vector
        self.b = 0.0        # Initialize bias to 0
        self.eta = eta      # Learning rate
        self.gamma = gamma  # Margin parameter

    def dot_product(self, v1: list, v2: list) -> float:
        return sum(x * y for x, y in zip(v1, v2))

    def vector_add(self, v1: list, v2: list, scalar: float = 1.0) -> list:
        return [x + scalar * y for x, y in zip(v1, v2)]

    def norm(self, v: list) -> float:
        return math.sqrt(sum(x * x for x in v))

    def train(self, X: list, y: list, max_iter: int = 1000):
        """
        Train the Margin Perceptron model
        :param X: Input feature matrix (list of lists)
        :param y: Label list
        :param max_iter: Maximum number of iterations
        """
        for _ in range(max_iter):
            all_points_correct = True
            for xi, yi in zip(X, y):
                score = yi * (self.dot_product(self.w, xi) + self.b)
                if score < self.gamma:
                    self.w = self.vector_add(self.w, xi, scalar=self.eta * yi)
                    self.b += self.eta * yi
                    all_points_correct = False
            if all_points_correct:
                break  # Stop training when all points satisfy the condition

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
