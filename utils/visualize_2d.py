import matplotlib.pyplot as plt

def plot_2d_points_and_decision_boundary(X: list[list[float]], y: list[int], w: list[float], b: float) -> None:
    # Plot data points, different colors for different classes
    for xi, yi in zip(X, y):
        if yi == 1:
            plt.scatter(xi[0], xi[1], color='blue', s=0.1, label='Class 1' if 'Class 1' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(xi[0], xi[1], color='red', s=0.1, label='Class -1' if 'Class -1' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot decision boundary
    x_min = min(point[0] for point in X) - 1
    x_max = max(point[0] for point in X) + 1
    x_values = [x_min + i * (x_max - x_min) / 100 for i in range(101)]

    # Calculate y_values
    y_values = [-(w[0] * x + b) / w[1] for x in x_values]

    plt.plot(x_values, y_values, color='green', label='Decision Boundary')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('2D Points and Decision Boundary')
    plt.legend("upper right")
    plt.grid(True)
    # Save the image to the specified path
    plt.savefig('2d_points_and_decision_boundary.png')
    plt.close()  # Close the image to prevent overlap in subsequent plots