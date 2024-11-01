# dataset_loader.py

def read_file(file_path: str) -> list[str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_dataset(lines: list[str]):
    # Read the first line
    first_line = lines[0]
    num_points, instance_dim, r = map(int, first_line.split(','))

    # Load data points
    data_points = []
    for line in lines[1:]:
        values = line.strip().split(',')
        coordinates = list(map(float, values[:-1]))
        label = int(values[-1])
        data_points.append((coordinates, label))

    return num_points, instance_dim, r, data_points


if __name__ == "__main__":
    file_path = '/Users/hairuohan/Documents/CUHK Acdemic/CMSC 5724/project/CMSC5724_Project2_MarginPerceptron/datasets/2d-r16-n10000.txt'

    lines = read_file(file_path)
    num_points, instance_dim, r, data_points = parse_dataset(lines)

    print(f"n: {num_points}, d: {instance_dim}, r: {r}")
    print("Data points:")
    for point in data_points:
        print(point)
