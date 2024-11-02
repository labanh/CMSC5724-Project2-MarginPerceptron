# CMSC5724_Project2_MarginPerceptron
CUHK CMSC 5724 Project #2: Margin Perceptron

# Usage
```bash
python margin_perceptron.py --dataset 2d-r16-n10000.txt

python margin_perceptron.py --dataset 4d-r24-n10000.txt

python margin_perceptron.py --dataset 8d-r12-n10000.txt
```

# Dataset
Your implementation should work on any dataset in the following format:
The first line contains three numbers n, d, and r, where n is the number of points, d is the dimensionality of the instance space, and r is the radius.
The i-th line (where i goes from 2 to n + 1) gives the (i - 1)-th point in the dataset as: \
x1,x2,...,xd,label \
where the first d values are the coordinates of the point, and label = 1 or -1. \
We have prepared three datasets for you: \
2d-r16-n10000 \
4d-r24-n10000 \
8d-r12-n10000

# Deliverables
An executable program. \
A readme file detailing how to use the program. \
Source code. \
A report explaining the margin of your classifier for each of the three datasets.
