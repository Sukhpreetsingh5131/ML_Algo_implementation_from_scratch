import numpy as np

# Example data
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([4, 2, 4, 6, 2])
x3 = np.array([2, 4, 1, 5, 6])
y = np.array([1, 2, 3, 4, 5])
X = np.vstack((x1, x2, x3)).T

# Initialization
w = np.zeros(X.shape[1])
b = 0
lemda = 30
learning_rate = 0.01  # Define a learning rate
iterations = 10

# Lasso Regression
for i in range(iterations):
    y_pred = np.dot(X, w) + b
    loss_function = np.sum((y_pred - y)**2) / (2 * len(y)) + lemda * np.sum(np.abs(w))
    DW = np.dot(X.T, (y_pred - y)) / len(y)
    DB = np.sum(y_pred - y) / len(y)
    w = w - learning_rate * DW
    b = b - learning_rate * DB
