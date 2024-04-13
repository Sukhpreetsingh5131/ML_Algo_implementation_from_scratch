import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initializing network parameters
def initialize_parameters(input_size, layer_sizes):
    np.random.seed(42)  # Seed for reproducibility
    weights = []
    biases = []
    layer_input_size = input_size

    for size in layer_sizes:
        weights.append(np.random.randn(size, layer_input_size) * 0.1)
        biases.append(np.zeros((size, 1)))
        layer_input_size = size

    return weights, biases

# Forward propagation function
def forward_propagation(X, weights, biases):
    activations = [X]
    current_input = X

    for i in range(len(weights)):
        z = np.dot(weights[i], current_input) + biases[i]
        current_input = sigmoid(z)
        activations.append(current_input)

    return activations[-1], activations  # Return last activation (output) and all activations

# Example data setup
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([3, 4, 1, 4, 5])
Y = np.array([1, 0, 1, 1, 0])
X = np.vstack((X1, X2)).T  # Transpose to make each column a feature vector
X = X.T  # Back to row vectors for each sample for simplicity in matrix multiplication

# Network structure
input_size = X.shape[0]
layer_sizes = [5, 8, 10]  # Number of neurons in each layer
weights, biases = initialize_parameters(input_size, layer_sizes)

# Perform forward propagation
output, activations = forward_propagation(X, weights, biases)

print("Output of the network:\n", output)
print("All layer activations:\n", activations)
