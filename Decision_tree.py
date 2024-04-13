import numpy as np

def entropy(count_0, count_1):
    total = count_0 + count_1
    p_zero = count_0 / total if total > 0 else 0
    p_one = count_1 / total if total > 0 else 0
    entropy_zero = -p_zero * np.log2(p_zero) if p_zero > 0 else 0
    entropy_one = -p_one * np.log2(p_one) if p_one > 0 else 0
    return entropy_zero + entropy_one

def calculate_information_gain(X, Y, feature_index):
    # Extract the column for the feature
    feature_values = X[:, feature_index]
    total_entropy = entropy(np.sum(Y == 0), np.sum(Y == 1))
    
    # Find unique values and their entropy
    unique_values, counts = np.unique(feature_values, return_counts=True)
    weighted_entropy = 0

    for value, count in zip(unique_values, counts):
        indices = np.where(feature_values == value)
        corresponding_Y_values = Y[indices]
        count_0 = np.sum(corresponding_Y_values == 0)
        count_1 = np.sum(corresponding_Y_values == 1)
        value_entropy = entropy(count_0, count_1)
        weighted_entropy += (count / len(Y)) * value_entropy
    
    # Information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

def analyze_features(X, Y):
    num_features = X.shape[1]
    information_gains = []

    for i in range(num_features):
        ig = calculate_information_gain(X, Y, i)
        information_gains.append(ig)
        print(f"Information Gain for Feature {i}: {ig}")

    return information_gains

# Example data
X = np.array([[1, 3], [2, 4], [3, 1], [4, 4], [5, 5]])
Y = np.array([1, 0, 1, 1, 0])

# Calculate information gains for each feature
information_gains = analyze_features(X, Y)
print(information_gains)
