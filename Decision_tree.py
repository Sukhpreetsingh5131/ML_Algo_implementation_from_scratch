import numpy as np
import math

# Example data
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([3, 4, 1, 4, 5])
Y = np.array([1, 0, 1, 1, 0])
X = np.vstack((X1, X2)).T  # Make sure to transpose to get the correct feature columns

# Function to calculate entropy
def entropy(y):
    if len(y) == 0:
        return 0
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    ent = -np.sum(probabilities * np.log2(probabilities))
    return ent

# Function to calculate entropy for each value of each feature
def entropy_of_feature(feature_column, Y):
    unique_values = np.unique(feature_column)
    entropy_values = {}
    for val in unique_values:
        y_subset = Y[feature_column == val]
        entropy_val = entropy(y_subset)
        entropy_values[val] = entropy_val
    return entropy_values

# Calculate the entropy for the entire dataset
entropy_dataset = entropy(Y)

# Calculate the IG for each feature and store them
information_gains = []
for feature in X.T:
    total_entropy = 0
    for feature_value in np.unique(feature):
        subset = Y[feature == feature_value]
        weight = len(subset) / len(Y)
        total_entropy += weight * entropy(subset)
    IG = entropy_dataset - total_entropy
    information_gains.append(IG)

# Find the feature with the maximum IG
max_IG = max(information_gains)
max_IG_feature = np.argmax(information_gains) + 1  # Adding 1 for human-readable feature index

print(f"Maximum Information Gain: {max_IG}, Feature: {max_IG_feature}")
#Now once we find out the Root node of feature 