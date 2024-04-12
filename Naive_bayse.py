import numpy as np

# Your data
X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([2, 3, 5, 6, 2])
Y = np.array([1, 0, 1, 1, 0])
X = np.vstack((X1, X2)).T

# Step 1: Calculate the prior probability p(y) for 1 and 0
occurance_of_0 = 0
occurance_of_1 = 0
for i in range(len(Y)):
    if Y[i] == 0:
        occurance_of_0 += 1
    else:
        occurance_of_1 += 1
total_len = len(Y)
p_Y_1 = occurance_of_1 / total_len
p_Y_0 = occurance_of_0 / total_len
print("Probability of Y=1:", p_Y_1)
print("Probability of Y=0:", p_Y_0)

# Step 2: Calculate mean and variance for each class
classes = np.unique(Y)
priors = [p_Y_0, p_Y_1]

X_class_0 = X[Y == 0]

print(X_class_0)
X_class_1 = X[Y == 1]
print(X_class_1)

mean_X_class_0 = X_class_0.mean(axis=0)
var_X_class_0 = X_class_0.var(axis=0)

mean_X_class_1 = X_class_1.mean(axis=0)
var_X_class_1 = X_class_1.var(axis=0)

mean = np.array([mean_X_class_0, mean_X_class_1])
var = np.array([var_X_class_0, var_X_class_1])

print("Mean for class 0:", mean_X_class_0)
print("Variance for class 0:", var_X_class_0)
print("Mean for class 1:", mean_X_class_1)
print("Variance for class 1:", var_X_class_1)

# Gaussian PDF function
def gaussian_pdf(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(- (x - mean) ** 2 / (2 * var))

# Prediction function
def predict(X_new):
    posteriors = []

    # Calculate posterior for each class
    for i in range(len(classes)):
        prior = priors[i]
        likelihood = 1
        for j in range(X_new.shape[0]):
            likelihood *= gaussian_pdf(X_new[j], mean[i][j], var[i][j])
        posteriors.append(prior * likelihood)

    # Return the class with the highest posterior probability
    return classes[np.argmax(posteriors)]

# Predicting a new data point
X_new = np.array([3, 4])  # New data point
predicted_class = predict(X_new)
print("Predicted class:", predicted_class)
