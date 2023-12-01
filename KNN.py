import numpy as np

X1 = np.array([1, 2, 3, 4, 5])
X2 = np.array([3, 4, 1, 4, 5])
Y = np.array([1, 0, 1, 1, 0])
X = np.vstack((X1, X2))

# Prediction point
prediction = np.array([[1], [3]])

# Transpose X for easy iteration
inputs = X.T

answer_array = []

# Calculate Euclidean distance for each point with index
for idx, point in enumerate(inputs):
    answer = np.sqrt(np.sum((prediction - point.reshape(-1, 1))**2))
    answer_array.append((answer, idx))  # Store distance along with its index

# Sort array based on distances
sorted_array = sorted(answer_array, key=lambda x: x[0])

# Select top K distances and retrieve indices
K = 2
top_k_indices = [idx for _, idx in sorted_array[:K]]

# Get corresponding Y values
corresponding_Y_values = [Y[idx] for idx in top_k_indices]

print("Top K distances:", sorted_array[:K])
print("Indices of Top K distances:", top_k_indices)
print("Corresponding Y values for the top K distances:", corresponding_Y_values)
#now we will figure out the Majiority voting 
couting_of_0=0
couting_of_1=1

for i in range(len(corresponding_Y_values)):
    if corresponding_Y_values[i]==0:
        couting_of_0+=1
    else:
        couting_of_1+=1

if(couting_of_1>couting_of_0):
    print('answer would be 0')
elif(couting_of_1<couting_of_0):
    print('answer would be 1')
else:
    print('could either be 0 or 1 ')
            