import numpy as np
X1=np.array([1,2,3,4,5,6])
X2=np.array([3,5,2,1,4,1])
X3=np.array([8,1,4,1,5,6])
Y=np.array([3,1,4,6,2,4])
#first we will initilize B,W,learning_rate
X=np.vstack((X1,X2,X3)).T
#print(X.ndim)
#print(X)
#initilize w with something 
W=np.zeros(X.shape[1])
#lets initilize B
B=0
m=len(Y)
#lets initilize learning rate 
learning_rate=0.02
iteration=100
for i in range(iteration):
    #lets first calculate prediction 
    Y_prediction=np.dot(X,W)+B
    #lets now calculate cost function 
    cost_function=np.sum((Y_prediction-Y)**2)/2*m
    #now lets calculate DW
    DW=np.dot(X.T,(Y_prediction-Y))/m
    DB=np.sum(Y_prediction-Y)/m
    W=W-learning_rate*DW
    B=B-learning_rate*DB
    print(f'when W is {W} and B is {B} then cost function would be {cost_function}')


