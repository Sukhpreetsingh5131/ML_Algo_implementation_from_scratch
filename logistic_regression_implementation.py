import numpy as np
#lets consider two features 
X1=np.array([1,2,3,4,5])
X2=np.array([2,4,6,2,1])
Y=np.array([1,0,1,0,1])

#lets combine first X1 and X2 
X=np.vstack((X1,X2)).T

iteration=10
learning_rate=0.08
W=np.zeros(X.shape[1])

B=0
for i in range(iteration):
    Z=np.dot(X,W)+B
    Y_prediction=1/(1+np.exp(-Z))
    #now we will calculate the loss 
    # Correcting the loss function calculation
    loss_function = -np.sum(Y * np.log(Y_prediction) + (1 - Y) * np.log(1 - Y_prediction)) / len(Y)
    print(loss_function)
    DW=np.dot(X.T,(loss_function-Y)/len(Y))
    DB=np.sum(loss_function-Y)/len(Y)
    W=W-learning_rate*DW
    B=B-learning_rate*DB

    print(f'when W is {W} and B is {B} then loss function would be this {loss_function}')




