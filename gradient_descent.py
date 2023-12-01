import numpy as np
X=np.array([1,2,3,4])
Y=np.array([3,2,1,5])
w=0
b=0
learning_rate=0.03
iteration=10
m=len(X)
for i in range(iteration):
    #first we calculate Y_prediction 
    Y_prediction=w*X+b
    # now we will calculate the error 
    error=Y_prediction-Y
    #now we will calculate cost function 
    cost_function_sum=sum(error)
    cost_function=cost_function_sum**2
    DW=np.sum(error*X)/m
    DB=np.sum(error)/m
    w=w-learning_rate*DW
    b=b-learning_rate*DB
    print(f'when w is {w} and b is {b} then error is {cost_function}')


    