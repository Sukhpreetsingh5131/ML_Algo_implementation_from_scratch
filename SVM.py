import numpy as np
X1=np.array([1,2,3,4,5])
X2=np.array([2,3,5,1,5])
Y=np.array([1,0,1,0,1])
X=np.vstack((X1,X2)).T
print(X)