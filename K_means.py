import numpy as np 
k=2
#now first we consider here 2 d array here you can also even consider 3d or even more does not matter at all 
X=np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
#make a copy
centroid=X.copy()
#now consider the k cluster
np.random.shuffle(centroid)
print(centroid[:k])

#in this way we have intilize k no of clusters 
distances = np.sqrt(((X - centroid[:, np.newaxis])**2).sum(axis=2))
print(np.argmin(distances, axis=0))
#now we got distance 