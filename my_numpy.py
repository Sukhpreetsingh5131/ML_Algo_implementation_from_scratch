import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4])

# Print the type of the array
print(type(arr))

#now lets figure out the diffrent domensions of array

arr1=np.array([1,2,3,4])
#1 d array 
arr2=np.array([[1,2,3],[2,4,5]])
# 2d 
print(arr2)
arr3=np.array([[[1,2,3],[3,5,2],[65,23,55]]])
print(arr3)

#now how to check the domensions of diffrent arrays 


#domensions of array ..
arr56=np.array([1,2,3,4,5])
print(arr56[2])
print(arr56[0]+arr56[2])

array=np.array([1,2,3,4,5])
array2=np.array([3])
final_array=array-array2
print(final_array)

#now lets calculate the mean 
X=np.array([[1,2],[3,5]])
mean=X.mean(axis=0)
print('sdjnsd')
print(mean)