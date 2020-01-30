import numpy as np

#### Create an empty array of 20 0's and replace the 4th object with the number 5
ary=np.zeros(20)
ary[3]=5
print(ary)

#### Create an array of 20 1's and store it as a variable named array_master. Copy the same array into another variable named array_copy
array_master=np.ones(20)
array_copy=array_master.copy()
print(array_copy)

#### Create an array containing 30 1's and broadcast all the one's to the value 100
array_master=np.ones(30)
array_master[:]=100
print(array_master)

#### Create an array of all even integers from 2 to 10 and name it a1

#### Create an array of all even integers from 22 to 30 and name it a2

####  a) Use the 2 arrays as rows and create a matrix [ Hint - Use stack function from numpy ]

a1=np.arange(2,11,2)
print(a1)
a2=np.arange(22,31,2)
print(a1)
print(np.stack((a1,a2)))
####  b) Use the 2 arrays as columns and create a matrix [ Hint - Use column_stack function from numpy ]
print(np.column_stack((a1,a2)))

#### Create a 5x6 matrix with values ranging from 0 to 29 and retrieve the value intersecting at 2nd row and 3rd column
array5_6=np.arange(30)
print(array5_6.reshape(5,6))

#### Create an identity matrix of shape 10x10 and replace the 0's with the value 21

array10_10=np.eye(10)
array10_10[array10_10==0]=21
print(array10_10)

#### Use NumPy to generate a random set of 10 numbers between 0 and 1
#### Display a boolean array output where all values > 0.2 are True, rest are marked as False
array_rd=np.random.rand(10)
print(array_rd>0.2)

#### Use NumPy to generate an array matrix of 5x2 random numbers sampled from a standard normal distribution
matrix_normal=np.random.randn(5,2)
print(matrix_normal)

#### Create an array of 30 linearly spaced points between 0 and 100:
array_linespace=np.linspace(0,100,30)
print(array_linespace)

## Numpy Indexing and Selections

# Using the below given Matrix, generate the output for the below questions

simple_matrix = np.arange(1,101).reshape(10,10)
print(simple_matrix)

#### a) Retrieve the last 2 rows and first 3 column values of the above matrix using index & selection technique
print(simple_matrix[8:,:3])
#### b) Retrieve the value 55 from the above matrix using index & selection technique
print(simple_matrix[5][4])
#### c) Retrieve the values from the 3rd column in the above matrix
print(simple_matrix[:,2])
#### d) Retrieve the values from the 4th row in the above matrix
print(simple_matrix[3,:])
#### e) Retrieve values from the 2nd & 4th rows in the above matrix
print(simple_matrix[(1,3),:])

### Calculate the following values for the given matrix
#### a) Calculate sum of all the values in the matrix
print(simple_matrix.sum())
#### b) Calculate standard deviation of all the values in the matrix
print(simple_matrix.std())
#### c) Calculate the variance of all values in the matrix
print(simple_matrix.var())
#### d) Calculate the mean of all values in the matrix
print(simple_matrix.mean())
#### e) Retrieve the largest number from the matrix
print(simple_matrix.max())
#### f) Retrieve the smallest number from the matrix
print(simple_matrix.min())
