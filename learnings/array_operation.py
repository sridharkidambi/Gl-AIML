import numpy as np

simple_list=[1,2,3]
print(np.array(simple_list))
list_of_list=[[1,2,3],[4,5,6],[7,8,9]]
print(np.array(list_of_list))
print(np.arange(0,10))
print(np.arange(0,10,5))
print(np.zeros(1000))
print(np.ones(1000))
print(np.linspace(0,20,3))
print(np.linspace(0,20,4))
print(np.eye(8))
#ramomization
print(np.random.rand(3,2)) # uniform distribution betweeen 0 and 1 (not inclusive)
print(np.random.randn(3,2)) #normally distributed ,mean =0 s.deviation =1
print(np.random.rand(3,2))
print(np.random.randint(3,20))
print(np.random.randint(3,20,5))

sample_array = np.arange(30)
rand_array=np.random.randint(0,100,20)
print(sample_array.reshape(5,6))
sample_array=sample_array.reshape(5,6)

print(rand_array.max())
print(rand_array.argmin())
print(sample_array.shape)
print(sample_array.dtype)
print(sample_array)
print(sample_array.transpose())
print(sample_array.T)