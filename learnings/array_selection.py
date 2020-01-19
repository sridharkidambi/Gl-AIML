import numpy as np


sample_array = np.arange(10,21)
print(sample_array)
print(sample_array[8])
print(sample_array[2:5])
sample_array[0:2] = 101
print(sample_array)
subset_sample_array=sample_array[0:7] #subset is a view and not a memory copy of original array.
print(subset_sample_array)
subset_sample_array[:]=1001
print(subset_sample_array)
subset_sample_array1= sample_array.copy()
subset_sample_array1[:]=999
print(sample_array)
print(subset_sample_array1)

print('excersie')
demo_array = np.arange(10,21)
print(demo_array)
subset_demo_array = demo_array[0:7]
print(subset_demo_array)

subset_demo_array[:]= 101
print(subset_demo_array)

# subset_demo_array

