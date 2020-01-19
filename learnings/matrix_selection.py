import numpy as np


sample_matrix= np.array(([1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,207]))
print(sample_matrix<7)
k=sample_matrix<7
print(sample_matrix[k])
print(sample_matrix[sample_matrix<7])
print(sample_matrix * sample_matrix)
print(sample_matrix + sample_matrix)
print(np.max(sample_matrix))
print(sample_matrix)
print(sample_matrix+1)
array_n=np.random.randn(6,6)
print(array_n)
print(np.round(array_n))
print(np.round(array_n,decimals=2))
print(np.round(array_n))
print(np.std(sample_matrix))
print(np.mean(sample_matrix))
print(np.median(sample_matrix))

sports=np.array(['golf','cric','fball','CRIC','cric'])
print(np.unique(sports))

