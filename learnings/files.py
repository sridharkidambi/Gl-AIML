import numpy as np


sample_array= np.arange(0,20)
np.save('sample_array',sample_array)
np.savetxt('sample_arrayTxt',sample_array,delimiter=',')
np.savez('sample_arrayZ',a=sample_array,b=sample_array)
np.savez_compressed('sample_arrayZC',sample_array,sample_array)
print(np.load('sample_array.npy'))
arch=np.load('sample_arrayZ.npz')
print(arch['a'])
# print(np.load('sample_arrayZ.npz'))
print('result')
print(np.loadtxt('sample_arrayTxt',delimiter=',')) 