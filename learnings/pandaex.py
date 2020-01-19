import numpy as np
import pandas as pd

lablels=['a','b','c','d','e']

list_values=[6,7,8,9,10]
array=np.array([1,2,3,4,5])
my_dict={'a':10,'b':20,'c':30,'d':40,'e':50}
print(pd.Series(data=list_values))

print(pd.Series(data=list_values,index=lablels))
print(pd.Series(data=my_dict))