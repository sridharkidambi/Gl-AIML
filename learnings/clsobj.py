import numpy as np
import pandas as pd

# self in class denotes the name of the object whihle multiple objects

class class1:
    name='satish'
    def FunctionName(self):
         print( 'the name is '+ self.name )

c1=class1()
c2=class1()
c1.FunctionName()
c2.name='shiva'
c2.FunctionName()

# sum_3 = lambda a,b,c :  a+b+c

sum_3 = lambda a,b,c :  a+b+c   

print(sum_3(1,2,3))

list_l =[[12,34,55,],[66,45,77],[45,77,88]]

print(np.array(list_l))