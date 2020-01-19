import pandas as pd
import numpy as np
def greetings(name):
    x=10
    print('inside x is :'+ str(x))
    global y
    y=100
    print('inside y is :'+ str(y))
    """ this greets the user """
    return "hello user " + name +" how are your doing"

x=100
print(greetings('sridhar'))
print('x outside is '+str(x))
print('y is '+ str(y))
print(greetings('savitha'))
print(greetings.__doc__)

square_root = lambda x: x*x

print(square_root(9))

my_list_values=[1,2,3,4,5,6,7]
chk_lst=list(filter(lambda x: (x%2==0),my_list_values))
print(chk_lst)

dat2= { 'custID':['1001','1002','1003','1004','1005'],
       'custpetName':['a','b','c','d',np.NaN] ,
       'rewards':[1000,2000,3000,3000,5000]

}

df2= pd.DataFrame( dat2)
print(df2)
# df2=del df2['custpetName']
# print(df2.apply(lambda x: x.fillna(x.median()), axis=0))