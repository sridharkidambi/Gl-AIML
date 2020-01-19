import numpy as np
import pandas as pd

lablels=['a','b','c','d','e']

list_values=[6,7,8,9,10]
array=np.array([1,2,3,4,5])
my_dict={'a':10,'b':20,'c':30,'d':40,'e':50}
my_dict2={'a':10,'b':20,'c':30,'d':40,'F':50}
print(pd.Series(data=list_values))

print(pd.Series(data=list_values,index=lablels))
print(pd.Series(data=my_dict))
my_dict_Series=pd.Series(data=my_dict)
my_dict_Series2=pd.Series(data=my_dict2)
print('output')
print(my_dict_Series + my_dict_Series2)
print(my_dict_Series['a'])
print(my_dict_Series[0])

from numpy.random  import randn

np.random.seed(1)  # to provide a repeatable same randon generation.
df=pd.DataFrame(randn(10,5),index='A B C D E F G H I J'.split(),columns=' Q W E R T'.split())
print(df)
print(df['Q'])
print(df[['Q','W']])
df['S']= df['Q'] + df['W']
print(df)
df.drop('S',axis=1,inplace=True)
print(df)
df.drop('A',axis=0,inplace=True)
print(df)
print('LOC ')

print(df.loc['D'])
print('LOC 1')

print(df.iloc[2])

print(df.loc[['C','D']], [['Q','W']])
print(df>0.5)
print(df[df>0.5])
print('>0.5')
print(df)
print(df[df['Q']> 0.5])
# print(df[(df['Q']> 0.5) & (df['W']>0)])
print(df[df['Q']> 0.5]['W'])
print(df[(df['Q']> 0.5) & (df['W']<0)])
print(df[(df['Q']> 0.5) & (df['W']>0)])
print(df.reset_index())
df.set_index('W')
print(df)
print(df.set_index('W'))
print(df)
df1=df[df>0.5]
print('DropNAN')
print(df1.dropna())
print(df1.dropna(axis=1))
print(df1.dropna(thresh=2))
print(df1.fillna(value=0))
print(df1['Q'].fillna(value=df1['Q'].mean()))
dat= { 'custID':['1001','1002','1002','1003','1003'],
       'custName':['a','b','c','d','e'] ,
       'profitInLakhs':[1000,2000,3000,4000,5000]

}

dat2= { 'custID':['1001','1002','1003','1004','1005'],
       'custpetName':['a','b','c','d',np.NaN] ,
       'rewards':[1000,2000,3000,3000,5000]

}
dat3= { 'custID':['1001','1002','1005','1006','1007'],
       'custName':['a','b','c','d','e'] ,
       'profitInLakhs':[1000,2000,3000,4000,5000]

}


dat5= { 'custID':['1001','1002','1003','1004','1005'],
       'custpetName':['a','b','c','d','e'] ,
       'rewards':[1000,2000,3000,3000,5000],
       

}
dat6= { 'custID1':['1001','1002','1005','1006','1007'],
       'custName1':['a','b','c','d','e'] ,
       'profitInLakhs1':[1000,2000,3000,4000,5000]

}

df2=pd.DataFrame(dat2)
df3=pd.DataFrame(dat3)



dt2=pd.DataFrame(dat)
print(dt2)

dt2_groupedBy=dt2.groupby('custID')
print(dt2.groupby('custID'))
print(dt2_groupedBy.mean())
print(dt2_groupedBy.count())
print(dt2_groupedBy.max())
print(dt2_groupedBy.describe())
print(dt2_groupedBy.describe().transpose())
print(dt2_groupedBy.describe().transpose()['1001'])
print(pd.concat([df2,df3 ],sort=False))
print(pd.concat([df2,df3 ],sort=False,axis=1))
print(pd.merge( df2,df3 , on='custID'))


df5=pd.DataFrame(dat5,index=[1,2,3,4,5])
df6=pd.DataFrame(dat6,index=[4,5,6,7,8])

print(df5.join(df6))
print(df5.join(df6,how='inner'))
print(df5.head(2))
print(df2['rewards'].unique)
print(df2['rewards'].nunique)

print(df2['rewards'].value_counts)
dfunique= df2[(df2['rewards']>2000) & (df2['custpetName']=='e')]
dfunique1= df2[(df2['rewards']>2000)]

print(dfunique)


def profit(a):
    return a*4

print('method apply')
print(dfunique['rewards'].apply(profit))
# print(dfunique['custID'].apply(sum))
print(dfunique['custpetName'].apply(len))
print(dfunique1)
print(df2.index)
print(df2.isnull())
print(df2.dropna())
print(df2.sort_values(by='custpetName',ascending=False))
print(df2.dropna())
# print(dfunique1['rewards'].apply(sum))  ----how to sum on int TypeError: 'int' object is not iterable

readDf=pd.read_csv('a.csv')
df2.to_csv('b.csv',index=False)
