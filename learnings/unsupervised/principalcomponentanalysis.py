import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.feature_extraction.text import CountVectorizer  #DT does not take strings as input for the model fit step....
from IPython.display import Image  
#import pydotplus as pydot
from sklearn import tree
from os import system
from sklearn.cluster import KMeans;
from scipy.stats import zscore;
from scipy.spatial.distance import cdist,pdist
from sklearn.cluster import AgglomerativeClustering;
from scipy.cluster.hierarchy import cophenet,dendrogram,linkage;
from sklearn.decomposition import PCA;
from sklearn import linear_model;
from sklearn.linear_model import LinearRegression


df_auto_sales=pd.read_csv("AUTO-mpg.csv")
print(df_auto_sales.shape)
print(df_auto_sales.head(5))

df_auto_sales_dropped=df_auto_sales.drop(labels=['car name','origin'],axis=1)

print(df_auto_sales_dropped.head(5))
hpIsDigit=pd.DataFrame(df_auto_sales_dropped.hp.str.isdigit());
df_auto_sales_dropped=df_auto_sales_dropped.replace('?',np.nan)
df_auto_sales_dropped[hpIsDigit["hp"]==False]
print(df_auto_sales_dropped.median())

medianFiller=lambda x: x.fillna(x.median())
df_auto_sales_dropped=df_auto_sales_dropped.apply(medianFiller,axis=0)
df_auto_sales_dropped['hp']=df_auto_sales_dropped['hp'].astype('float64')

X=df_auto_sales_dropped.drop(['mpg'],axis=1)
y=df_auto_sales_dropped['mpg']
# sns.pairplot(X,diag_kind='kde')
# plt.show();
X_scaled= X.apply(zscore);
print(X_scaled.head())

cov_matrix=np.cov(X_scaled,rowvar=False);
print(cov_matrix)

pca=PCA(n_components=6)
pca.fit(X_scaled)
print('eigen values')
# eigen values
print(pca.explained_variance_)
# eigen vectors
print(pca.components_)

# eigen ratio showing weightage
print(pca.explained_variance_ratio_)

plt.bar(list(range(1,7)),pca.explained_variance_ratio_,alpha=0.5,align='center')
plt.xlabel('eigen weightage')
plt.ylabel('% of propotions')
plt.show();

# cummulative way of expressing 
plt.step(list(range(1,7)),np.cumsum(pca.explained_variance_ratio_),where='mid')
plt.xlabel('eigen weightage')
plt.ylabel('cummulative  propotions')
plt.show();

# 3 components have a good elbow coverage
pca3=PCA(n_components=3)
pca3.fit(X_scaled)
print('eigen values 3')
# eigen values
print(pca3.explained_variance_)
# eigen vectors
print(pca3.components_)

# eigen ratio showing weightage
print(pca3.explained_variance_ratio_)

X_scaled_tranformed=pca3.transform(X_scaled)

sns.pairplot(pd.DataFrame(X_scaled_tranformed),diag_kind='kde');
plt.show()

print(pd.DataFrame(X_scaled_tranformed).head(5))

# LinearModel
linear_reg=LinearRegression();
linear_reg.fit(X_scaled,y)
print(linear_reg.score(X_scaled,y))

linear_reg_transform=LinearRegression();
linear_reg_transform.fit(X_scaled_tranformed,y)
print(linear_reg_transform.score(X_scaled_tranformed,y))

# reducing the dimension by 3 we ahve almost close to the 6 paramenters scoere by just 3%