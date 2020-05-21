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

df_cust_spend=pd.read_csv("Cust_Spend_Data.csv");
print(df_cust_spend.head(10))

cust_data_attr=df_cust_spend.iloc[:,2:]
print(cust_data_attr.head(5));
cust_data_attr_scaled=cust_data_attr.apply(zscore);
print(cust_data_attr_scaled. ())

sns.pairplot(cust_data_attr_scaled,height=2,aspect=2,diag_kind="kde");
plt.show()

model =AgglomerativeClustering(n_clusters=3,linkage='average')
model.fit(cust_data_attr_scaled);
cust_data_attr['labels']=model.labels_;
cust_data_cluster=cust_data_attr.groupby(['labels'])
print(cust_data_cluster.head(5))
print(cust_data_cluster.mean())

z=linkage(cust_data_attr_scaled,method='average');
c,coph_dists=cophenet(z,pdist(cust_data_attr_scaled));

print(c)
plt.figure(figsize=(10,5))
plt.title('aglomerative hirearchial clusteringn Dendogram');
plt.xlabel('sample index');
plt.ylabel('distance');
dendrogram(z,leaf_rotation=90,color_threshold=40,leaf_font_size=8)
plt.tight_layout()
plt.show();