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
from scipy.spatial.distance import cdist

df_tech_spt=pd.read_csv("technical_support_data.csv");
print(df_tech_spt.head(5));
print(df_tech_spt.shape);
print(df_tech_spt.describe().transpose());
print(df_tech_spt.dtypes);
print(df_tech_spt.PROBLEM_TYPE.value_counts())


techsupportAttr=df_tech_spt.iloc[:,1:]
print(techsupportAttr);
techsupportAttr_scaled=techsupportAttr.apply(zscore);
# sns.pairplot(techsupportAttr_scaled,diag_kind='kde');
# plt.show();

clusters=range(1,10);
meanDistortions=[];
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(techsupportAttr_scaled)
    predict=model.predict(techsupportAttr_scaled)
    meanDistortions.append(sum(np.min(cdist(techsupportAttr_scaled,model.cluster_centers_,'euclidean'),axis=1))/techsupportAttr_scaled.shape[0])


plt.plot(clusters,meanDistortions,'bx-')
plt.xlabel('k')
plt.ylabel('Avg distortion')
plt.title('Selecting K with Elbow Method')
plt.show();
# take look into the clsuter 3 and 5 they have an elbow.

final_model=KMeans(3)
final_model.fit(techsupportAttr_scaled);
prediction=final_model.predict(techsupportAttr_scaled);
df_tech_spt["GROUP"]=prediction
techsupportAttr_scaled["GROUP"]=prediction
print("groups assigned")
print(df_tech_spt.head(5))

techsupport_cluster=techsupportAttr_scaled.groupby(['GROUP'])
techsupport_cluster.mean()

techsupportAttr_scaled.boxplot(by='GROUP',layout=(2,4),figsize=(15,10))
plt.show()


final_model=KMeans(5)
final_model.fit(techsupportAttr_scaled);
prediction=final_model.predict(techsupportAttr_scaled);
df_tech_spt["GROUP"]=prediction
techsupportAttr_scaled["GROUP"]=prediction
print("groups assigned")
print(df_tech_spt.head(5))

techsupport_cluster=techsupportAttr_scaled.groupby(['GROUP'])
techsupport_cluster.mean()

techsupportAttr_scaled.boxplot(by='GROUP',layout=(2,4),figsize=(15,10))
plt.show()