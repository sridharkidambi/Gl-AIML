import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn import metrics;
from sklearn.neighbors import  KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
from scipy.stats import zscore;

bc_data  = pd.read_csv('wisc_bc_data.csv');
print(bc_data.head(5));
print(bc_data.info);
print(bc_data.describe());
print(bc_data.shape);
print(bc_data.dtypes);
print(bc_data['diagnosis'].value_counts());
bc_data['diagnosis']=bc_data.diagnosis.astype('category');
print(bc_data.dtypes);
print(bc_data.describe());
bc_data=bc_data.drop(labels=['id'],axis=1);
print(bc_data.head());
x_axis=bc_data.drop(labels='diagnosis',axis=1);
y_axis=bc_data['diagnosis'];

x_axis_scaled =x_axis.apply(zscore);
print(x_axis_scaled.describe());

x_train,x_test,y_train,y_test=train_test_split(x_axis_scaled,y_axis,test_size=0.30,random_state=1);

NNH=KNeighborsClassifier(n_neighbors=5,weights='distance');
NNH.fit(x_train,y_train);


predicted_labels=NNH.predict(x_test);
print(NNH.score(x_test,y_test)*100);
cm= metrics.confusion_matrix(y_test,predicted_labels, labels=['M','B']);
df_cm=pd.DataFrame(cm,index=[i for i in ['M','B']],columns=[i for i in ['Predict_M','Predict_B']]);
plt.figure(figsize=(7,5));
# sns.heatmap(df_cm,annot=True,fmt='g');

# finding the best K value:
scores=[];
for k in range(1,50):
    NNH=KNeighborsClassifier(n_neighbors=k,weights='distance');
    NNH.fit(x_train,y_train);
    scores.append(NNH.score(x_test,y_test));
plt.plot(range(1,50),scores);
plt.show();