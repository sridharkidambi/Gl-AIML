import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn import metrics;
from sklearn.linear_model import LogisticRegression;
from sklearn.naive_bayes import GaussianNB;


df_diabetics =pd.read_csv("diabetes.csv");

print(df_diabetics.head());
print(df_diabetics.shape);

print(df_diabetics.isnull().values.any());

columns= list(df_diabetics)[0:-1]
print(columns)

df_diabetics.hist(stacked=False,bins=100,figsize=(40,30),layout=(5,2));
print(df_diabetics.corr());
# plt.show();
corr =df_diabetics.corr();
fig ,ax =plt.subplots(figsize=(11,11));
ax.matshow(corr);
plt.xticks(range(len(corr.columns)),corr.columns);
plt.yticks(range(len(corr.columns)),corr.columns);

sns.pairplot(df_diabetics,diag_kind='kde');


## Calculate diabetes ratio of True/False from outcome variable 
n_true=len(df_diabetics.loc[df_diabetics['class']==True]);
n_false=len(df_diabetics.loc[df_diabetics['class']==False]);

print('% of TRUE : '+ str(n_true/(n_false+n_true)*100));
print('% of FALSE : '+ str(n_false/(n_false+n_true)*100));

## Spliting the data 

x=df_diabetics.drop('class',axis=1);
y=df_diabetics['class'];

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1);

print(x_train.head());
diab_model=GaussianNB();
print(diab_model.fit(x_train,y_train.ravel()));

diab_train_predict=diab_model.predict(x_train);
print(metrics.accuracy_score(y_train,diab_train_predict));

diab_test_predict=diab_model.predict(x_test);

print(metrics.accuracy_score(y_test,diab_test_predict));

cm=metrics.confusion_matrix(y_test,diab_test_predict,labels=(1,0));
print('Confusion metrix: ');

print(cm);
df_cm=pd.DataFrame(cm,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
plt.figure(figsize=(7,5));
sns.heatmap(df_cm,annot=True);


print('Classification report : ');
print(metrics.classification_report(y_test,diab_test_predict,labels=[1,0]));

plt.show();
