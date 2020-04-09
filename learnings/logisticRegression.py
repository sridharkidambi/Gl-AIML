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

# ROC and AOC demo
from sklearn.metrics import roc_curve,auc;
from sklearn import svm;

# changing threshold values:
from sklearn.preprocessing import binarize;

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
print('% of training data : '+ str(len(x_train)/len(df_diabetics.index)*100));
print('% of test data : '+ str(len(x_test)/len(df_diabetics.index)*100));

print('% original diabetics true  are : '+ str(len(df_diabetics.loc[df_diabetics['class']==1])) + '     % is     ' + str( 100 * len(df_diabetics.loc[df_diabetics['class']==1])/len(df_diabetics.loc[df_diabetics.index])));
print('% original diabetics false  are : '+ str(len(df_diabetics.loc[df_diabetics['class']==0])) + '     % is     ' + str( 100 * len(df_diabetics.loc[df_diabetics['class']==0])/len(df_diabetics.loc[df_diabetics.index])));

print('% train diabetics true  are : '+ str(len(y_train.loc[y_train[:]==1])) + '     % is     ' + str( 100 * len(y_train.loc[y_train[:]==1])/len(x_train.loc[x_train.index])));
print('% train diabetics true  are : '+ str(len(y_train.loc[y_train[:]==0])) + '     % is     ' + str( 100 * len(y_train.loc[y_train[:]==0])/len(x_train.loc[x_train.index])));

print('% Test diabetics true  are : '+ str(len(y_test.loc[y_test[:]==1])) + '     % is     ' + str( 100 * len(y_test.loc[y_test[:]==1])/len(x_test.loc[x_test.index])));
print('% Test diabetics true  are : '+ str(len(y_test.loc[y_test[:]==0])) + '     % is     ' + str( 100 * len(y_test.loc[y_test[:]==0])/len(x_test.loc[x_test.index])));

# Data Preparation
### Replace 0s with serial mean 
rep_o= SimpleImputer(missing_values=0,strategy='mean');
cols=x_train.columns;
x_train=pd.DataFrame(rep_o.fit_transform(x_train));
x_test=pd.DataFrame(rep_o.fit_transform(x_test));
x_train.columns=cols;
x_test.columns=cols;
print(x_train.head());

# Logistic Regression

logistic_reg_model = LogisticRegression(solver='liblinear');
logistic_reg_model_2 = svm.SVC(kernel='linear',probability=True);
logistic_reg_model.fit(x_train,y_train);
logistic_reg_model_2.fit(x_train,y_train);
print('model trained')
y_predict =logistic_reg_model.predict(x_test);
y_predict_1 =logistic_reg_model.predict_proba(x_test);
y_predict_2 =logistic_reg_model_2.predict_proba(x_test);
print('model predict done')

coeff_df=pd.DataFrame(logistic_reg_model.coef_);
print(coeff_df);
coeff_df['intercept']=logistic_reg_model.intercept_;
print(coeff_df);

model_score=logistic_reg_model.score(x_test,y_test);
print(model_score*100);
cm=metrics.confusion_matrix(y_test,y_predict,labels=(1,0));
print(cm);
df_cm=pd.DataFrame(cm,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
plt.figure(figsize=(7,5));
sns.heatmap(df_cm,annot=True);



# The confusion matrix

# True Positives (TP): we correctly predicted that they do have diabetes 48

# True Negatives (TN): we correctly predicted that they don't have diabetes 132

# False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error") 14 Falsely predict positive Type I error

# False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error") 37 Falsely predict negative Type II error


# Changing threshold values:
# print(y_predict_2[:,1])
y_predict_class = binarize(y_predict_2,threshold= 0.3)[0];
print('values predict: ')
print(y_predict_class);
print('values predict ends. ')
# cm1=metrics.confusion_matrix(y_test,y_predict_class,labels=(1,0));
# print(cm1);
# df_cm1=pd.DataFrame(cm1,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
# plt.figure(figsize=(7,5));
# sns.heatmap(df_cm1,annot=True);


# ROC curve for y_predict_1 i

fpr1,tpr1,thrshold1=roc_curve(y_test, y_predict_1 [:,1])
roc_auc1=auc(fpr1,tpr1);
print("Area under cuver is :%f" % roc_auc1);
print("fpr is {}" ,fpr1);
print("thershold is {}" ,thrshold1);
print("tpr is {}" ,tpr1);

fpr2,tpr2,thrshold2=roc_curve(y_test, y_predict_2 [:,1])
roc_auc2=auc(fpr2,tpr2);
print("Area under cuver is :%f" % roc_auc2);
print("fpr is {}" ,fpr2);
print("thershold is {}" ,thrshold2);
print("tpr is {}" ,tpr2);
print("Area under cuver is :%f" % roc_auc1);
print("Area under cuver is :%f" % roc_auc2);




plt.show();