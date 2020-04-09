#%%
from sklearn.tree import DecisionTreeClassifier
import pandas as pd;
import numpy as np;
import seaborn as sns;
import matplotlib.pylab as plt;
from sklearn import linear_model;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import roc_curve,auc;
from sklearn.naive_bayes import GaussianNB;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, model_selection, svm
from sklearn.linear_model import LogisticRegression
#%%
df_bank_personalLoan=pd.read_csv("Bank_Personal_Loan_Modelling.csv");
print(df_bank_personalLoan.head())
print(df_bank_personalLoan.columns)
print(df_bank_personalLoan.shape);
print(df_bank_personalLoan.info());
print(df_bank_personalLoan.apply(lambda x: sum(x.isnull())));
print(df_bank_personalLoan.describe().transpose());
print(df_bank_personalLoan.apply(lambda x: sum(x.isnull())));

sns.pairplot(df_bank_personalLoan.iloc[: , 1:6]);
sns.pairplot(df_bank_personalLoan.iloc[: , 6:]);
# plt.show();

# inference from the data:
# 1.Motgrage is right skewed  as most of the people who had mortgrage are within 100K and they vary widely until ~ 600k
# CC Avg is right skewed and most of the customers do have have a marajity of people do have  of less than or equal to  2.5k$ and they spread.
#  Age is evenly distributed with most of the people fall between  30 and 58.
#  Income is righ skewed with the majority drawing a salary of less thank 100k
# Experience is normally distributed.negative values are junk data.most customers have experience between 2 and 39.
# family ande education are ordinal variables and they are evenly distributed

# removing the negative experience and assign it the mean value
print(df_bank_personalLoan[df_bank_personalLoan["Experience"]<0]["Experience"].count());

df_exp_greaterZero=df_bank_personalLoan.loc[df_bank_personalLoan["Experience"]>0]
# print(df_exp_greaterZero)
df_exp_lessZero=df_bank_personalLoan.Experience <0
# print(df_exp_lessZero)
neg_list=df_bank_personalLoan.loc[df_exp_lessZero]["ID"].tolist(); 
# print(neg_list)

print(df_exp_lessZero.value_counts())

for item in neg_list:
    age=df_bank_personalLoan.loc[np.where(df_bank_personalLoan["ID"]==item)]["Age"].tolist()[0];
    education=df_bank_personalLoan.loc[np.where(df_bank_personalLoan["ID"]==item)]["Education"].tolist()[0];
    df_similar_records=df_exp_greaterZero[(df_exp_greaterZero.Age == age) & (df_exp_greaterZero.Education == education)]
    median_exp=df_similar_records["Experience"].median();
    df_bank_personalLoan.loc[df_bank_personalLoan.loc[np.where(df_bank_personalLoan["ID"]== item)].index,"Experience"]=median_exp
    
print(df_bank_personalLoan[df_bank_personalLoan["Experience"]<0]["Experience"].count());
print(df_bank_personalLoan.describe().transpose());

# effect of eduction and income on personal loan
sns.boxplot(x="Education",y="Income",hue="Personal Loan",data=df_bank_personalLoan)
# persons who have personal loan have same income levels irrespective of their education.People having Education 1 have more income oppurtunities.

sns.boxplot(x="Education",y="Mortgage",hue="Personal Loan",data=df_bank_personalLoan)
# Mortgage is common for customers who have personal loan and no personal loan.Persons with education level 1 have comparitive more personal loans.

sns.boxplot(x="Experience",y="Securities Account",hue="Personal Loan",data=df_bank_personalLoan)
# not proper can be ignored.

sns.countplot(x="Securities Account", data=df_bank_personalLoan,hue="Personal Loan")
# people not having secuties do have personal loan.

sns.countplot(x='Family',data=df_bank_personalLoan,hue='Personal Loan')
# Family size has no major impact for taking personal loans.family size of 3 have an better oppurtunity for taking personal loan and can be used in future decisioning systems.

sns.countplot(x='CD Account',data=df_bank_personalLoan,hue='Personal Loan');
# Customers who dont have a CD account dont have a personal loan too.However same number so custoemrs do exist for people having a CD account and personal loan.

sns.distplot(df_bank_personalLoan[df_bank_personalLoan["Personal Loan"] == 0]['CCAvg'],color="r")
sns.distplot(df_bank_personalLoan[df_bank_personalLoan["Personal Loan"] == 1]['CCAvg'],color="b")

print('Creditcard spending non-persoanl loan customers: ', df_bank_personalLoan[df_bank_personalLoan["Personal Loan"]== 0]["CCAvg"].median(),'k$')
print('Creditcard spending persoanl loan customers: ', df_bank_personalLoan[df_bank_personalLoan["Personal Loan"]== 1]["CCAvg"].median(),'k$')

# The count shows people having high cc average has more personal loan chances compared to people having lower credit card spending.
fig ,ax =plt.subplots()
colors={1:"green",2:"red",3:"yellow"}
ax.scatter(df_bank_personalLoan["Experience"],df_bank_personalLoan["Age"],c=df_bank_personalLoan["Education"].apply(lambda x:colors[x]));
plt.xlabel("Experience");
plt.ylabel("Age");
# There is no people beyond age 45 having personal loans
# there is a gap in expereince for the people with undergraduate education.

corr =df_bank_personalLoan.corr()
sns.set_context("notebook",font_scale=1.0,rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7));

mask=np.zeros_like(corr);
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
#  The more the income  the more the spending on creditcard and they are slightly +vely correlated
#  Age and Expereince afe very closely correlated

sns.boxplot(x=df_bank_personalLoan.Family,y=df_bank_personalLoan["Income"],hue=df_bank_personalLoan["Personal Loan"])
# The families with income less than $100k dont take loan and families with higher income take loans.

# plt.show();

# Split the Data
# Since aga and Experince are highly corelated we can reduce the dimension by 1 i.e removing the Experience
x= df_bank_personalLoan.drop(['ID','Experience'], axis=1)
y=df_bank_personalLoan[["Personal Loan"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=100)

# LogisticRegression
print('Logistic Regression Model');
logistic_reg_model = LogisticRegression(solver='liblinear');
logistic_reg_model_2 = svm.SVC(kernel='linear',probability=True);

logistic_reg_model.fit(x_train,y_train["Personal Loan"]);
logistic_reg_model_2.fit(x_train,y_train["Personal Loan"]);

y_predict_1 =logistic_reg_model.predict(x_test);
y_predict_2 =logistic_reg_model_2.predict(x_test);

y_predict_proba_1 =logistic_reg_model.predict_proba(x_test);
y_predict_proba_2 =logistic_reg_model_2.predict_proba(x_test);

print(logistic_reg_model.score(x_test,y_test["Personal Loan"])*100,'%.');
print(logistic_reg_model_2.score(x_test,y_test["Personal Loan"])*100,'%.');


print(y_predict_proba_2[:5])
print(y_test.head(5));

# ROC curve for y_predict_1 i

fpr1,tpr1,thrshold1=roc_curve(y_test, y_predict_proba_1 [:,1])
roc_auc1=auc(fpr1,tpr1);
print("Area under cuver is :%f" % roc_auc1);

fpr2,tpr2,thrshold2=roc_curve(y_test, y_predict_proba_2 [:,1])
roc_auc2=auc(fpr2,tpr2);
print("Area under cuver is :%f" % roc_auc2);
print("Area under cuver is :%f" % roc_auc1);
print("Area under cuver is :%f" % roc_auc2);
print(type(y_test))
print(type(y_predict_proba_1))
cm=metrics.confusion_matrix(y_test,y_predict_1,labels=(1,0));
print(cm);
df_cm=pd.DataFrame(cm,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
plt.figure(figsize=(7,5));
sns.heatmap(df_cm,annot=True);
# plt.show();

# Naive bayes
print('Naive bayes Model');

naive_model =GaussianNB();
naive_model.fit(x_train,y_train["Personal Loan"]);
print(naive_model.score(x_test,y_test["Personal Loan"])*100,'%.');
y_naive_predict=naive_model.predict(x_test);
print(y_naive_predict[:5])
print(y_test.head(5));
print('Naive bayes score is',naive_model.score(x_test,y_test)*100,'%');


cm=metrics.confusion_matrix(y_test,y_naive_predict,labels=(1,0));
print(cm);
df_cm=pd.DataFrame(cm,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
plt.figure(figsize=(7,5));
sns.heatmap(df_cm,annot=True);
plt.show();


fpr1,tpr1,thrshold1=roc_curve(y_test, y_naive_predict)
roc_auc1=auc(fpr1,tpr1);
print("Area under cuver is :%f" % roc_auc1);


#  K-NN Model
print('K-NN Model');
knn = KNeighborsClassifier(n_neighbors= 5 , weights = 'uniform', metric='euclidean')
knn.fit(x_train, y_train["Personal Loan"])    
knn_predicted = knn.predict(x_test)
acc = accuracy_score(y_test, knn_predicted)
print(acc)
print('K-NN score is',knn.score(x_test,y_test)*100,'%');

fpr1,tpr1,thrshold1=roc_curve(y_test, knn_predicted)
roc_auc1=auc(fpr1,tpr1);
print("Area under cuver is :%f" % roc_auc1);

cm=metrics.confusion_matrix(y_test,knn_predicted,labels=(1,0));
print(cm);
df_cm=pd.DataFrame(cm,index=[i for i in ['Predict 1 ','Predict 0']],  columns=[i for i in ['Predict 1 ','Predict 0']]);
plt.figure(figsize=(7,5));
sns.heatmap(df_cm,annot=True);
plt.show();

X=df_bank_personalLoan.drop(['Personal Loan','Experience','ID'],axis=1);
y=df_bank_personalLoan.pop('Personal Loan')
models=[];
models.append(('LR',LogisticRegression()));
models.append(('NB',GaussianNB()));
models.append(('KNN',KNeighborsClassifier()));
results=[];
names =[];
scoring="accuracy";

for name ,model in models:
    kfold=model_selection.KFold(n_splits=10, random_state=12345);
    cv_results=model_selection.cross_val_score(model,X,y,cv=kfold,scoring=scoring);
    results.append(cv_results);
    names.append(name);
    print(name, cv_results.mean(), cv_results.std());

    # LogisticRegression holds better performance than others.

