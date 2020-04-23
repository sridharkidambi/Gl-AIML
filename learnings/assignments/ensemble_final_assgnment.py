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
from dataclasses import replace
import os
import sys;
# Ensemble Learning - Bagging
from sklearn.ensemble import BaggingClassifier

# Ensemble Learning - AdaBoosting
from sklearn.ensemble import AdaBoostClassifier

# Ensemble Learning - GradientBoost
from sklearn.ensemble import GradientBoostingClassifier

# Ensemble RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from confusionMatrix import plotConfusionMatrix,dtree_grid_search,RF_grid_search;
# from confusionMatrix import dtree_grid_search;
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost  import XGBClassifier 


df_bank_data= pd.read_csv('/Users/sridharkidambi/python/Gl-AIML/learnings/files/ensemble/bank-full.csv')

print(df_bank_data.head(5));
print(df_bank_data.info());
print(df_bank_data.describe().transpose());

for item in df_bank_data.columns:
    if(df_bank_data[item].dtype =='object'):
        df_bank_data[item]=pd.Categorical(df_bank_data[item])
print(df_bank_data.head(5));
print(df_bank_data.info());

# understand the values
print(df_bank_data.job.value_counts());
print(df_bank_data.marital.value_counts());
print(df_bank_data.education.value_counts());
print(df_bank_data.default.value_counts());
print(df_bank_data.housing.value_counts());
print(df_bank_data.loan.value_counts());
print(df_bank_data.contact.value_counts());
print(df_bank_data.month.value_counts());
print(df_bank_data.poutcome.value_counts());
print(df_bank_data.Target.value_counts());

sns.pairplot(df_bank_data);
# plt.show();

# as the age increases the contact to thecustomer is gradually decreased for campaign
# people in their 50's are ready for the campaign talks.
# print(df_bank_data[df_bank_data['Target'] == 'yes'])
# print(df_bank_data[df_bank_data['Target'] == 'no'])
print('SK testing')
# checking null values and found no null values across fields.
print(df_bank_data.apply(lambda x: sum(x.isnull())));

# print()

sns.countplot(x="education",hue="Target",data=df_bank_data);
# plt.show();

sns.countplot(x="poutcome",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="month",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="contact",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="loan",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="housing",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="education",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="housing",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="default",hue="Target",data=df_bank_data);
# plt.show();
sns.countplot(x="marital",hue="Target",data=df_bank_data);
# plt.show();
# sns.countplot(x="job",hue="Target",data=df_bank_data);
# plt.show();

# secondary and teritary educated people have subscribed for the product compared to others.
# The outcome of previous campaign resulted in more subscription which needs to be unearthed(unknown)
# The may month has seen peek hike in the subscription and rejection compared to other months.
# The subcripton probability is more in cellular based customers.
# People having personal loan has shown less interest for the product subsciption than people having no personal loans.
# People having housing loan has shown less interest for the product subsciption than people having no Housing.
# people having credit card default has no subscription to this product.
# Married people have shown more interest to the product 
# people in management and techinician have shown more interst to the product.\\\\\\\\\\\\\\\k


corr =df_bank_data.corr()
sns.set_context("notebook",font_scale=1.0,rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7));

mask=np.zeros_like(corr);
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')


# custom implementation of dummy variables 
replaceStruct = {
                # "job":     {"< 0 DM": 1, "1 - 200 DM": 2 ,"> 200 DM": 3 ,"unknown":-1},
                "marital": {"married": 1, "single":2 , "divorced": 3},
                 "education": {"secondary": 1, "tertiary":2 , "primary": 3, "unknown": 4},
                 "housing":     {"no": 0, "yes": 1 },
                "loan":     {"no": 0, "yes": 1 },
                "contact":     {"unknown": 0, "cellular": 1, "telephone": 2 },
                "default":     {"no": 0, "yes": 1 }, 
                "poutcome":     {"unknown": 0, "failure": 1,"other": 2, "success": 3 }, 
                "Target":     {"no": 0, "yes": 1 } 
                }
oneHotCols=["job","month"]

df_bank_data=df_bank_data.replace(replaceStruct)
df_bank_data=pd.get_dummies(df_bank_data, columns=oneHotCols)
print(df_bank_data.info())
print(df_bank_data.head(5))
print(df_bank_data.Target.value_counts())
print(df_bank_data.Target.value_counts( normalize=True))


# split the data
X = df_bank_data.drop('Target', axis = 1)
y = df_bank_data['Target']

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# the percentage of poeple who have subscribed is only 11% we dont have sufficient data to proceed woth the  
print(y_train.value_counts(normalize=True));


dtree_model = DecisionTreeClassifier() # without any tweaking # no pruning 
dtree_model.fit(X_train, y_train)

print(dtree_model.score(X_train, y_train) )     
print(dtree_model.score(X_test, y_test)) 
predictions_ = dtree_model.predict(X_test) 
f1= f1_score(y_test,predictions_, average='micro')

plt.figure(figsize=(5,3))
cnf_mat = confusion_matrix(y_test,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
# plt.show();
print(cnf_mat)
print(classification_report(y_test, predictions_))

# ramdom forest without tweaking or pruning

RF_model = RandomForestClassifier() # without any tweaking # no pruning 
RF_model.fit(X_train, y_train)

RF_model.score(X_train, y_train) # overfitting - DT, imbalanced
RF_model.score(X_test, y_test)
predictions_RF = RF_model.predict(X_test) 
print(classification_report(y_test, predictions_RF))

# SMOTE with --
# 1. Grid Search CV

# 2. Cross Validation Score

sm = SMOTE(random_state = 2) # SMOTE works on the principle of KNN
X_sm, y_sm = sm.fit_sample(X, y)
print(X_sm.shape)
print(y_sm.value_counts(normalize=True))


X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_sm, y_sm, test_size = 0.3, random_state = 0)
print(X_train_res.shape)



best_param, acc_DT_gs, dt_gs = dtree_grid_search(X_train_res,y_train_res, 4) # after SMOTE
acc_val_DT_gs = dt_gs.score(X_val_res, y_val_res)
print(best_param)

print(acc_DT_gs, acc_val_DT_gs) 

y_pred_proba = dt_gs.predict_proba(X_val_res)[::,1]
fpr, tpr, _ = roc_curve(y_val_res,  y_pred_proba)
auc = roc_auc_score(y_val_res, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

predictions_ = dt_gs.predict(X_val_res) 
f1_DT_gs = f1_score(y_val_res,predictions_, average='micro')

plt.figure(figsize=(5,3))
cnf_mat = confusion_matrix(y_val_res,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1_DT_gs))

print(classification_report(y_val_res, predictions_))
# plt.show();

# ------------------- Ensemble -----------------------------------------------------------------------------------
rfcl = RandomForestClassifier(n_estimators = 50) # no. of trees = 50
rfcl.fit(X_train_res, y_train_res)

acc_RF = rfcl.score(X_train_res,y_train_res)
acc_val_RF = rfcl.score(X_val_res,y_val_res)
predictions_RF = rfcl.predict(X_val_res) 
f1_RF= f1_score(y_val_res,predictions_RF, average='micro')

print(classification_report(y_val_res, predictions_RF)) 

best_params, rf_gscv, score_gscv = RF_grid_search(X_train_res,y_train_res,4)
print(best_params)
acc_RF_gs = score_gscv
acc_val_RF_gs = rf_gscv.score(X_val_res,y_val_res)
predictions_RF_gs = rf_gscv.predict(X_val_res) 
f1_RF_gs = f1_score(y_val_res,predictions_RF_gs, average='micro')


abcl = AdaBoostClassifier( n_estimators= 100, learning_rate=0.1, random_state=22)
abcl.fit(X_train_res, y_train_res)


acc_AB = abcl.score(X_train_res,y_train_res)
acc_val_AB = abcl.score(X_val_res,y_val_res)
predictions_AB = abcl.predict(X_val_res) 
f1_AB= f1_score(y_val_res,predictions_AB, average='micro')

from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier(n_estimators=50, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)
bgcl.fit(X_train, y_train)

acc_BG = bgcl.score(X_train_res,y_train_res)
acc_val_BG = bgcl.score(X_val_res,y_val_res)
predictions_BG = bgcl.predict(X_val_res) 
f1_BG = f1_score(y_val_res,predictions_BG, average='micro')


xgboost = XGBClassifier(learning_rate =0.05, n_estimators=300, max_depth=5)
xgboost.fit(X_train_res, y_train_res)


acc_XGB = xgboost.score(X_train_res,y_train_res)
acc_val_XGB = xgboost.score(X_val_res,y_val_res)
predictions_XGB = xgboost.predict(X_val_res) 
f1_XGB = f1_score(y_val_res,predictions_XGB, average='micro')

d = {'Model': ['DT_gs', 'RF', 'RF_gs', 'AdaBoost', 'Bagging', 'XGBoost'], 
     'Train_Acc':[acc_DT_gs, acc_RF, acc_RF_gs, acc_AB, acc_BG, acc_XGB], 
     'Val_Acc': [acc_val_DT_gs, acc_val_RF, acc_val_RF_gs, acc_val_AB, acc_val_BG, acc_val_XGB], 
     'F1_score':[f1_DT_gs, f1_RF, f1_RF_gs, f1_AB, f1_BG, f1_XGB]}

# d = {'Model': ['DT_gs', 'RF', 'RF_gs', 'AdaBoost', 'Bagging'], 
#      'Train_Acc':[acc_DT_gs, acc_RF, acc_RF_gs, acc_AB, acc_BG], 
#      'Val_Acc': [acc_val_DT_gs, acc_val_RF, acc_val_RF_gs, acc_val_AB, acc_val_BG], 
#      'F1_score':[f1_DT_gs, f1_RF, f1_RF_gs, f1_AB, f1_BG]}
df_metrics = pd.DataFrame(d)
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
plt.plot(df_metrics['Model'], df_metrics['Train_Acc'], marker = '^')
plt.plot(df_metrics['Model'], df_metrics['Val_Acc'], marker = 'o')
plt.plot(df_metrics['Model'], df_metrics['F1_score'], marker = '>', linestyle='--')
plt.legend(fontsize = 15)
plt.show()
