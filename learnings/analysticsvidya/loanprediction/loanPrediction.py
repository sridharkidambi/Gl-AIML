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

# Ensemble Learning - Bagging
from sklearn.ensemble import BaggingClassifier

# Ensemble Learning - AdaBoosting
from sklearn.ensemble import AdaBoostClassifier

# Ensemble Learning - GradientBoost
from sklearn.ensemble import GradientBoostingClassifier

# Ensemble RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

# loading the datset
df_loan_train=pd.read_csv("train_ctrUa4K.csv");
df_loan_test=pd.read_csv("test_lAUu6dG.csv");

print(df_loan_train.head(5));
print(df_loan_train.info());
print(df_loan_train.describe().transpose());
print(df_loan_train.Gender.value_counts());
print(df_loan_train.Married.value_counts());

for item in df_loan_train.columns:
    if(df_loan_train[item].dtype =='object'):
        df_loan_train[item]=pd.Categorical(df_loan_train[item])
print(df_loan_train.info());

print(df_loan_train.apply(lambda x: sum(x.isnull())));

# sns.pairplot(df_loan_train);
# plt.show()
