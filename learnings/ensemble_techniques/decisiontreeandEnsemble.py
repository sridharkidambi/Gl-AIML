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



df_creditData=pd.read_csv("credit.csv");
print(df_creditData.head(5));
print(df_creditData.shape);
print(df_creditData.describe().transpose());
print(df_creditData.dtypes);

for item in df_creditData.columns:
    if(df_creditData[item].dtype =='object'):
        df_creditData[item]=pd.Categorical(df_creditData[item])
print(df_creditData.head(10));

print(df_creditData["checking_balance"].value_counts());
print(df_creditData["purpose"].value_counts());
print(df_creditData["credit_history"].value_counts());
print(df_creditData["savings_balance"].value_counts());
print(df_creditData["employment_duration"].value_counts());
print(df_creditData["other_credit"].value_counts());
print(df_creditData["existing_loans_count"].value_counts());
print(df_creditData["phone"].value_counts());
print(df_creditData["housing"].value_counts());
print(df_creditData["job"].value_counts());
print(df_creditData["default"].value_counts());

# custom implementation of dummy variables 
replaceStruct = {
                "checking_balance":     {"< 0 DM": 1, "1 - 200 DM": 2 ,"> 200 DM": 3 ,"unknown":-1},
                "credit_history": {"critical": 1, "poor":2 , "good": 3, "very good": 4,"perfect": 5},
                 "savings_balance": {"< 100 DM": 1, "100 - 500 DM":2 , "500 - 1000 DM": 3, "> 1000 DM": 4,"unknown": -1},
                 "employment_duration":     {"unemployed": 1, "< 1 year": 2 ,"1 - 4 years": 3 ,"4 - 7 years": 4 ,"> 7 years": 5},
                "phone":     {"no": 1, "yes": 2 },
                #"job":     {"unemployed": 1, "unskilled": 2, "skilled": 3, "management": 4 },
                "default":     {"no": 0, "yes": 1 } 
                }
oneHotCols=["purpose","housing","other_credit","job"]

df_creditData=df_creditData.replace(replaceStruct)
df_creditData=pd.get_dummies(df_creditData, columns=oneHotCols)
df_creditData.head(10)

## Split Data
X = df_creditData.drop("default" , axis=1)
y = df_creditData.pop("default")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)

dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)

print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))

train_char_label = ['No', 'Yes']
Credit_Tree_File = open('credit_tree.dot','w')
dot_data = tree.export_graphviz(dTree, out_file=Credit_Tree_File, feature_names = list(X_train), class_names = list(train_char_label))
Credit_Tree_File.close()

# http://webgraphviz.com/ graph visualization
#Works only if "dot" command works on you machine

# retCode = system("dot -Tpng credit_tree.dot -o credit_tree.png")
# if(retCode>0):
#     print("system command returning error: "+str(retCode))
# else:
#     display(Image("credit_tree.png"))

## Reducing over fitting (Regularization)
dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=1)
dTreeR.fit(X_train, y_train)
print(dTreeR.score(X_train, y_train))
print(dTreeR.score(X_test, y_test))

train_char_label = ['No', 'Yes']
Credit_Tree_FileR = open('credit_treeR.dot','w')
dot_data = tree.export_graphviz(dTreeR, out_file=Credit_Tree_FileR, feature_names = list(X_train), class_names = list(train_char_label))
Credit_Tree_FileR.close()

#Works only if "dot" command works on you machine

# retCode = system("dot -Tpng credit_treeR.dot -o credit_treeR.png")
# if(retCode>0):
#     print("system command returning error: "+str(retCode))
# else:
#     display(Image("credit_treeR.png"))

# importance of features in the tree building ( The importance of a feature is computed as the 
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(dTreeR.feature_importances_, columns = ["Imp"], index = X_train.columns))

print(dTreeR.score(X_test , y_test))
y_predict = dTreeR.predict(X_test)

cm=metrics.confusion_matrix(y_test, y_predict, labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')
# plt.show()

bgcl = BaggingClassifier(base_estimator=dTree, n_estimators=50,random_state=1)
#bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(X_train, y_train)

y_predict = bgcl.predict(X_test)

print(bgcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


abcl = AdaBoostClassifier(n_estimators=10, random_state=1)
#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)
abcl = abcl.fit(X_train, y_train)


y_predict = abcl.predict(X_test)
print(abcl.score(X_test , y_test))

cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


gbcl = GradientBoostingClassifier(n_estimators = 50,random_state=1)
gbcl = gbcl.fit(X_train, y_train)

y_predict = gbcl.predict(X_test)
print(gbcl.score(X_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


rfcl = RandomForestClassifier(n_estimators = 50, random_state=1,max_features=12)
rfcl = rfcl.fit(X_train, y_train)

y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')

plt.show()
