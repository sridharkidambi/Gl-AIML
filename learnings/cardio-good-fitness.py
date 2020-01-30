import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model



cardio_file=pd.read_csv("CardioGoodFitness.csv")
print(cardio_file.head())


# print(cardio_file.describe(include="all"))
# print(cardio_file.describe())
# print(cardio_file.info())
# cardio_file.hist(figsize=(20,30))
# sns.boxplot(x="MaritalStatus",y="Income",data=cardio_file, hue="Gender")
print(pd.crosstab(cardio_file["Product"],cardio_file["MaritalStatus"]))
# sns.countplot(x="Product", hue="Gender",data=cardio_file)
# print(pd.pivot_table(cardio_file,index=["Product","Gender"],columns=["MaritalStatus"],aggfunc="mean"))
# print(pd.pivot_table(cardio_file,values=["Income","Miles"], index=["Product","Gender"],columns=["MaritalStatus"],aggfunc="mean"))
# sns.pairplot(cardio_file)
# print(cardio_file["Income"].mean())
# print(cardio_file["Income"].median())
# sns.distplot(cardio_file["Age"])
# cardio_file.hist(by="Gender",column="Age")
# cardio_file.hist(by="Product",column="Miles",figsize=(20,30))
# print(cardio_file.corr())


# corr=cardio_file.corr()
# sns.heatmap(corr,annot=True)

# regr=linear_model.LinearRegression()
# y=cardio_file["Miles"]
# x=cardio_file[["Usage","Fitness"]]
# print(regr.fit(x,y))
# print(regr.coef_)
# print(regr.intercept_)


plt.show()
