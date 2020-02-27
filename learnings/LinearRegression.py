import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

car_data= pd.read_csv("auto-mpg.csv")

car_data=car_data.drop("car name",axis=1)
car_data["origin"]=car_data["origin"].replace({1:"america",2:"europe",3:"asia"})
car_data= pd.get_dummies(car_data,columns=["origin"])
print(car_data.describe())
# horsepower missid in the describe()
hpIsDigit=pd.DataFrame(car_data.horsepower.str.isdigit())
print(car_data[hpIsDigit["horsepower"]==False])
print(car_data.median())

car_data=car_data.replace("?",np.NAN)
nafiller = lambda x : x.fillna(x.median())
car_data = car_data.apply(nafiller,axis=0)
car_data["horsepower"]=car_data["horsepower"].astype("float64")

print(car_data.head(5))
print(car_data.info())
print(car_data.describe())


# sns.pairplot(car_data,diag_kind= 'kde')
# plt.show()

x= car_data.drop(["origin_europe","mpg"], axis=1)
y=car_data[["mpg"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)

regression_model= LinearRegression()
regression_model.fit(x_train,y_train)

for idx,col_name in enumerate(x_train.columns):
    print("the coeff for {} is {} ".format(col_name,regression_model.coef_[0][idx]))

intercept=regression_model.intercept_[0]
print("the interceptor  for our model is {}".format(intercept))

print(regression_model.score(x_train,y_train))
print(regression_model.score(x_test,y_test))

poly =PolynomialFeatures(degree=2,interaction_only=True)
x_train2=poly.fit_transform(x_train)
x_test2=poly.fit_transform(x_test)

poly_clf=linear_model.LinearRegression()
poly_clf.fit(x_train2,y_train)
y_pred=poly_clf.predict(x_test2)

print(poly_clf.score(x_train2,y_train))
print(poly_clf.score(x_test2,y_test))

print(x_train.shape)
print(x_train2.shape)