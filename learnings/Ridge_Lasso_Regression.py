import pandas as pd;
import numpy as np;
import seaborn as sns
import matplotlib.pyplot as plt;
from sklearn.linear_model import Ridge;
from sklearn.linear_model import Lasso;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import r2_score;

from sklearn import preprocessing;

from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import PolynomialFeatures;


# Reading and cleaning the Data.

mpg_df=pd.read_csv("car-mpg.csv");
mpg_df=mpg_df.drop(labels="car_name",axis=1);
mpg_df["origin"]=mpg_df["origin"].replace({1:"america" ,2:"europe",3:"asia"});
mpg_df=pd.get_dummies(mpg_df,columns=["origin"]);
mpg_df=mpg_df.replace("?",np.nan);  
mpg_df=mpg_df.apply(lambda x: x.fillna(x.median()),axis=0);
print(mpg_df.head(5));

# separate independent and dependent variables

X=mpg_df.drop(columns=["mpg"]); 
y=mpg_df[["mpg"]];

X_scaled=preprocessing.scale(X);
X_scaled=pd.DataFrame(X_scaled,columns=X.columns);

y_scaled=preprocessing.scale(y);
y_scaled=pd.DataFrame(y_scaled,columns=y.columns)

print(X_scaled);
print(y_scaled);

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.30,random_state=1);

# fit a simple linear model

regressiobModel=LinearRegression();
regressiobModel.fit(X_train,y_train);

for idx,col_name  in enumerate(X_train.columns):
    print("The coefficients for the variable {} is :{} ".format(col_name,regressiobModel.coef_[0][idx]));

intercept=regressiobModel.intercept_[0];
print("The intercept for the model is : {} ".format(intercept));

# Create a regularized RIDGE model and note the coefficients

ridge=Ridge(alpha=.3);
ridge.fit(X_train,y_train);
print("Ridge Model is :",(ridge.coef_));


lasso=Lasso(alpha=0.2);
lasso.fit(X_train,y_train);
print("Lasso Coeff is :",(lasso.coef_));

## Let us compare their scores

print(regressiobModel.score(X_train,y_train));
print(regressiobModel.score(X_test,y_test));

print(ridge.score(X_train,y_train));
print(ridge.score(X_test,y_test));

print(lasso.score(X_train,y_train));
print(lasso.score(X_test,y_test));
# More or less similar results but with less complex models.  Complexity is a function of variables and coefficients
## Note - with Lasso, we get equally good result in test though not so in training.  Further, the number of dimensions is much less
# in LASSO model than ridge or un-regularized model

# Let us generate polynomial models reflecting the non-linear interaction between some dimensions

poly=PolynomialFeatures(degree=2,interaction_only=True);
X_poly=poly.fit_transform(X_scaled);

X_train,X_test,y_train,y_test=train_test_split(X_poly,y,test_size=0.30,random_state=1)
X_train.shape;
print(X_train.shape);
print(X_poly)

regressiobModel.fit(X_train,y_train);
print(regressiobModel.coef_[0]);

ridge=Ridge(alpha=0.3)
ridge.fit(X_train,y_train);
print("riddgecoeff  ",ridge.coef_);

lasso=Lasso(alpha=0.2)
lasso.fit(X_train,y_train)
print("Lasso coeff ",lasso.coef_);

print(regressiobModel.score(X_train,y_train))
print(regressiobModel.score(X_test,y_test))


print(ridge.score(X_train,y_train))
print(ridge.score(X_test,y_test))


print(lasso.score(X_train,y_train))
print(lasso.score(X_test,y_test))
