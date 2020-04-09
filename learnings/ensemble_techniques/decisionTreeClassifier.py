import pandas  as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import random;
import seaborn as sns;
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier

num_bins = 10;
df_diabetics =pd.read_csv("pima-indians-diabetes.csv");
print(df_diabetics.head(10));
print(df_diabetics.shape);
print(df_diabetics.dtypes);
print(df_diabetics.columns);

df_diabetics.loc[df_diabetics.Plas == 0, 'Plas'] = df_diabetics.Plas.median()
df_diabetics.loc[df_diabetics.Pres == 0, 'Pres'] = df_diabetics.Pres.median()
df_diabetics.loc[df_diabetics.skin == 0, 'skin'] = df_diabetics.skin.median()
df_diabetics.loc[df_diabetics.test == 0, 'test'] = df_diabetics.test.median()
df_diabetics.loc[df_diabetics.mass == 0, 'mass'] = df_diabetics.mass.median()

print(df_diabetics.describe().transpose());
print(df_diabetics["class"].value_counts())
print(df_diabetics.groupby("class").agg({'class': 'count'}));
# sns.kdeplot(df_diabetics, cumulative=True, bw=1.5)
sns.pairplot(df_diabetics);
# sns.pairplot(df_diabetics, hue="class", palette="husl");
df_diabetics.corr();
plt.show();

# splitting data into training and test set for independent attributes
n=df_diabetics['class'].count()
train_set = df_diabetics.head(int(round(n*0.7))) # Up to the last initial training set row
test_set = df_diabetics.tail(int(round(n*0.3))) # Past the last initial training set row

# capture the target column ("class") into separate vectors for training set and test set
train_labels = train_set.pop("class")
test_labels = test_set.pop("class")

dt_model = DecisionTreeClassifier(criterion = 'entropy'  )
dt_model.fit(train_set, train_labels)

dt_model.score(test_set , test_labels)
test_pred = dt_model.predict(test_set)

print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = train_set.columns))#Print the feature importance of the decision model