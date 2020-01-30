import numpy as np
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
sns.set(color_codes=True)
# %matplotlib  inline
df1=pd.read_csv("Automobile.csv")
# print(df1.head())
print('univarient distributions:')
# sns.distplot(df1["length"])
# plt.show()
sns.distplot(df1["normalized_losses"],kde=True,rug=False,fit=gamma,hist=False,bins=100)
plt.show()
print('Bi-varient  distributions:')
# sns.jointplot(df1["engine_size"],df1["horsepower"])
# sns.jointplot(df1["engine_size"],df1["horsepower"],kind="hex")
# plt.show()
# sns.jointplot(df1["engine_size"],df1["horsepower"],kind="resid")
# plt.show()
# sns.jointplot(df1["engine_size"],df1["horsepower"], kind="reg")
# plt.show()
# sns.jointplot(df1["engine_size"],df1["horsepower"], kind="kde")
# plt.show()

# sns.pairplot(df1[["normalized_losses","engine_size","horsepower"]])
# plt.show()
# sns.stripplot(df1["fuel_type"],df1["horsepower"])
# plt.show()
# sns.stripplot(df1["fuel_type"],df1["horsepower"],jitter=True)
# plt.show()
# sns.swarmplot(df1["fuel_type"],df1["horsepower"])
# plt.show()


# sns.boxplot(df1["number_of_doors"],df1["horsepower"])
# plt.show()
# sns.boxplot(df1["number_of_doors"],df1["horsepower"],hue=df1["fuel_type"])
# plt.show()

#multiple categories in the dataset;bootstrapping or aggregation  to compute confidence intereval around the estimated values and plots using the error bar.includes the zeros
# sns.barplot(df1["body_style"],df1["horsepower"],hue=df1["engine_location"])
# plt.savefig('sample.png')
# plt.show()
# countplot instead of continuos data in y axis we can have freq of values defined.
# sns.countplot(df1["body_style"],hue =df1["engine_location"])
# plt.show()
#point plot -> differs from barplot by showing only the points of highest
# sns.pointplot(df1["fuel_system"],df1["horsepower"],hue =df1["number_of_doors"])
# plt.show()

# sns.pointplot(df1["body_style"],df1["horsepower"],hue=df1["engine_location"])
# plt.show()
#factor plot ->between multiple categorical items from the dataset;its called multipanel categorical plot.
# sns.catplot(x="fuel_type",
#                y="horsepower",
#                col="engine_location",
#                data=df1,
#                kind="swarm"
#               )
# plt.show()
# sns.lmplot(x="horsepower",y="peak_rpm",data=df1)
# plt.show()
# sns.lmplot(x="horsepower",y="peak_rpm",data=df1,hue="fuel_type")
# plt.show()
