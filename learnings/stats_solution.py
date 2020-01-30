import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sample_data=pd.read_csv("Inc_Exp_Data.csv")
print(sample_data.head())
print(sample_data.describe())
print(sample_data["Highest_Qualified_Member"].value_counts())

# 4.What is the Mean Expense of a Household?
print(sample_data["Mthly_HH_Expense"].mean())

# 5.What is the Median Household Expense?
print(sample_data["Mthly_HH_Expense"].median())

#  6.What is the Monthly Expense for most of the Households?
mthly_expen_tmp=pd.crosstab(index=sample_data["Mthly_HH_Expense"],columns="count")
mthly_expen_tmp.reset_index(inplace=True)
# print(mthly_expen_tmp)
print(mthly_expen_tmp[mthly_expen_tmp["count"]==mthly_expen_tmp["count"].max()])

# 7.Plot the Histogram to count the Highest qualified member
sample_data["Highest_Qualified_Member"].value_counts().plot(kind="bar")
# sns.countplot(sample_data["Highest_Qualified_Member"])



# 8.Calculate IQR(difference between 75% and 25% quartile)
print("IQR:")
print((sample_data["Mthly_HH_Income"].quantile(.75))- (sample_data["Mthly_HH_Income"].quantile(.25)))

# 9.Calculate Standard Deviation for first 4 columns.
print((sample_data.iloc[:,0:5]).std())

# 10.Calculate Variance for first 3 columns.
print(sample_data.iloc[:,0:5].var())

print((pd.DataFrame(sample_data.iloc[:,0:3].var().to_frame()).T))

# 11.Calculate the count of Highest qualified member.

# sns.countplot(sample_data["Highest_Qualified_Member"])
# 12.Plot the Histogram to count the No_of_Earning_Members
sns.countplot(sample_data["No_of_Earning_Members"])


# 13.Suppose you have option to invest in Stock A or Stock B. The stocks have different expected returns and standard deviations. The expected return of Stock A is 15% and Stock B is 10%. Standard Deviation of the returns of these stocks is 10% and 5% respectively.
# Which is better investment?

plt.show()
