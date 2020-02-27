import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from   scipy import stats
from   statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
import statsmodels.api         as     sm
from   statsmodels.formula.api import ols
import copy
from sklearn.preprocessing import LabelEncoder
from   statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

top=    [8,7,6,7,9,9,10]
middle= [8,7,6,9,10,8,np.NaN]
bottom= [5,6,7,6,7,np.NaN,np.NaN]
# top=    [8,7,6,7,9]
# middle= [8,7,6,9,10]
# bottom= [5,6,7,6,7]
col_mean = np.nanmean(middle, axis=0)
print (col_mean)
# inds = np.where(np.isnan(middle))
# print(inds[0][0])
# print(middle[inds[0][0]])
# middle[inds[0][0]] = np.take(col_mean, inds[0][0])


df_employees= pd.DataFrame()
df_employees["top"]=top
df_employees["middle"]=middle
df_employees["bottom"]=bottom
# df_employees["middle"].isna().apply(col_mean)

print(df_employees["middle"].describe())
print(df_employees.isna().apply(pd.value_counts))


# H0: The mean of scores  across top midlle and bottom training is equal
# HA: The mean of scores  across top midlle and bottom training is not equal


f_stat, p_value = stats.f_oneway(top,middle,bottom)

print(f_stat)
print(p_value)
if (p_value< .05):
    print('Reject the Null hypothesis->The mean of scores  across top midlle and bottom training is not equal.')
else:
    print('The mean of scores  across top midlle and bottom training is equal')


# Assuming level of significance as 0.05, formulate the null and alternative hypotheses and determine which test statistic needs to be used. Also create a Decision Rule.

df_anova=pd.read_csv("ANOVA.csv")
# print(df_anova.head(500))
formula = 'Qty ~ C(Loc) + C(Brand)'
model = ols(formula, df_anova).fit()
aov_table = anova_lm(model, typ=2)

print(aov_table)

print(pairwise_tukeyhsd(df_anova['Qty'], df_anova['Loc']))
print(pairwise_tukeyhsd(df_anova['Qty'], df_anova['Brand']))
print(pairwise_tukeyhsd(df_anova['Loc'], df_anova['Brand']))