import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from scipy.stats import ttest_1samp,ttest_ind,f
from statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
import statsmodels.api         as     sm
from   statsmodels.formula.api import ols

# two sample testing independent samples.
k=np.array([
    [9.8,0],
    [8.81,1],
    [9.9,0],
    [8.91,1], 
    [9.2,0],
    [8.21,1],
    [9.7,0],
    [8.71,1]
])

k1=k[:,1]==0
print(k1)
group1=k[k1][:, 0]
print(group1)
k2=k[:,1]==1
group2=k[k2][:, 0]

t_statistics ,P_value = ttest_ind(group1,group2)
print(t_statistics)
print(P_value)

# two sample testing NOT independent samples.
energy=np.array([
    [1000,1100],
    [2000,2500],
    [3000,7500],
    [5000,9500], 
    [4000,6500],
    [4000,2500],
    [3000,5500],
    [2000,4500]
])

pre=energy[:,0]
post=energy[:,1]
t_statistics,P_value= ttest_1samp(post-pre,0)
print(t_statistics)
print(P_value)


# Comparision of propotions

df=pd.read_csv("insurance.csv")
print(df.head(5))

female_smks=df[df["sex"]=="female"].smoker.value_counts()[1]
print(female_smks)
male_smks=df[df["sex"]=="male"].smoker.value_counts()[1]
print(male_smks)

male_counts=df[df["sex"]=="male"].sex.value_counts()[0]
print(male_counts)

female_counts=df[df["sex"]=="female"].sex.value_counts()[0]
print(female_counts)

perct_female_smks=round((female_smks/female_counts),5)
perct_male_smks=round((male_smks/male_counts),5)
print(perct_female_smks)
print(perct_male_smks)
print(round((2/1),5))

stat,pvalue= proportions_ztest([female_smks,male_smks],[female_counts,male_counts])
print(stat)
print(pvalue)

# Variance test
nineteen=df[df["age"]==19]
print(nineteen.sex.value_counts())
sample_male_bmi=nineteen[nineteen["sex"]=="male"].bmi.iloc[:-2]
sample_female_bmi=nineteen[nineteen["sex"]=="female"].bmi
# print(sample_male_bmi)
# print(sample_female_bmi)
sample_male_bmi_var=np.var(sample_male_bmi)
sample_female_bmi_var=np.var(sample_female_bmi)
print(sample_male_bmi_var)
print(sample_female_bmi_var)

n=33
dof=n-1
alpha=.05
chi_critical=46.19

chi=(dof*sample_male_bmi_var)/sample_female_bmi_var
print(chi)

# Comparision of multiple samples -Anova
mean_pressure_compact_car    =  np.array([643, 655,702])
mean_pressure_midsize_car    =  np.array([469, 427, 525])
mean_pressure_fullsize_car   =  np.array([484, 456, 402])
print('Count, Mean and standard deviation of mean pressue exerted by compact car: %3d, %3.2f and %3.2f' % (len(mean_pressure_compact_car ), mean_pressure_compact_car .mean(),np.std(mean_pressure_compact_car ,ddof =1)))
print('Count, Mean and standard deviation of mean pressue exerted by midsize car: %3d, %3.2f and %3.2f' % (len(mean_pressure_midsize_car), mean_pressure_midsize_car.mean(),np.std(mean_pressure_midsize_car,ddof =1)))
print('Count, Mean and standard deviation of mean pressue exerted by full size car: %3d, %3.2f and %3.2f' % (len(mean_pressure_fullsize_car), mean_pressure_fullsize_car.mean(),np.std(mean_pressure_fullsize_car,ddof =1)))

mean_pressure_df = pd.DataFrame()

df1            = pd.DataFrame({'Car_Type': 'C', 'Mean_Pressure':mean_pressure_compact_car})
df2            = pd.DataFrame({'Car_Type': 'M', 'Mean_Pressure':mean_pressure_midsize_car})
df3            = pd.DataFrame({'Car_Type': 'F', 'Mean_Pressure':mean_pressure_fullsize_car})
mean_pressure_df = mean_pressure_df.append(df1) 
mean_pressure_df = mean_pressure_df.append(df2) 
mean_pressure_df = mean_pressure_df.append(df3)
sns.boxplot(x = "Car_Type", y = "Mean_Pressure", data = mean_pressure_df)
plt.title('Mean pressure exerted by car types')
plt.show()
mod = ols('Mean_Pressure ~ Car_Type', data = mean_pressure_df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)