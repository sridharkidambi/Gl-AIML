# 1. Import the necessary libraries (2 marks)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from   scipy import stats
from   statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
# import statsmodels.api         as     sm
# from   statsmodels.formula.api import ols
import copy
from sklearn.preprocessing import LabelEncoder
# 2. Read the data as a data frame (2 marks)

df_insurance= pd.read_csv("insurance.csv")

# a. Shape of the data (2 marks)
print(df_insurance.shape)

# b. Data type of each attribute (2 marks)
print(df_insurance.info())

# c. Checking the presence of missing values (3 marks)
print(df_insurance.isna().apply(pd.value_counts))

# d. 5 point summary of numerical attributes (3 marks)
print(df_insurance.describe().T)

# e. Distribution of ‘bmi’, ‘age’ and ‘charges’ columns. (4 marks)
# plt.figure(figsize= (25,15))
# plt.subplot(1,3,1)
# plt.hist(df_insurance.bmi,  alpha = 0.7)
# plt.xlabel('bmi-variable')

# plt.subplot(1,3,2)
# plt.hist(df_insurance.age, alpha = 0.7)
# plt.xlabel('age-variable')

# plt.subplot(1,3,3)
# plt.hist(df_insurance.charges, color='lightblue', edgecolor = 'black', alpha = 0.7)
# plt.xlabel('charges-variable')

# plt.show()


# f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns (2 marks)

Varaible_skewness =pd.DataFrame({
    'skewness': [stats.skew(df_insurance["bmi"]),stats.skew(df_insurance["age"]),stats.skew(df_insurance["charges"])]
},index=["bmi","age","charges"])
print(Varaible_skewness)
# Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns (4 marks)
# plt.subplot(1,3,1)
# sns.boxplot(x=df_insurance["bmi"])
# plt.subplot(1,3,2)
# sns.boxplot(x=df_insurance["age"])
# plt.subplot(1,3,3)
# sns.boxplot(x=df_insurance["charges"])
# plt.show()

# h. Distribution of categorical columns (include children) (4 marks)


# x = df_insurance.smoker.value_counts().index    
# y = [df_insurance['smoker'].value_counts()[i] for i in x]   

# plt.subplot(2,2,1)
# plt.bar(x,y)  #plot a bar chart
# plt.xlabel('Smoker')
# plt.ylabel('Count')
# plt.title('Smoker distribution')


# x1 = df_insurance.sex.value_counts().index    
# y1 = [df_insurance['sex'].value_counts()[i] for i in x1]   
# plt.subplot(2,2,2)
# plt.bar(x1,y1)  #plot a bar chart
# plt.xlabel('sex')
# plt.ylabel('Count')
# plt.title('sex distribution')


# x2 = df_insurance.region.value_counts().index    
# y2 = [df_insurance['region'].value_counts()[i] for i in x2]   
# plt.subplot(2,2,3)
# plt.bar(x2,y2)  #plot a bar chart
# plt.xlabel('region')
# plt.ylabel('Count')
# plt.title('region distribution')


# x3 = df_insurance.children.value_counts().index    
# y3 = [df_insurance['children'].value_counts()[i] for i in x3]   
# plt.subplot(2,2,4)
# plt.bar(x3,y3)  #plot a bar chart
# plt.xlabel('children')
# plt.ylabel('Count')
# plt.title('children distribution')

# plt.show()

# i. Pair plot that includes all the columns of the data frame (4 marks)

# df_insurance_encoded = copy.deepcopy(df_insurance)
# df_insurance_encoded.loc[:,['sex', 'smoker', 'region']] = df_insurance_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform)
# sns.pairplot(df_insurance_encoded)
# plt.show()

# Do charges of people who smoke differ significantly from the people who don't? (7 marks)

# sns.scatterplot(df_insurance.age,df_insurance.charges,hue=df_insurance.smoker)
# plt.show()

# h0: The charges dont differ between people who smoke and dont.
# HA:The charges differ differ between people who smoke and dont.

# smoking_people=np.array( df_insurance[df_insurance["smoker"] =="yes"].charges)
# no_smoking_people=np.array(df_insurance[df_insurance["smoker"] =="no"].charges)

# t,p_value=stats.ttest_ind(smoking_people,no_smoking_people)

# if (p_value< .05):
#     print('Reject the Null hypothesis->The charges differ for people who smoke and dont.')
# else:
#     print('Fail to Reject the Null hypothesis->The charges dont differ for people who smoke and dont.')

# b. Does bmi of males differ significantly from that of females? (7 marks)

# sns.scatterplot(df_insurance.age,df_insurance.bmi,hue=df_insurance.sex)
# plt.show()

# h0: The BMI has  no influence  on gender.
# HA:The BMI do has  influence  on gender.

male_bmis=np.array( df_insurance[df_insurance["sex"] =="male"].bmi)
female_bmis=np.array( df_insurance[df_insurance["sex"] =="female"].bmi)

t,p_value=stats.ttest_ind(male_bmis,female_bmis)

print(p_value)
if (p_value< .05):
    print('Reject the Null hypothesis->The BMI has influence on gender.')
else:
    print('Fail to Reject the Null hypothesis->The BMI has  no influence on gender.')

# c. Is the proportion of smokers significantly different in different genders? (7 marks)

# H0: Proportion of smokers doesnt differ significantly in different genders.
# HA: Proportion of smokers do  differ significantly in different genders.

smokers_sex_link=pd.crosstab(df_insurance["sex"],df_insurance["smoker"])

chi, p_value, dof, expected = stats.chi2_contingency(smokers_sex_link)

if (p_value< .05):
    print('Reject the Null hypothesis->Proportion of smokers do  differ significantly in different genders.')
else:
    print('Proportion of smokers doesnt differ significantly in different genders.')


# d. Is the distribution of bmi across women with no children, one child and two children, the same? (7 marks)

# H0: The distrubition of bmi across women with no children, one child and two children is the same
# HA: The distrubition of bmi across women with no children, one child and two children is NOT the same

df_females= copy.deepcopy(df_insurance[df_insurance["sex"]=="female"])

One_child_females=df_females[df_females.children==1]["bmi"]
Two_child_females=df_females[df_females.children==2]["bmi"]
Three_child_females=df_females[df_females.children==3]["bmi"]
f_stat, p_value = stats.f_oneway(One_child_females,Two_child_females,Three_child_females)

if (p_value< .05):
    print('Reject the Null hypothesis->The distrubition of bmi across women with no children, one child and two children is NOT the same.')
else:
    print('The distrubition of bmi across women with no children, one child and two children is the same')
