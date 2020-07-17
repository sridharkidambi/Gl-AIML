import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import os;
from apyori import apriori;

dataset=pd.read_csv('Market_Basket_Optimisation.csv');

print(dataset.head())
print(dataset.shape)

transactions=[]
# transactions=dataset.values.tolist()
for i in range(0,7500):
    for j in range(0,20):
        # print(str(dataset.values[i,j]))
        transactions.append(str(dataset.values[i,j]))

# print(transactions)
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2);
# results = list(rules);

# print(results)

# for item in results:
#     pair=item[0];
#     item =[x for x in pair]
#     print("Rule :"+item[0]+ "->"+ item[1])

#     print("Confidence "+ str(item[2][0][2]))
#     print("Support "+ str(item[1]))
#     print("Lift  "+ str(item[2][0][3]))

# print(transactions)
# print(str(dataset.values[0][0]))