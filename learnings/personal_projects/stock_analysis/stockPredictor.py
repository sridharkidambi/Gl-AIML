import pandas as pd;
import numpy as np;
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

stock_file =pd.read_csv("RELIANCE.NS.csv")

print(stock_file.head())

sns.scatterplot(stock_file.Date,stock_file.Open);
plt.show();

sns.scatterplot(stock_file.Date,stock_file.Close);
plt.show();

