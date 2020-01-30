import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
### Import the seaborn library and set color codes as true

sns.set(color_codes=True)

games=pd.read_csv("games.csv")
### Drop na values for negating issues during visualization
games.dropna(inplace=True)
print(games.head(3))
### View the distance plot for minage
# sns.distplot(games["minage"])
# plt.show()
### Is there a linear relationship between Minage & average_rating?
# sns.jointplot(games["average_rating"],games["minage"])
# plt.show()
### Compare the relationship between playingtime , minage and average rating using pairplot
# sns.pairplot(games[["average_rating","minage","playingtime"]])
# plt.show()
### Compare type of game and playingtime using a stripplot
# sns.stripplot(games["type"],games["playingtime"],jitter=True)
# plt.show()
### Analyze the linear trend between playing time(less than 500 mins) and average_rating received for the same
sns.regplot(x="playingtime",y="average_rating",data=games[games["playingtime"]<500])
plt.show()