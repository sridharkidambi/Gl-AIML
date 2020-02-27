# ** Import pandas as pd.**
import pandas as pd
# ** Read games.csv as a dataframe called games.**
auto=pd.read_csv("games.csv")
print(auto)
# ** Check the head of the DataFrame. **
print(auto.head())
print('Info:')
print(auto.info())
# **What is the mean playin time for all games put together ?**
print(auto['playingtime'].mean())
# ** What is the highest number of comments received for a game? **
print(auto['total_comments'].max())
print(auto[auto['total_comments']== auto['total_comments'].max()])
# ** What is the name of the game with id 1500?  **
print(auto[auto["id"]== 1500]["name"])
# ** And which year was it published? **
print(auto[auto["id"]== 1500]["yearpublished"])
# ** Which game has received highest number of comments? **
print(auto[auto['total_comments']== auto['total_comments'].max()]["name"])
# ** What was the average minage of all games per game "type"? (boardgame & boardgameexpansion)**
print(auto.groupby("type").mean())
print(auto.groupby("type").mean()["minage"])
# ** How many unique games are there in the dataset? **
print(auto["id"].nunique())
print(auto["name"].nunique())
# ** How many boardgames and boardgameexpansions are there in the dataset?  **
print(auto["type"].value_counts())
print("print type")
# print(auto.groupby("type"))
# print("print")
# print(auto.groupby("type"))
# ** Is there a correlation between playing time and total comments for the games? - Use the .corr() function **
print(auto[["playingtime","total_comments"]]    )
print(auto[["playingtime","total_comments"]].corr())
