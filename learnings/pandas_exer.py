import numpy as np
import pandas as pd

# **Check the head of the DataFrame.**
auto=pd.read_csv("Automobile.csv")
print(auto.head())
# ** How many rows and columns are there? **
print(auto.info())
# ** What is the average Price of all cars in the dataset? **
print(auto["price"].mean())
# **Which is the cheapest make and costliest make of car in the lot?**
print(auto[auto["price"]== auto["price"].min()])
print(auto[auto["price"]== auto["price"].max()])
# ** How many cars have horsepower greater than 100? **
print(auto[auto["horsepower"]>100])

print(auto[auto["horsepower"]>100].count())
# ** How many hatchback cars are in the dataset ? **
print(auto[auto["body_style"]=="hatchback"].count())
print(auto[auto["body_style"]=="hatchback"].info())
# ** What are the 3 most commonly found cars in the dataset? **
print(auto['make'].value_counts().head(3))
# ** Someone purchased a car for 7099, what is the make of the car? **
print(auto[auto["price"] == 7099]["make"])
# *** Which cars are priced greater than 40000? **
print(auto[auto["price"]>40000])
# ** Which are the cars that are both a sedan and priced less than 7000? **
print(auto[(auto["price"]<7000) & (auto["body_style"]=="sedan")])





