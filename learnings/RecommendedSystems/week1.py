import os
import numpy as np  
import pandas as pd

# os.chdir('/')

ratings_data = pd.read_csv("ratings.csv")  
ratings_data.head() 

movie_names = pd.read_csv("movies.csv")  
movie_names.head() 

movie_data = pd.merge(ratings_data, movie_names, on='movieId')  
print(movie_data.head(5))

print(movie_data.groupby('title')['rating'].mean().head())
print(movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print(movie_data.groupby('title')['rating'].count())
print('data set movie')
print(movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head())
ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean().sort_values(ascending=False)) 
print(ratings_mean_count)
ratings_mean_count['rating_counts']=pd.DataFrame(movie_data.groupby('title')['rating'].count().sort_values(ascending=False))
print(ratings_mean_count)
