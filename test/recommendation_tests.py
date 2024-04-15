#%%  Tests For Functional Requirements: test 1.1 (One Popular Movie)
# Please run tests in order

import sys
import pandas as pd
import random
import sklearn.metrics  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

sys.path.append('C:\\Users\\seyed\\Desktop\\RecommSys\\src')

from modules import Input_Format, MovieSelector

# load raw ratings
file_path = "user_item_test1_1.csv"  # Replace with your CSV file path
input_format = Input_Format(file_path)
dataframe = input_format.load_csv_dataset()
if dataframe is not None:
    print("DataFrame Ratings loaded successfully")


# convert raw ratings to user-item interaction matrix
df_user_movie = input_format.convert_dataset(dataframe=dataframe)


# load movies information
file_path = "movies.csv"  # Replace with your CSV file path
input_format = Input_Format(file_path)
df_movies = input_format.load_csv_dataset(use_endoding=True)
if df_movies is not None:
    print("DataFrame Movies loaded successfully")


# make recommendations
movie_selector = MovieSelector(df_user_movie = df_user_movie, df_movies = df_movies, num_top_movies=1)
target_userId = 8
Recommended_Movies = movie_selector.recommend_movies(target_userId = target_userId) 

print('target_userId:',  target_userId, '\n',
      'Recommended Movie:', Recommended_Movies)




################################################################
#%% Tests For Functional Requirements: test 1.2 (One Popular Movie)
# load raw ratings
file_path = "user_item_test1_2.csv"  # Replace with your CSV file path
input_format = Input_Format(file_path)
dataframe = input_format.load_csv_dataset()
if dataframe is not None:
    print("DataFrame Ratings loaded successfully")


# convert raw ratings to user-item interaction matrix
df_user_movie = input_format.convert_dataset(dataframe=dataframe)


# make recommendations
movie_selector = MovieSelector(df_user_movie = df_user_movie, df_movies = df_movies, num_top_movies=1)
target_userId = 8
Recommended_Movies = movie_selector.recommend_movies(target_userId = target_userId)

print('target_userId:',  target_userId, '\n',
      'Recommended Movie:', Recommended_Movies)




################################################################
#%% Tests For Functional Requirements: test 2 (Artificial Close Users)

# load raw ratings
file_path = "user_item_test2.csv"  # Replace with your CSV file path
input_format = Input_Format(file_path)
dataframe = input_format.load_csv_dataset()
if dataframe is not None:
    print("DataFrame Ratings loaded successfully")


# convert raw ratings to user-item interaction matrix
df_user_movie = input_format.convert_dataset(dataframe=dataframe)


# make recommendations
movie_selector = MovieSelector(df_user_movie = df_user_movie, df_movies = df_movies, num_top_movies=1)
target_userId = 8
Recommended_Movies = movie_selector.recommend_movies(target_userId = target_userId)

print('target_userId:',  target_userId, '\n',
      'Recommended Movie:', Recommended_Movies)



#################################################################
#%% Tests For Non Functional Requirements: test 1 (Accuracy)


# load raw ratings
file_path = "user_item.csv"  # Replace with your CSV file path
input_format = Input_Format(file_path)
dataframe = input_format.load_csv_dataset()
if dataframe is not None:
    print("DataFrame Ratings loaded successfully")


# convert raw ratings to user-item interaction matrix
df_user_movie = input_format.convert_dataset(dataframe=dataframe)


movie_selector = MovieSelector(df_user_movie = df_user_movie, df_movies = df_movies)
target_userId = 1

test_euclidean = [movie_selector.rating_prediction(target_userId, i) for i in range(1,51)]
original = df_user_movie.loc[target_userId][:50]

plt.plot(test_euclidean, label = "Euclidean",color="#243c63")
plt.plot(original, label = "Original", color ='#8cacdc')
plt.xlabel("Datapoint")
plt.ylabel("Rating")
plt.title('Original VS Euclidean')
plt.legend()
plt.gcf().set_size_inches(10, 3) 
plt.show()

print('RMSE (Movie Recommender):' , sklearn.metrics.mean_squared_error(original, test_euclidean), '\n'
      'MAE (Movie Recommender):' , mean_absolute_error(original,test_euclidean))
      
random_list = []
for i in range(50):
    random_list.append(random.choice([1, 2, 3, 4, 5]))


print('RMSE with random predictions:' , sklearn.metrics.mean_squared_error(original, random_list), '\n'
      'MAE with random predictions:' , mean_absolute_error(original,random_list))

