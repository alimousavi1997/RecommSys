import sys
import pandas as pd
from modules import Input_Format, MovieSelector


if __name__ == "__main__":

    # load raw ratings
    file_path = "user_item.csv"  # Replace with your CSV file path
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
    movie_selector = MovieSelector(df_user_movie = df_user_movie, df_movies = df_movies)
    target_userId = int(input('please enter target userId:'))
    Recommended_Movies = movie_selector.recommend_movies(target_userId)
    print(Recommended_Movies)
   
    

