import unittest
import pandas as pd
import sys

sys.path.append('C:\\Users\\seyed\\Desktop\\RecommSys\\src') # Replace the path
from modules import MovieSelector, Input_Format 

class TestMovieSelector(unittest.TestCase):
    def setUp(self):

        # Set up sample data for testing
        file_path = "user_item.csv"  # Replace with your CSV file path
        input_format = Input_Format(file_path)
        dataframe = input_format.load_csv_dataset()
       
        # convert raw ratings to user-item interaction matrix
        self.df_user_movie = input_format.convert_dataset(dataframe=dataframe)
        
        file_path = "movies.csv"  # Replace with your CSV file path
        input_format = Input_Format(file_path)
        self.df_movies = input_format.load_csv_dataset(use_endoding=True)


        # Initialize MovieSelector object
        self.movie_selector = MovieSelector(self.df_user_movie, self.df_movies)

    def test_rating_prediction(self):
        # Test rating prediction for a specific user and movie
        predicted_rating = self.movie_selector.rating_prediction(1, 101)
        self.assertIsInstance(predicted_rating, float)

    def test_recommend_movies(self):
        # Test movie recommendation for a specific user
        recommended_movies = self.movie_selector.recommend_movies(10)
        self.assertIsInstance(recommended_movies, list)
        self.assertEqual(len(recommended_movies), 10)  # Assuming 1 top recommended movies



if __name__ == '__main__':
    unittest.main()