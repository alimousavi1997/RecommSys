import unittest
import sys
import pandas as pd

sys.path.append('C:\\Users\\seyed\\Desktop\\RecommSys\\src') # Replace the path
from modules import MovieSelector  


class TestMovieSelector(unittest.TestCase):

    def setUp(self):
        # Create sample dataframes for testing
        self.df_user_movie = pd.DataFrame({
            'userId': [1, 2, 3],
            'movie1': [5, 0, 4],
            'movie2': [0, 3, 0],
            'movie3': [2, 0, 0]
        }).set_index('userId')

        self.df_movies = pd.DataFrame({
            'movieId': [1, 2, 3],
            'movieTitle': ['Movie A', 'Movie B', 'Movie C']
        })

        self.movie_selector = MovieSelector(self.df_user_movie, self.df_movies)

    def test_rating_prediction(self):
        # Test if rating prediction returns a float
        rating = self.movie_selector.rating_prediction(1, 'movie1')
        self.assertIsInstance(rating, float)

        # Add more specific test cases based on your requirements

    def test_recommend_movies(self):
        # Test if recommended movies are in the list of movie titles
        recommended_movies = self.movie_selector.recommend_movies(1)
        expected_movies = ['Movie A', 'Movie B']
        self.assertCountEqual(recommended_movies, expected_movies)

        # Add more specific test cases based on your requirements


if __name__ == '__main__':
    unittest.main()