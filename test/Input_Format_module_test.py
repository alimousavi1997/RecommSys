import unittest
import pandas as pd
import sys

sys.path.append('C:\\Users\\seyed\\Desktop\\RecommSys\\src') # Replace the path
from modules import Input_Format  



class TestInputFormat(unittest.TestCase):
    def setUp(self):
        # Set up a sample CSV file for testing
        self.file_path = 'user_item_test1.csv'
        self.sample_data = {
            'userId': [1, 1, 2, 2],
            'movieId': [101, 102, 101, 103],
            'rating': [5, 4, 3, 2]
        }
        self.df = pd.DataFrame(self.sample_data)
        self.df.to_csv(self.file_path, index=False)
    
    def tearDown(self):
        # Clean up the sample CSV file after testing
        import os
        os.remove(self.file_path)

    def test_load_csv_dataset(self):
        input_format = Input_Format(self.file_path)
        dataframe = input_format.load_csv_dataset()
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(len(dataframe), 4)  # Assuming 4 rows in the sample data

    def test_load_csv_dataset_with_encoding(self):
        input_format = Input_Format(self.file_path)
        dataframe = input_format.load_csv_dataset()
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(len(dataframe), 4)  # Assuming 4 rows in the sample data

    def test_convert_dataset(self):
        input_format = Input_Format(self.file_path)
        dataframe = input_format.load_csv_dataset()
        converted_dataframe = input_format.convert_dataset(dataframe)
        self.assertIsInstance(converted_dataframe, pd.DataFrame)
        self.assertEqual(converted_dataframe.shape, (2, 3))  # Assuming 2 users and 3 movies

if __name__ == '__main__':
    unittest.main()