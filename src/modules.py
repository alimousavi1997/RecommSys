import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics  
from sklearn.metrics import mean_absolute_error



class Input_Format:

    def __init__(self, file_path):
        self.file_path = file_path
    

    def load_csv_dataset(self , use_endoding = False):
        if(use_endoding):
            try:
                dataframe = pd.read_csv(self.file_path , encoding='unicode_escape')
                return dataframe
            except FileNotFoundError:
                print("File not found. Please provide a valid file path.")
                return None
        try:
            dataframe = pd.read_csv(self.file_path)
            return dataframe
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
            return None
        except Exception as InvalidFileFormat:
            print(f"An error occurred: {InvalidFileFormat}")
            return None
        
        
    def convert_dataset(self, dataframe):
        dataframe = dataframe.pivot(index = 'userId', columns = 'movieId', values = 'rating')
        dataframe = dataframe.fillna(0)
        return dataframe
    



class KNN:

    def __init__(self,k=3):
        self.k = k


    def fit(self,X,y):
        self.X_train = X
        self.y_train = y


    def nearest_neighbor_indices(self ,x):
        
        # compute distnaces
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        # get k nearest smaples, labels
        k_indices = np.argsort(distances)[:self.k]

        return k_indices


    def _predict(self , x):
       
        # get k nearest smaples, labels
        k_indices = self.nearest_neighbor_indices(x)
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def euclidean_distance(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))





def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))





class MovieSelector:
    
    def __init__(self, df_user_movie, df_movies, num_top_movies=10):
        self.df_user_movie = df_user_movie
        self.df_movies = df_movies
        self.num_top_movies = num_top_movies

    def rating_prediction(self, target_userId, target_movieId):
        # Exclude the target user from the user-item interaction matrix
        df_user_movie_without_target_user = self.df_user_movie.drop(target_userId)

        # Filter users who have watched the target movie
        df_users_watched_target_movieId = df_user_movie_without_target_user[
            df_user_movie_without_target_user[target_movieId] != 0]
        userIds_watched_target_movieId = list(df_users_watched_target_movieId.index)


        # Exclude the target movie from the user-item interaction matrix as well
        df_user_movie_without_target_movieId = df_user_movie_without_target_user.drop(target_movieId, axis=1)


        # Final dataframe suitable for KNN analysis
        df_to_knn_analysis = df_user_movie_without_target_movieId.loc[
            df_user_movie_without_target_movieId.index.isin(userIds_watched_target_movieId)]


        # Prepare training set
        X_train = df_to_knn_analysis.to_numpy()
        y_train = df_users_watched_target_movieId[target_movieId].to_numpy()


        # Create classifier
        clf = KNN(k=12)
        clf.fit(X_train, y_train)


        # Vector profile of the target_user
        x = self.df_user_movie.loc[target_userId].drop(target_movieId).to_numpy()
        predicted_rating = clf._predict(x)

        return predicted_rating



    def recommend_movies(self, target_userId):
        predicted_ratings = []

        # Predict ratings for all the movies
        for movieId in self.df_user_movie.columns:
            predicted_ratings.append([self.rating_prediction(target_userId, movieId), movieId])

        # Sort ratings and output best recommendations
        predicted_ratings = sorted(predicted_ratings)
        recommended_movieIds = [predicted_ratings[i][1] for i in
                                range(len(predicted_ratings) - self.num_top_movies, len(predicted_ratings))]
        recommended_movieTitles = list(
            self.df_movies[self.df_movies['movieId'].isin(recommended_movieIds)]['movieTitle'])

        return recommended_movieTitles







    
