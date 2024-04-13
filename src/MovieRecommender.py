import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics  
from sklearn.metrics import mean_absolute_error


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))



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






def rating_prediction(target_userId, target_movieId, df_user_movie):
    
    # drop target user in the user-item interaction matrix
    df_user_movie_without_target_user = df_user_movie.drop(target_userId)

    # filter users who have watched the target movie
    df_users_watched_target_movieId = df_user_movie_without_target_user[df_user_movie_without_target_user[target_movieId] != 0]
    userIds_watched_target_movieId = list(df_users_watched_target_movieId.index)

    # Also drop target movie in the user-item interaction matrix 
    df_user_movie_without_target_movieId = df_user_movie_without_target_user.drop(target_movieId , axis =1)
    

    # final df suitable for knn analysis
    df_to_knn_analysis = df_user_movie_without_target_movieId.loc[
            df_user_movie_without_target_movieId.index.isin(userIds_watched_target_movieId)]
    
    # prepare training set
    X_train = df_to_knn_analysis.to_numpy()
    y_train = df_users_watched_target_movieId[target_movieId].to_numpy()
    

    # create classifier
    clf = KNN(k = 12)
    clf.fit(X_train, y_train)
    
    x = df_user_movie.loc[target_userId].drop(target_movieId).to_numpy() # x is the vector profile of the target_user
    predicted_rating = clf._predict(x)

    return predicted_rating




def top_movies(target_userId , df_user_movie, num_top_movies = 10):

    predicted_ratings = []
    
    # predict ratings for all the movies 
    for movieId in df_user_movie.columns:
        predicted_ratings.append([rating_prediction(target_userId, movieId, df_user_movie),movieId])

    # sort ratings and output best recommendations
    predicted_ratings = sorted(predicted_ratings)
    recommended_movieIds = [predicted_ratings[i][1] for i in range(len(predicted_ratings)-num_top_movies,len(predicted_ratings))]
    recommended_movieTitles = list(df_movies[df_movies['movieId'].isin(recommended_movieIds)]['movieTitle'])

    return recommended_movieTitles




#load raw ratings
df = pd.read_csv('user_item.csv')

# convert the raw ratings to the dataframe
df_user_movie = df.pivot(index = 'userId', columns = 'movieId', values = 'rating')

# fill missing ratings with 0
df_user_movie = df_user_movie.fillna(0)

# load movies information
df_movies = pd.read_csv('C:\\Users\\seyed\\Desktop\\RecommSys\\movies.csv', encoding='unicode_escape')

# make recommendations
target_userId = int(input('please enter target userId:'))
print(top_movies(target_userId,df_user_movie))
