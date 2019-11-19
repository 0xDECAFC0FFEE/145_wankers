#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Content based, TF-IDF approach only. Does not filter out movies with few ratings
# TODO: filter out movies with few ratings. This helps the memory issue,
# but will require computation on the fly for movies that has few ratings.
# TODO: Build User Profile (so we can actually use training data). 
# TODO: Mix in the tags with relevances to create a better meta data in place of just genres
# TODO: try different similarity functions (such as Euclidean and 1-Jaccard,
# but I don't think they will improve the performance)

import random
import numpy as np
import pandas as pd
import math


# In[101]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm


# In[3]:


# Read the data into Pandas dataframe
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/train_ratings_binary.csv")
validation = pd.read_csv("dataset/val_ratings_binary.csv")


# In[4]:


# Go through movies.csv row by row and 1. replace (no genres listed) to None 2.Split the genres
# TODO: Mix in the tags here to create better meta data. 
for i, row in movies.iterrows(): 
    genre = row['genres']
    if genre == "(no genres listed)":
        genre = "None"
    else: 
        genre = genre.replace('|', ' ')
    movies.at[i, 'genres'] = genre


# In[5]:


# Call sklearn's TF_INF vectorizer to fit data, and normalize.
vectorizer = TfidfVectorizer()
movie_matrix= vectorizer.fit_transform(movies['genres'])
movie_matrix_df = pd.DataFrame(movie_matrix.toarray(), index=movies.index.tolist())


# In[28]:


# # Heavy memory usage ignore
# ratings_f1 = pd.merge(movies['movieId'], ratings,
#                       on="movieId", how="right")
# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(ratings_f1[['movieId','userId', 'rating']],
#                             reader)
# trainset, testset = train_test_split(data, test_size=.25)
# algorithm = SVD()
# # Train the algorithm on the trainset, and predict ratings for the testset
# algorithm.fit(trainset)
# accuracy.rmse(algorithm.test(testset))


# In[6]:


# Create the similarity matrix. Should be able to supply different similarity function
movie_similarity = cosine_similarity(movie_matrix, movie_matrix)


# In[8]:


# Create a Series of indexes. This is because movieIds are discontinous. 
# movieId -> line number in the csv AKA matrix index
movieIds = movies['movieId']
movie_indices = pd.Series(movies.index, index=movieIds)


# In[150]:


get_movie_indices = lambda x : movie_indices[x]
prev_user_id = -1
user_likes = 0
user_dislikes = 0

def is_similar_to_user_likes(userId, movieId):
    global prev_user_id
    global user_likes
    global user_dislikes
    
    if (userId != prev_user_id):
        prev_user_id = userId;
        user_likes = ratings.query('userId == %s & rating == 1'%(userId))['movieId']
        user_dislikes = ratings.query('userId == %s & rating == 0'%(userId))['movieId']
    
    like_score = [movie_similarity[movie_indices[movieId]][movie_indices[x]] for x in user_likes]
    dislike_score = [movie_similarity[movie_indices[movieId]][movie_indices[x]] for x in user_dislikes]

    like_score.sort(reverse=True)
    dislike_score.sort(reverse=True)

    
    return np.sum(like_score[:20]) > np.sum(dislike_score[:20])
    


# In[153]:


prediction =  []

for i, row in tqdm(validation.iterrows()):
    if (i > 100000): 
        break;
    prediction.append(is_similar_to_user_likes(row['userId'], row['movieId']))


# In[163]:


score = 0
for i in tqdm(range(100000)):
    if (prediction[i] == validation['rating'][i]):
        score+=1


# In[165]:


score/100000

