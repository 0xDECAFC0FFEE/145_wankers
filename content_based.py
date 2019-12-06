#!/usr/bin/env python
# coding: utf-8



# Library imports
import numpy as np
import pandas as pd
import math
import csv
import timeit
import numexpr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

# Read the data into Pandas dataframe
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/train_ratings_binary.csv")
validation = pd.read_csv("dataset/val_ratings_binary.csv")
genome_scores = pd.read_csv("dataset/genome-scores.csv")
genome_tags = pd.read_csv("dataset/genome-tags.csv")
tags = pd.read_csv("dataset/tags_shuffled_rehashed.csv")
test = pd.read_csv("dataset/test_ratings.csv")

# Find the tags that are highly relevant to the movies.
genome_scores_f = genome_scores[genome_scores['relevance'] > 0.45]

# Join the filtered genome scores with genome tags to form a new dataframe
genome_scores_m = pd.merge(genome_scores_f, genome_tags, on='tagId', how='left')
genome_scores_m = genome_scores_m[genome_scores_m['tag'].apply(lambda x: len(x.split()) == 1)]
genome_scores_m = pd.DataFrame(genome_scores_m.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))

# Join movies with the new tag table, remove all empty entries 
movies_merge = pd.merge(movies, genome_scores_m, on='movieId', how='left')
movies_merge.fillna("", inplace=True)

# Go through movies_merge row by row and 
# 1. Replace (no genres listed) to None 
# 2. Split the genres 
# 3. Add the tags to create a new description
for i, row in movies_merge.iterrows(): 
    genre = row['genres']
    if genre == "(no genres listed)":
        # Maybe Null is a better word, since it is not a common English word 
        genre = "None"
    else: 
        genre = genre.replace('|', ' ')
    movies_merge.at[i, 'description'] = genre + " "+ row['tag']

# Go through movies_merge row by row and 
# 1. Merge title and year into description
for i, row in movies_merge.iterrows(): 
    try: 
        mTitle = row['title'][:row['title'].index('(')]
        year = row['title'][row['title'].index('(')+1:-1]
    except: 
        mTitle = row['title']
        year = ""
    movies_merge.at[i, 'description'] = row['description'] + " "+ mTitle + year

# Drop the now useless genres and tag columns
movies_merge.drop(['genres','tag'], axis=1, inplace=True)

# Call sklearn's TF_INF vectorizer to fit data, and normalize.
vectorizer = TfidfVectorizer(analyzer='word',strip_accents='unicode')
movie_matrix= vectorizer.fit_transform(movies_merge['description'])

# Compute the similarity between each movie to every other moives, by finding out the cosine similarity
movie_similarity = cosine_similarity(movie_matrix, movie_matrix)

# Create a Series of indexes. This is because movieIds are discontinous. 
# movieId -> line number in the csv AKA matrix index
movieIds = movies['movieId']
movie_indices = pd.Series(movies.index, index=movieIds)

# Cache the results to speed up the calculation. Also helps with memory issue. 
prev_user_id = -1
user_likes = 0
user_dislikes = 0

def is_similar_to_user_likes(userId, movieId):
    global prev_user_id
    global user_likes
    global user_dislikes
    
    # If userId has changed, then query the dataframe again and update cache. 
    if (userId != prev_user_id):
        prev_user_id = userId;
        user_cond = ratings.userId.values
        rating_cond = ratings.rating.values
        x = user_cond
        y = rating_cond
        user_likes = ratings[numexpr.evaluate('(x == ' + str(userId) + ') & (y == 1)')]['movieId']
        user_dislikes = ratings[numexpr.evaluate('(x == ' + str(userId) + ') & (y == 0)')]['movieId']
    
    like_score = [movie_similarity[movie_indices[movieId]][movie_indices[x]] for x in user_likes]
    dislike_score = [movie_similarity[movie_indices[movieId]][movie_indices[x]] for x in user_dislikes]

    # Use the most similar movies 
    like_score.sort(reverse=True)
    dislike_score.sort(reverse=True)
    
    # Peg the number of scores to be considered. 
    topN = min([len(like_score), len(dislike_score), 20])
    
    # Predict True AKA user likes the movies, if the sum of like_score is more than that of dislike_score
    return np.sum(like_score[:topN]) > np.sum(dislike_score[:topN])
    
# For validation
# Write the prediction to pred.csv. 
# CSV headers are: userId, movieId, ratings
# Runtime: 8 hours
with open("pred.csv", "a", newline='') as file:
    writer = csv.writer(file, delimiter = ",",lineterminator='\r')
    writer.writerow(["userId","movieId","ratings"])
    for i, row in tqdm(test_ratings.iterrows()):
        writer.writerow([row['userId'], row['movieId'], is_similar_to_user_likes(row['userId'], row['movieId'])])

# For test
# Write the prediction to content_pred.csv. 
# CSV headers are: Id, ratings
# Runtime: 8 hours
with open("content_pred.csv", "a", newline='') as file:
    writer = csv.writer(file, delimiter = ",",lineterminator='\r')
    writer.writerow(["Id", "rating"])
    for i, row in tqdm(test_ratings.iterrows()):
        writer.writerow([i, is_similar_to_user_likes(row['userId'], row['movieId'])])
        
# Plot RoC curve for validation
prediction = pd.read_csv("pred.csv")
false_positive_rate, true_positive_rate, thresholds = roc_curve(validation['rating'], prediction['rating'])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()