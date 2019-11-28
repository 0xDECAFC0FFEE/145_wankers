#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library imports
import numpy as np
import pandas as pd
import math
import csv
import timeit
import numexpr


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc


# In[3]:


# Read the data into Pandas dataframe
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/train_ratings_binary.csv")
validation = pd.read_csv("dataset/val_ratings_binary.csv")
genome_scores = pd.read_csv("dataset/genome-scores.csv")
genome_tags = pd.read_csv("dataset/genome-tags.csv")
tags = pd.read_csv("dataset/tags_shuffled_rehashed.csv")


# In[4]:


# Find the tags that are highly relevant to the movies.
genome_scores_f = genome_scores[genome_scores['relevance'] > 0.4]

# Join the filtered genome scores with genome tags to form a new dataframe
genome_scores_m = pd.merge(genome_scores_f, genome_tags, on='tagId', how='left')
genome_scores_m = pd.DataFrame(genome_scores_m.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))

# Join movies with the new tag table, remove all empty entries 
movies_merge = pd.merge(movies, genome_scores_m, on='movieId', how='left')
movies_merge.fillna("", inplace=True)


# In[110]:


# Go through movies_merge row by row and 
# 1. Replace (no genres listed) to None 
# 2. Split the genres 
# 3. Add the tags to create a new description
# Hypothetically,the title and the year of the movie could also be added to the description
# Doing a lambda mapping might be faster, but this loop is  quick already
for i, row in movies_merge.iterrows(): 
    genre = row['genres']
    if genre == "(no genres listed)":
        # Maybe Null is a better word, since it is not a common English word 
        genre = "None"
    else: 
        genre = genre.replace('|', ' ')
    movies_merge.at[i, 'description'] = genre + " "+ row['tag']


# In[111]:


# Drop the now useless genres and tag columns
movies_merge.drop(['genres','tag'], axis=1, inplace=True)


# In[134]:


# This is what movies_merge looks like
movies_merge


# In[112]:


# Call sklearn's TF_INF vectorizer to fit data, and normalize.
vectorizer = TfidfVectorizer()
movie_matrix= vectorizer.fit_transform(movies_merge['description'])

# Compute the similarity between each movie to every other moives, by finding out the cosine similarity
movie_similarity = cosine_similarity(movie_matrix, movie_matrix)


# In[114]:


# Create a Series of indexes. This is because movieIds are discontinous. 
# movieId -> line number in the csv AKA matrix index
movieIds = movies['movieId']
movie_indices = pd.Series(movies.index, index=movieIds)


# In[319]:


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
    


# In[320]:


# Write the prediction to pred.csv. 
# CSV headers are: userId, movieId, ratings
# Runtime: 10 hours
with open("pred.csv", "a", newline='') as file:
    writer = csv.writer(file, delimiter = ",",lineterminator='\r')
    writer.writerow(["userId", "movieId", "rating"])
    for i, row in tqdm(validation.iterrows()):
        writer.writerow([row['userId'], row['movieId'], is_similar_to_user_likes(row['userId'], row['movieId'])])
        


# In[312]:


ratings.userId.values==2


# In[322]:


prediction = pd.read_csv("pred.csv")
false_positive_rate, true_positive_rate, thresholds = roc_curve(validation['rating'], prediction['rating'])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[306]:


prediction


# In[ ]:




