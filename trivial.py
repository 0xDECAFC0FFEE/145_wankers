import random
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import csv
from utils import *

tag_names = Path("dataset")/"genome-tags.csv"                   # tag name lookup
movie_review_relevance = Path("dataset")/"genome-scores.csv"    # movieid/tagid/relevance
movie_genres = Path("dataset")/"movies.csv"                     # movieid/movie title/genres
reviews = Path("dataset")/"tags_shuffled_rehashed.csv"          # userid/movieid/tag
Xs = Path("dataset")/"train_ratings_binary.csv"                 # train set - userid/movieid/ratings
val_set = Path("dataset")/"val_ratings_binary.csv"              # val set - userid/movieid/ratings
test_set = Path("dataset")/"test_ratings.csv"                   # test set - userid/movieids

def genre_parser(genre):
    if genre == "(no genres listed)":
        return ["none/other"]
    return genre.split("|")

# with open(movie_genres, newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     genres = [genre_parser(movie["genres"]) for movie in reader]
#     counts = Counter(chain(*genres))
#     print(counts)
print("THERE ARE 20 GENRES IN TOTAL: {'Drama': 13344, 'Comedy': 8374, 'Thriller': 4178, 'Romance': 4127, 'Action': 3520, 'Crime': 2939, 'Horror': 2611, 'Documentary': 2471, 'Adventure': 2329, 'Sci-Fi': 1743, 'Mystery': 1514, 'Fantasy': 1412, 'War': 1194, 'Children': 1139, 'Musical': 1036, 'Animation': 1027, 'Western': 676, 'Film-Noir': 330, 'none/other': 246, 'IMAX': 196}")

# with open(reviews, newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     seen_before = {(i["userId"], i["movieId"]) for i in reader}
#     with open(test_set, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         overlap = [(y["userId"], y["movieId"]) in seen_before for y in reader]
#         print(Counter(overlap))
print("ALL TESTING USER-MOVIE PAIRS HAVE NO REVIEWS")


# with open(val_set, newline="") as csvfile:
#     reader = csv.DictReader(csvfile)
#     total = 0
#     pos = 0
#     for rating in tqdm(reader):
#         total += 1
#         pos += rating["rating"] == "1"
#     print("trivial solution % true = ", pos/total)
print("trivial solution % true =  0.48282997052437016")

genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Crime', 'Horror', 'Documentary', 'Adventure', 'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Children', 'Musical', 'Animation', 'Western', 'Film-Noir', 'none/other', 'IMAX']
with open(movie_genres, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    movie_genres_one_hot = {int(float(movie["movieId"])): np.array([genre in movie["genres"] for genre in genres]) for movie in reader}

pos_user_movies = defaultdict(lambda: [])
neg_user_movies = defaultdict(lambda: [])
with open(Xs, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for rating in tqdm(reader):
        userid = int(float(rating["userId"]))
        movieid = int(float(rating["movieId"]))
        if rating["rating"] == "1":
            pos_user_movies[userid].append(movieid)
        else:
            neg_user_movies[userid].append(movieid)


pred_ys = []
y_true = []
acc_num = 0
acc_denom = 0

fitted_users = {}

with open(val_set, newline="") as csvfile:
    reader = csv.DictReader(csvfile)

    movieid_mid_lookup = get_movieid_mid_lookup()
    mid_to_proj_tag = build_pca_model(1, recompute=True)

    for i, rating in tqdm(list(enumerate(reader))):
        userid = int(float(rating["userId"]))
        movieid = int(float(rating["movieId"]))

        if userid in fitted_users:
            clf = fitted_users[userid]
        else:

            pos_Xs = [np.concatenate((
                movie_genres_one_hot[train_movieid], 
                mid_to_proj_tag[movieid_mid_lookup[train_movieid]]), axis=0) for train_movieid in pos_user_movies[userid]]

            neg_Xs = [np.concatenate((
                movie_genres_one_hot[train_movieid], 
                mid_to_proj_tag[movieid_mid_lookup[train_movieid]]), axis=0) for train_movieid in neg_user_movies[userid]]

            Xs = pos_Xs + neg_Xs
            ys = ([1]*len(pos_Xs)) + ([0]*len(neg_Xs))

            clf = MultinomialNB().fit(np.array(Xs), np.array(ys))
            fitted_users[userid] = clf

        val_X = np.concatenate((movie_genres_one_hot[movieid], mid_to_proj_tag[movieid_mid_lookup[movieid]]), axis=0)

        probs = clf.predict_proba(np.array([val_X]))
        pred_ys.append(probs[0][1]/(probs[0][1] + probs[0][0]))
        y_true.append(rating["rating"] == "1")

        if i % 2000 == 0:
            print(y_true[-10:])
            print(pred_ys[-10:])
            try:
                print(roc_auc_score(y_true, pred_ys))
            except:
                pass
