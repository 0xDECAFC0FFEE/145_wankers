# implementation of probabilistic matrix factorisation

import pickle
import random
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm.notebook import tqdm
from itertools import chain
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
import csv
import tensorflow as tf

tag_names = Path("dataset")/"genome-tags.csv"                   # tag name lookup
movie_review_relevance = Path("dataset")/"genome-scores.csv"    # movieid/tagid/relevance
movie_genres = Path("dataset")/"movies.csv"                     # movieid/movie title/genres
reviews = Path("dataset")/"tags_shuffled_rehashed.csv"          # userid/movieid/tag
train_set = Path("dataset")/"train_ratings_binary.csv"          # train set - userid/movieid/ratings
val_set = Path("dataset")/"val_ratings_binary.csv"              # val set - userid/movieid/ratings
test_set = Path("dataset")/"test_ratings.csv"                   # test set - userid/movieids

NUM_MOVIES = 26744
NUM_USERS = 138493
NUM_TRAINING_SET = 11946576


# # internal movieids are used as movieids aren't contiguous
# userid_uid_lookup = lambda userid: userid-1

# movieid_mid_lookup = {}
# next_unassigned_mid = 0

# def add_movieids_to_lookuptable(filename):
#     global next_unassigned_mid

#     print(f"updating lookuptable with mids from {filename}")
#     with open(filename, newline="") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for rating in tqdm(reader):
#             movieid = int(float(rating["movieId"]))
#             if movieid not in movieid_mid_lookup:
#                 movieid_mid_lookup[movieid] = next_unassigned_mid
#                 next_unassigned_mid += 1

# add_movieids_to_lookuptable(train_set)
# add_movieids_to_lookuptable(val_set)
# add_movieids_to_lookuptable(test_set)
# add_movieids_to_lookuptable(movie_genres)

# with open("movieid_mid_lookup", "wb+") as lookup_file:
#     pickle.dump(movieid_mid_lookup, lookup_file)

userid_uid_lookup = lambda userid: userid-1

with open("movieid_mid_lookup", "rb") as lookup_file:
    movieid_mid_lookup = pickle.load(lookup_file)

# # cleaning up dataset
# def get_dataset(filename, include_ys=True):
#     print(f"retrieving dataset from {filename}")
#     with open(filename, newline="") as csvfile:
#         reader = csv.DictReader(csvfile)
#         user_Xs = []
#         movie_Xs = []
#         ys = []
#         for rating in tqdm(reader):
#             userid = int(float(rating["userId"]))
#             uid = userid_uid_lookup(userid)
#             user_Xs.append(uid)
            
#             movieid = int(float(rating["movieId"]))
#             mid = movieid_mid_lookup[movieid]
#             movie_Xs.append(mid)
            
#             if include_ys:
#                 score = 1 if (rating["rating"] == "1") else -1
#                 ys.append(score)
#     if include_ys:
#         return np.array(user_Xs).reshape(-1, 1), np.array(movie_Xs).reshape(-1, 1), np.array(ys).reshape(-1, 1)
#     else:
#         return np.array(user_Xs).reshape(-1, 1), np.array(movie_Xs).reshape(-1, 1)

# def genre_parser(genre):
#     if genre == "(no genres listed)":
#         return ["none/other"]
#     return genre.split("|")

# ALL_GENRES = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Crime', 'Horror', 'Documentary', 'Adventure', 'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Children', 'Musical', 'Animation', 'Western', 'Film-Noir', 'none/other', 'IMAX']
# with open(movie_genres, newline="") as csvfile:
#     reader = csv.DictReader(csvfile)
#     movie_genres_one_hot = {movieid_mid_lookup[int(float(movie["movieId"]))]: np.array([genre in movie["genres"] for genre in ALL_GENRES]) for movie in reader}        

# user_Xs, movie_Xs, ys = get_dataset(train_set)
# user_val_Xs, movie_val_Xs, val_ys = get_dataset(val_set)

# with open("mid_genres_one_hot", "wb+") as genre_file:
#     pickle.dump(movie_genres_one_hot, genre_file)
# with open("training_set", "wb+") as training_set_file:
#     pickle.dump((user_Xs, movie_Xs, ys), training_set_file)
# with open("val_set", "wb+") as val_set_file:
#     pickle.dump((user_val_Xs, movie_val_Xs, val_ys), val_set_file)

with open("mid_genres_one_hot", "rb") as genre_file:
    movie_genres_one_hot = pickle.load(genre_file)
with open("training_set", "rb") as training_set_file:
    user_Xs, movie_Xs, ys = pickle.load(training_set_file)
with open("val_set", "rb") as val_set_file:
    user_val_Xs, movie_val_Xs, val_ys = pickle.load(val_set_file)

def batchify(*args, batch_size=1000, shuffle=True):
    if batch_size == -1:
        return [args]
    
    num_elems = len(args[0])

    if shuffle:
        shuffle_indices = np.arange(num_elems, dtype=np.int64)
        np.random.shuffle(shuffle_indices)
        for i in range(0, num_elems, batch_size):
            array_indices = shuffle_indices[i: i+batch_size]
            try:
                yield [arg[array_indices] for arg in args]
            except:
                raise Exception("args to batchify must be numpy arrays if shuffle True")
    else:
        for i in range(0, num_elems, batch_size):
            yield [arg[i: i+batch_size] for arg in args]

## all movies in test data accounted for in genre information dataset

# no_genre_count = 0
# total = 0

# with open(test_set, newline="") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for rating in tqdm(reader):
#         if movieid_mid_lookup[int(float(rating["movieId"]))] not in movie_genres_one_hot:
#             no_genre_count += 1
#         total += 1

# print(f"{no_genre_count}/{total} entries in the test data doesn't have genre info ({no_genre_count/total}%)")

# # no memory - implicitly calculating user movie matrix from now on

# movie_embeddings = tf.Variable(tf.random_normal([5, NUM_MOVIES], stddev=0.03, dtype=tf.float32))
# user_embeddings = tf.Variable(tf.random_normal([NUM_USERS, 5], stddev=0.03, dtype=tf.float32))
# movie_bias = tf.Variable(tf.random_normal([1, NUM_MOVIES], stddev=0.03, dtype=tf.float32))
# user_bias = tf.Variable(tf.random_normal([NUM_USERS, 1], stddev=0.03, dtype=tf.float32))

# user_movie_score = tf.tensordot(user_embeddings, movie_embeddings, axes = 1)+.14*tf.tile(movie_bias, [NUM_USERS, 1]) +.87*tf.tile(user_bias, [1, NUM_MOVIES])

embedding_dim = 40
assert embedding_dim > 20

movie_genre_embeddings = tf.placeholder(dtype=tf.float32, shape=[None, 20])
movie_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer()([NUM_MOVIES, embedding_dim]))
user_embeddings = tf.Variable(tf.contrib.layers.xavier_initializer()([NUM_USERS, embedding_dim]))
movie_bias = tf.Variable(tf.random_normal([NUM_MOVIES], stddev=0.03, dtype=tf.float32))
user_bias = tf.Variable(tf.random_normal([NUM_USERS], stddev=0.03, dtype=tf.float32))

user_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1])
movie_slice_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1])
user_bias_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1])
movie_bias_idxs = tf.placeholder(dtype=tf.int64, shape=[None, 1])

user_embedding_columns = tf.reshape(tf.gather_nd(user_embeddings, user_slice_idxs), [-1, embedding_dim])
movie_embedding_rows = tf.reshape(tf.gather_nd(movie_embeddings, movie_slice_idxs), [-1, embedding_dim])
print("movie_embedding_rows shape", movie_embedding_rows.shape)

user_slice_bias = tf.reshape(tf.gather_nd(user_bias, user_slice_idxs), [-1, 1])
movie_slice_bias = tf.reshape(tf.gather_nd(movie_bias, movie_slice_idxs), [-1, 1])
# print("user_slice_bias shape", user_slice_bias.shape)

# print((user_embedding_columns * tf.concat((movie_embedding_rows, movie_genre_embeddings), axis=1)).shape)
# print(movie_slice_bias.shape)
# print(user_slice_bias.shape)

input_layer = tf.concat((
    movie_embedding_rows * user_embedding_columns,
    movie_embedding_rows,
    user_embedding_columns,
    user_slice_bias,
    movie_slice_bias), axis=1)
print(movie_embedding_rows.shape, user_embedding_columns.shape, user_slice_bias.shape)
print("input layer shape", input_layer.shape)

W1 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[embedding_dim*3+2, 40], dtype=tf.float32))
b1 = tf.Variable(initial_value=np.zeros(shape=[40], dtype=np.float32))
l1 = tf.nn.relu(tf.matmul(input_layer, W1) + b1)

W2 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[40, 20], dtype=tf.float32))
b2 = tf.Variable(initial_value=np.zeros(shape=[20], dtype=np.float32))
l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

W3 = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[20, 1], dtype=tf.float32))
b3 = tf.Variable(initial_value=np.zeros(shape=[1], dtype=np.float32))
pred_y =tf.nn.sigmoid(tf.matmul(l2, W3) + b3)*2-1

# embedding_pred_vectors = tf.reshape(tf.reduce_sum(user_embedding_columns * tf.concat((movie_embedding_rows, movie_genre_embeddings), axis=1), axis=1), (-1, 1))
# pred_y = embedding_pred_vectors + .14*movie_slice_bias + .87*user_slice_bias
# print(embedding_pred_vectors.shape)
# print(pred_y.shape)

y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])




def compute_slices(user_Xs, movie_Xs, embedding_dim):
    user_slice_idxs = [[user_X] for user_X in user_Xs]
    movie_slice_idxs = [[movie_X] for movie_X in movie_Xs]

    return [np.array(user_slice_idxs).reshape([-1, 1]), np.array(movie_slice_idxs).reshape([-1, 1])]

learning_rate = .05
epochs = 15

loss = tf.reduce_mean(tf.squared_difference(pred_y, y_true))
# + sum([.00001 * tf.nn.l2_loss(weight) for weight in all_weights])
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), leave=False):
#         for b_m_Xs, b_u_Xs, b_ys in batchify(movie_Xs, user_Xs, ys, batch_size=-1):
#             print("batch vals", b_m_Xs, b_u_Xs, b_ys)
        
        print("computing slice indices")
        slice_indices = compute_slices(user_Xs, movie_Xs, embedding_dim)
        user_slice, movie_slice = slice_indices

#             print("user_slice indexes shape", np.array(user_slice).shape)
#             print("movie bias indexes shape", np.array(m_bias_slice).shape)

        print("computing genres")
        genres = np.array([movie_genres_one_hot[x[0]] for x in movie_Xs])
        feed_dict = {user_slice_idxs: user_slice, 
                     movie_slice_idxs: movie_slice, 
                     movie_genre_embeddings: genres,
                     y_true: ys}
        print("training")
        _, lossval, pred_y_val = sess.run((train_step, loss, pred_y), feed_dict=feed_dict)
        print("train loss", lossval, "pred_ys", pred_y_val[:5].flatten(), "true_ys", ys[:5].flatten())

        print("computing val acc...")
        val_slice_indices = compute_slices(user_val_Xs, movie_val_Xs, embedding_dim)
        user_val_slice, movie_val_slice = val_slice_indices
        genres = np.array([movie_genres_one_hot[x[0]] for x in movie_val_Xs])
        feed_dict = {user_slice_idxs: user_val_slice, 
                         movie_slice_idxs: movie_val_slice,
                         movie_genre_embeddings: genres,
                         y_true: val_ys}
        val_y_pred, val_loss_val = sess.run((pred_y, loss), feed_dict=feed_dict)
        print("val loss", val_loss_val)
        print("val acc", sum([((1 if pred > .5 else -1) == true) for pred, true in zip(val_y_pred, val_ys)])/len(val_ys))
        
        



print(movie_genres_one_hot[27278])

val_slice_indices = compute_slices(user_val_Xs, movie_val_Xs, embedding_dim)
user_val_slice, movie_val_slice = val_slice_indices

print(user_val_slice.shape, movie_val_slice.shape, np.array(val_ys).shape)

slice_indices = compute_slices(user_Xs, movie_Xs, embedding_dim)
user_slice, movie_slice = slice_indices

print(user_slice.shape, movie_slice.shape, np.array(ys).shape)

print(len(user_Xs), len(ys))
print(len(user_val_Xs), len(val_ys))


