# current solutions
please record all your solutions in the readme to make the writeup easier
80% of our grade will be on the writeup
- team 1:
    - trivial solution:
        - assume all movies with the genre are the same
        - assume users only like movies based on genre
        - train a naive bayes classifier for each user based solely on the movies they've liked
        - 65% val acc
    - Content based solution
        - merge in tags into the movies with relevance > 40%
        - merge the tags with the genre to form a better description
        - fit the new movie matrix into a TF-INF matrix
        - movie_similarity = cosine_similarity(movie_matrix, movie_matrix)
        - predict user's like/dislike based on how similar the movie is to likes and dislikes
        - like_score = [similarities of movie to each one user likes]
        - dislike_score = [similarities of movie to each one user dislikes]
        - limit number of movies to check to: topN = min([len(like_score), len(dislike_score), 20])
        - predict true if sum(like_score[:topN]) > sum(dislike_score[:topN])
        - 70% roc auc 
- team 2: