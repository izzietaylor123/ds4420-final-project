import pandas as pd
import numpy as np
from pathlib import Path


DATA = Path(__file__).parent / "data" # path to data folder (contains the different CSVs that need to be stitched together)

def load_data(sample_users, min_movie_ratings, min_tag_count, max_tags, seed, keep_users=()):
    """Read the CSVs and build index maps plus per-movie feature tables."""
    rng = np.random.default_rng(seed) # for reproducibility

    print("Reading rating.csv ...")

    # pull out the ratings 
    ratings = pd.read_csv(
        DATA / "rating.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    if sample_users and sample_users < ratings["userId"].nunique(): # sample users if requested
        users = ratings["userId"].unique()
        chosen = set(rng.choice(users, sample_users, replace=False).tolist()) | set(keep_users)
        ratings = ratings[ratings["userId"].isin(chosen)]

    counts = ratings["movieId"].value_counts() # count of ratings per movie (used if we are implementing a min threshold for number of ratings)
    ratings = ratings[ratings["movieId"].isin(counts[counts >= min_movie_ratings].index)]
    ratings = ratings.reset_index(drop=True)

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    ratings["u"] = ratings["userId"].map(user2idx).astype(np.int64)
    ratings["m"] = ratings["movieId"].map(movie2idx).astype(np.int64)
    n_movies = len(movie_ids)

    # pull out the movies and their genres to do multi-hot 
    movies = pd.read_csv(DATA / "movie.csv").set_index("movieId").loc[movie_ids] # connect the movies we pulled from ratings.csv to their genres in movie.csv
    genre_lists = movies["genres"].str.split("|")
    genres = sorted({g for gl in genre_lists for g in gl if g != "(no genres listed)"})
    g2i = {g: i for i, g in enumerate(genres)} # map each genre to an index for the multi-hot encoding
    genre_mat = np.zeros((n_movies, len(genres)), dtype=np.float32)
    for row, gl in enumerate(genre_lists):
        for g in gl:
            if g in g2i:
                genre_mat[row, g2i[g]] = 1.0

    # pull out the tags w/ padded index matrix (0 = padding) 
    tags = pd.read_csv(DATA / "tag.csv", usecols=["movieId", "tag"]).dropna() # connect to the tag.csv file to get the tags for each movie
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["movieId"].isin(movie2idx)] # filter to only include tags for movies that are in our ratings dataset
    tag_counts = tags["tag"].value_counts()
    vocab = tag_counts[tag_counts >= min_tag_count].index # filter to only include tags that meet the minimum threshold for number of ratings
    t2i = {t: i + 1 for i, t in enumerate(vocab)}  # 0 reserved for padding
    tags = tags[tags["tag"].isin(t2i)] 
    # per-movie most frequent tags first, capped at max_tags
    per_movie = tags.groupby(["movieId", "tag"]).size().rename("n").reset_index()
    per_movie = per_movie.sort_values(["movieId", "n"], ascending=[True, False])
    per_movie["pos"] = per_movie.groupby("movieId").cumcount() # get the position of each tag for each movie (0 = most frequent, 1 = second most frequent, etc.)
    per_movie = per_movie[per_movie["pos"] < max_tags]
    tag_mat = np.zeros((n_movies, max_tags), dtype=np.int64)
    tag_mat[per_movie["movieId"].map(movie2idx).to_numpy(), per_movie["pos"].to_numpy()] = (
        per_movie["tag"].map(t2i).to_numpy()
    )

    print(
        f"{len(ratings):,} ratings | {len(user_ids):,} users | {n_movies:,} movies | "
        f"{len(genres)} genres | {len(vocab):,} tags "
        f"({(tag_mat[:, 0] > 0).mean():.1%} of movies have tags)"
    )
    return dict(
        ratings=ratings, user_ids=user_ids, movie_ids=movie_ids, user2idx=user2idx, # return the ratings dataframe, user ids, movie ids, and the mapping of user ids to indices
        titles=movies["title"].to_numpy(), genre_names=genres, genre_mat=genre_mat, # return the movie titles, genre names, and the multi-hot genre matrix
        tag_mat=tag_mat, n_tags=len(vocab) + 1, # return the tag matrix and the number of tags (including padding)
        movie_counts=ratings["m"].value_counts().reindex(range(n_movies), fill_value=0).to_numpy(), # return the number of ratings per movie (reindexed to include all movies, with 0 for movies with no ratings)
    )


def build_seen(ratings, n_users): # used to remove movies from a user's recommendation list that they have already rated
    """CSR-style (indptr, items) of the movies each user has already rated."""
    r = ratings.sort_values("u")
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(r["u"].to_numpy(), minlength=n_users), out=indptr[1:])
    return indptr, r["m"].to_numpy(), r["rating"].to_numpy()