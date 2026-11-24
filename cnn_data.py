from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def load_ratings(sample_users=20000, min_movie_ratings=5, seed=0):
    rng = np.random.default_rng(seed)
    ratings = pd.read_csv(
        DATA / "rating.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    if sample_users and sample_users < ratings["userId"].nunique():
        users = ratings["userId"].unique()
        chosen = set(rng.choice(users, sample_users, replace=False).tolist())
        ratings = ratings[ratings["userId"].isin(chosen)]

    counts = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(counts[counts >= min_movie_ratings].index)]
    ratings = ratings.reset_index(drop=True)

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    ratings["u"] = np.searchsorted(user_ids, ratings["userId"].to_numpy())
    ratings["m"] = np.searchsorted(movie_ids, ratings["movieId"].to_numpy())

    titles = pd.read_csv(DATA / "movie.csv").set_index("movieId").loc[movie_ids, "title"].to_numpy()
    print(f"{len(ratings):,} ratings | {len(user_ids):,} users | {len(movie_ids):,} movies")
    return dict(ratings=ratings, user_ids=user_ids, movie_ids=movie_ids, titles=titles)


def split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    n_val = n_test = int(0.1 * n)
    return {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}
