
import numpy as np
from loading_data import load_data
import matplotlib.pyplot as plt
from scipy.stats import norm


# bayesian model
def calculate_posterior(genre_counts, genre_sums, prior_mean, prior_var, rating_var):
    """Calculate the Normal-Normal posterior for each user and genre."""
    # calculate the posterior variance and mean using the prior and observed ratings
    post_var = 1 / (1 / prior_var + genre_counts / rating_var)
    post_mean = post_var * (
        prior_mean / prior_var + genre_sums / rating_var
    )
    return post_mean, post_var


def predict_ratings(users, movies, post_mean, genre_mat, prior_mean):
    """Predict movie ratings using the user's posterior genre preferences."""
    movie_genres = genre_mat[movies]
    genre_scores = post_mean[users]

    # average the posterior means across the movie's genres
    genre_total = np.sum(movie_genres, axis=1)
    predictions = np.divide(
        np.sum(genre_scores * movie_genres, axis=1),
        genre_total,
        out=np.full(len(users), prior_mean, dtype=np.float64),
        where=genre_total > 0
    )
    return np.clip(predictions, 0.5, 5.0)


def evaluate(users, movies, actual, post_mean, genre_mat, prior_mean):
    """Calculate the MAE and RMSE for predicted ratings."""
    predictions = predict_ratings(users, movies, post_mean, genre_mat, prior_mean)

    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}

# visualizations
def plot_prior_posterior(user_id, genre, d, post_mean, post_var, prior_mean, prior_var):
    """Plot the prior and posterior distributions for a user's genre preference."""

    # get the selected user's index and genre
    user_idx = d["user2idx"][user_id]
    genre_idx = d["genre_names"].index(genre)

    # get the posterior mean and variance for the selected user and genre
    genre_mean = post_mean[user_idx, genre_idx]
    genre_var = post_var[user_idx, genre_idx]

    # create rating values for the distribution curves
    x = np.linspace(0.5, 5.5, 500)

    prior = norm.pdf(x, prior_mean, np.sqrt(prior_var))
    posterior = norm.pdf(x, genre_mean, np.sqrt(genre_var))

    # plot the prior and posterior distributions
    plt.figure(figsize=(8, 5))
    plt.plot(x, prior, label="Prior")
    plt.plot(x, posterior, label="Posterior")

    plt.xlabel("Genre Preference (rating points)")
    plt.ylabel("Probability Density")
    plt.title(f"Prior vs. Posterior: User {user_id}'s {genre} Preference")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bayesian_prior_posterior.png", dpi=300)
    plt.show()
    plt.close()

# model training and evaluation
def train():
    # load the movie ratings and features using the same settings as the MLP
    d = load_data(
        sample_users=20000,
        min_movie_ratings=5,
        min_tag_count=2,
        max_tags=32,
        seed=0
    )

    df = d["ratings"]

    # random 80/10/10 split using the same seed as the MLP
    perm = np.random.default_rng(0).permutation(len(df))
    n_val = n_test = int(0.1 * len(df))

    idx = {
        "test": perm[:n_test],
        "val": perm[n_test:n_test + n_val],
        "train": perm[n_test + n_val:]
    }

    train_df = df.iloc[idx["train"]]
    val_df = df.iloc[idx["val"]]
    test_df = df.iloc[idx["test"]]

    print("training:", len(train_df))
    print("validation:", len(val_df))
    print("testing:", len(test_df))

    # calculate the prior mean and rating variance using training data
    prior_mean = float(train_df["rating"].mean())
    rating_var = float(train_df["rating"].var())

    print("prior mean:", prior_mean)
    print("rating variance:", rating_var)

    # get the movie genres and the number of users and genres
    genre_mat = d["genre_mat"]
    n_users = len(d["user_ids"])
    n_genres = len(d["genre_names"])

    # count the ratings and calculate the rating sums for each user and genre
    genre_counts = np.zeros((n_users, n_genres), dtype=np.float32)
    genre_sums = np.zeros((n_users, n_genres), dtype=np.float64)

    train_users = train_df["u"].to_numpy()
    train_movies = train_df["m"].to_numpy()
    train_ratings = train_df["rating"].to_numpy()

    np.add.at(genre_counts, train_users, genre_mat[train_movies])
    np.add.at(genre_sums, train_users, genre_mat[train_movies] * train_ratings[:, None])

    val_users = val_df["u"].to_numpy()
    val_movies = val_df["m"].to_numpy()
    val_actual = val_df["rating"].to_numpy()

    # calculate the baseline error using the training mean
    baseline_mae = np.mean(np.abs(prior_mean - val_actual))
    baseline_rmse = np.sqrt(np.mean((prior_mean - val_actual) ** 2))

    print("baseline MAE:", round(baseline_mae, 4))
    print("baseline RMSE:", round(baseline_rmse, 4))

    # compare different prior variances using validation RMSE
    best_var = None
    best_rmse = float("inf")

    # store the validation results for plotting
    prior_variances = []
    validation_rmses = []

    for var in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        post_mean, post_var = calculate_posterior(
            genre_counts, genre_sums, prior_mean, var, rating_var
        )

        val = evaluate(val_users, val_movies, val_actual, post_mean, genre_mat, prior_mean)
        print(f"prior variance: {var:.2f} | validation RMSE: {val['rmse']:.4f}")

        prior_variances.append(var)
        validation_rmses.append(val["rmse"])

        if val["rmse"] < best_rmse:
            best_rmse = val["rmse"]
            best_var = var

    # plot how prior variance affects validation RMSE
    plt.figure(figsize=(8, 5))
    plt.plot(prior_variances, validation_rmses, marker="o")

    plt.xlabel("Prior Variance")
    plt.ylabel("Validation RMSE (rating points)")
    plt.title("Effect of Prior Variance on Bayesian Model Performance")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bayesian_prior_variance.png", dpi=300)
    plt.show()
    plt.close()

    # use the best prior variance from validation
    post_mean, post_var = calculate_posterior(
        genre_counts, genre_sums, prior_mean, best_var, rating_var
    )

    # for i, genre in enumerate(d["genre_names"]):
    #     print(genre, "ratings:", int(genre_counts[user_idx, i]),
    #         "posterior mean:", round(post_mean[user_idx, i], 2),
    #         "posterior variance:", round(post_var[user_idx, i], 2))

    print("best prior variance:", best_var)

    # evaluate the final model on the test set
    test_users = test_df["u"].to_numpy()
    test_movies = test_df["m"].to_numpy()
    test_actual = test_df["rating"].to_numpy()

    test = evaluate(test_users, test_movies, test_actual, post_mean, genre_mat, prior_mean)
    print("test:", test)
    return d, df, train_df, post_mean, post_var, prior_mean, best_var


# movie recommendations
def recommend(user_id, top_k, min_count, d, df, post_mean, prior_mean):
    """Recommend the highest rated unseen movies for a specific user."""
    if user_id not in d["user2idx"]:
        print(f"user {user_id} was not in the training sample")
        return

    user_idx = d["user2idx"][user_id]

    # predict ratings for every movie for the selected user
    all_movies = np.arange(len(d["movie_ids"]))
    all_users = np.full(len(all_movies), user_idx)

    scores = predict_ratings(all_users, all_movies, post_mean, d["genre_mat"], prior_mean)

    # remove movies the user has already rated
    seen_movies = df.loc[df["u"] == user_idx, "m"].to_numpy()
    scores[seen_movies] = -np.inf

    # skip movies with fewer than the minimum number of ratings
    scores[d["movie_counts"] < min_count] = -np.inf

    top_movies = np.argsort(-scores)[:top_k]

    print(f"\nTop {top_k} recommendations for user {user_id}:")
    for rank, m in enumerate(top_movies, 1):
        print(f"{rank:2d}. {d['titles'][m]:<60} predicted rating: {scores[m]:.2f}")


def main():
    d, df, train_df, post_mean, post_var, prior_mean, best_var = train()

    # generate movie recommendations
    recommend(2, 10, 50, d, df, post_mean, prior_mean)

    # plot the prior and posterior distributions
    plot_prior_posterior(2, "Adventure", d, post_mean, post_var, prior_mean, best_var)

if __name__ == "__main__":
    main()
