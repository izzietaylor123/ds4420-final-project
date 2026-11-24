"""Movie recommendation MLP with collaborative-filtering embeddings.

Inputs per (user, movie) pair:
  * user ID            -> learned user embedding
  * movie ID           -> learned movie embedding
  * genre multi-hot    -> 20-ish dim binary vector from movie.csv
  * average tag embed. -> mean of learned embeddings of the movie's user-applied
                          tags (tag.csv); movies with no tags get a zero vector

Target (--target):
  * rating : regress the 0.5-5.0 star rating (MSE)
  * like   : predict P(will like) = P(rating >= --like-threshold) (BCE)

Usage:
  python mlp_model.py train --user-id 42 --top-k 10     # train, then recommend
  python mlp_model.py recommend --user-id 42 --top-k 10 # reuse saved checkpoint
"""

import argparse
from pathlib import Path
from loading_data import load_data, build_seen

import numpy as np
import torch
import torch.nn as nn

CKPT = Path(__file__).parent / "mlp_model.pt"



class RecMLP(nn.Module): # simple multi-layer perceptron for movie recommendation
    def __init__(self, n_users, n_movies, genre_mat, tag_mat, n_tags,
                 emb_dim=32, tag_dim=32, hidden=(128, 64), dropout=0.2, global_mean=0.0):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        # Average tag embedding: padding (id 0) is masked out of the mean, and
        # movies with no tags yield a zero vector.
        self.tag_emb = nn.Embedding(n_tags, tag_dim, padding_idx=0)
        self.register_buffer("genre_mat", torch.as_tensor(genre_mat)) # register genre_mat as a buffer so it is not a parameter to be learned, but still part of the model's state
        self.register_buffer("tag_mat", torch.as_tensor(tag_mat)) # same with tag_mat
        self.register_buffer("global_mean", torch.tensor(float(global_mean))) # same with global_mean

        layers, d = [], emb_dim * 2 + genre_mat.shape[1] + tag_dim # input dimension is the concatenation of user embedding, movie embedding, genre multi-hot, and average tag embedding
        for h in hidden: # add hidden layers with ReLU activation and dropout
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers) # create the MLP as a sequential model
        for e in (self.user_emb, self.movie_emb, self.tag_emb): # initialize the embeddings with a normal distribution
            nn.init.normal_(e.weight, std=0.05)
        with torch.no_grad(): # zero out the padding embedding for tags so that it does not contribute to the average tag embedding
            self.tag_emb.weight[0].zero_()

    def avg_tag_embedding(self, m): # compute the average tag embedding for a batch of movies m
        ids = self.tag_mat[m]                          # (B, max_tags)
        mask = (ids > 0).unsqueeze(-1).float()
        summed = (self.tag_emb(ids) * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1.0)

    def forward(self, u, m): # compute the forward pass for a batch of user indices u and movie indices m
        x = torch.cat(
            [self.user_emb(u), self.movie_emb(m), self.genre_mat[m], self.avg_tag_embedding(m)],
            dim=1,
        )
        return self.mlp(x).squeeze(1) + self.global_mean  # logit (like) or rating


# Training + evaluation ------------------------------------------------------------- #

def pick_device(): # pick the device to run the model on (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_target(r, target, thr): # convert the ratings r to the appropriate target format based on the specified target and threshold
    return (r >= thr).float() if target == "like" else r


@torch.no_grad()
def evaluate(model, u, m, r, target, thr, bs=65536): # evaluate the model on a batch of user indices u, movie indices m, and ratings r, using the specified target and threshold
    model.eval()
    out = torch.cat([model(u[i:i + bs], m[i:i + bs]) for i in range(0, len(u), bs)])
    if target == "like": # if the target is "like", compute the binary cross-entropy loss and accuracy
        y = (r >= thr).float()
        bce = nn.functional.binary_cross_entropy_with_logits(out, y).item()
        acc = ((out > 0).float() == y).float().mean().item()
        return {"bce": bce, "acc": acc}
    pred = out.clamp(0.5, 5.0)
    return {"rmse": ((pred - r) ** 2).mean().sqrt().item(), "mae": (pred - r).abs().mean().item()} # return the root mean squared error and mean absolute error for the predicted ratings


def train(args): # train the model using the specified arguments
    torch.manual_seed(args.seed) # set the random seed for reproducibility
    d = load_data(args.sample_users, args.min_movie_ratings, args.min_tag_count,
                  args.max_tags, args.seed, keep_users=[args.user_id] if args.user_id else ())
    df = d["ratings"] # pull out the df
    device = pick_device()
    print("device:", device)

    # random 80/10/10 split
    perm = np.random.default_rng(args.seed).permutation(len(df))
    n_val = n_test = int(0.1 * len(df))
    idx = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}
    t = {k: (torch.as_tensor(df["u"].to_numpy()[v], device=device),
             torch.as_tensor(df["m"].to_numpy()[v], device=device),
             torch.as_tensor(df["rating"].to_numpy()[v], device=device)) for k, v in idx.items()}

    train_mean = float(t["train"][2].mean()) # compute the mean rating for the training set, which will be used as a baseline for evaluation
    base = train_mean if args.target == "rating" else 0.0
    model = RecMLP(len(d["user_ids"]), len(d["movie_ids"]), d["genre_mat"], d["tag_mat"], d["n_tags"],
                   args.emb_dim, args.tag_dim, tuple(args.hidden), args.dropout, base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if args.target == "like" else nn.MSELoss()

    u, m, r = t["train"] # pull out the training set user indices, movie indices, and ratings

    # per-movie mean of the target on the training set (global mean for movies with no training ratings);
    # used as a popularity baseline and to re-rank recommendations by personalization
    n_movies = len(d["movie_ids"])
    y_train = make_target(r, args.target, args.like_threshold).cpu().double() # on the CPU because MPS has no float64
    m_cpu = m.cpu()
    cnt = torch.bincount(m_cpu, minlength=n_movies).double()
    tot = torch.bincount(m_cpu, weights=y_train, minlength=n_movies)
    movie_mean = torch.where(cnt > 0, tot / cnt.clamp(min=1.0), y_train.mean()).float().numpy()

    # baselines for context
    vu, vm, vr = t["val"] # pull out the validation set user indices, movie indices, and ratings
    if args.target == "rating":
        print(f"baseline (predict train mean) val RMSE: {((vr - train_mean) ** 2).mean().sqrt():.4f}")
        mm_pred = torch.as_tensor(movie_mean, device=device)[vm]
        print(f"baseline (predict movie mean) val RMSE: {((vr - mm_pred) ** 2).mean().sqrt():.4f}")
    best, best_state, bad = float("inf"), None, 0 # initialize the best score, best model state, and bad epoch counter for early stopping
    for epoch in range(1, args.epochs + 1): # repeats through epochs
        model.train() # set the model to training mode
        order = torch.randperm(len(u), device=device)
        total = 0.0
        for i in range(0, len(u), args.batch_size): # iterate through the training set in batches of size args.batch_size
            b = order[i:i + args.batch_size] # get the indices for the current batch
            opt.zero_grad() # zero the gradients for the optimizer
            loss = loss_fn(model(u[b], m[b]), make_target(r[b], args.target, args.like_threshold)) # compute the loss for the current batch
            loss.backward() # backpropagate the loss to compute gradients
            opt.step() # update the model parameters using the optimizer
            total += loss.item() * len(b) # accumulate the total loss for the epoch
        val = evaluate(model, vu, vm, vr, args.target, args.like_threshold) # evaluate the model on the validation set and get the evaluation metrics
        score = val["rmse"] if args.target == "rating" else val["bce"] # choose the appropriate score for early stopping based on the target
        print(f"epoch {epoch:02d} train_loss {total / len(u):.4f} val {val}")
        if score < best - 1e-4: # if the current score is better than the best score by a small margin, update the best score and save the model state
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience: # if the model has not improved for args.patience epochs, stop training early
                print("early stopping")
                break

    model.load_state_dict(best_state) # load the best model state after training is complete
    print("test:", evaluate(model, *t["test"], args.target, args.like_threshold)) # evaluate the model on the test set and print the evaluation metrics

    indptr, items, seen_ratings = build_seen(df, len(d["user_ids"])) # get the seen movies so we can remove them from the recommendation list
    torch.save({ # save the model state, configuration, and other relevant data to a checkpoint file so we can access it without retraining
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": dict(n_users=len(d["user_ids"]), n_movies=len(d["movie_ids"]), n_tags=d["n_tags"],
                       emb_dim=args.emb_dim, tag_dim=args.tag_dim, hidden=tuple(args.hidden),
                       dropout=args.dropout, target=args.target, like_threshold=args.like_threshold),
        "user_ids": d["user_ids"], "movie_ids": d["movie_ids"], "titles": d["titles"],
        "genre_mat": d["genre_mat"], "tag_mat": d["tag_mat"], "movie_counts": d["movie_counts"],
        "movie_mean": movie_mean,
        "seen_indptr": indptr, "seen_items": items, "seen_ratings": seen_ratings,
    }, CKPT)
    print("saved", CKPT)


# Recommendation ------------------------------------------------------------- #
"""
Recommends movies to a specific user without training the model (need to run training first)
* user_id : MovieLens userId to recommend for
* top_k : number of recommendations to return
* min_count : only recommend movies with at least this many training ratings
* alpha : popularity penalty; movies are ranked by score - alpha * movie_mean, so alpha=0 ranks by the
          raw score and alpha=1 ranks by how much the model expects this user to deviate from the movie's average
"""
@torch.no_grad()
def recommend(user_id, top_k, min_count, alpha=0.0):
    ck = torch.load(CKPT, weights_only=False) # load the checkpoint file
    c = ck["config"]
    model = RecMLP(c["n_users"], c["n_movies"], ck["genre_mat"], ck["tag_mat"], c["n_tags"],
                   c["emb_dim"], c["tag_dim"], c["hidden"], c["dropout"]) # initialize the model with the configuration from the checkpoint
    model.load_state_dict(ck["state"])
    model.eval()

    pos = np.searchsorted(ck["user_ids"], user_id) # find the position of the user_id in the sorted list of user_ids from the checkpoint
    if pos >= len(ck["user_ids"]) or ck["user_ids"][pos] != user_id:
        raise SystemExit(f"user {user_id} was not in the training sample; retrain with --user-id {user_id}") # if the user was not specified during training, they may not have been in the random sample and do not have a checkpoint
    seen = ck["seen_items"][ck["seen_indptr"][pos]:ck["seen_indptr"][pos + 1]] # get the indices of the movies that the user has already rated

    m = torch.arange(c["n_movies"]) # create a tensor of all movie indices
    scores = model(torch.full_like(m, pos), m) # get the scores for all movies for the specified user
    if c["target"] == "like": # if the target is "like", apply the sigmoid function to convert logits to probabilities
        scores = torch.sigmoid(scores)
    scores = scores.numpy()
    if alpha > 0: # if there was an alpha, apply a popularity penalty to the scores by subtracting alpha times the movie's mean rating from the scores
        if "movie_mean" not in ck:
            raise SystemExit("checkpoint has no movie_mean; retrain to use --rerank-alpha")
        rank_scores = scores - alpha * ck["movie_mean"] # remove the movie's baseline popularity from the ranking
    else:
        rank_scores = scores.copy()
    rank_scores[seen] = -np.inf # only use unseen movies
    rank_scores[ck["movie_counts"] < min_count] = -np.inf  # skip obscure titles
    label = "P(like)" if c["target"] == "like" else "pred rating"
    seen_r = ck["seen_ratings"][ck["seen_indptr"][pos]:ck["seen_indptr"][pos + 1]]  # the user's actual ratings
    print(f"\nUser's top {top_k} rated movies (for context):")
    for rank, i in enumerate(np.argsort(-seen_r, kind="stable")[:top_k], 1): # print the user's top rated movies to sanity check their recommendations
        print(f"{rank:2d}. {ck['titles'][seen[i]]:<60} rating: {seen_r[i]:.1f}")
    print(f"\nTop {top_k} recommendations for user {user_id} ({len(seen)} movies already rated):")
    for rank, i in enumerate(np.argsort(-rank_scores)[:top_k], 1): # print the user's top recommended movies with their predicted ratings
        val = f"{scores[i]:.1%}" if c["target"] == "like" else f"{min(scores[i], 5.0):.2f}"
        print(f"{rank:2d}. {ck['titles'][i]:<60} {label}: {val}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "recommend"])
    p.add_argument("--user-id", type=int, help="MovieLens userId to recommend for")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rec-min-ratings", type=int, default=50,
                   help="only recommend movies with at least this many training ratings")
    p.add_argument("--rerank-alpha", type=float, default=0.0,
                   help="rank by score - alpha * movie_mean to reduce popularity bias (0 = raw score)")
    p.add_argument("--target", choices=["rating", "like"], default="rating")
    p.add_argument("--like-threshold", type=float, default=4.0)
    p.add_argument("--sample-users", type=int, default=20000,
                   help="random user subset to keep training fast (0 = all users)")
    p.add_argument("--min-movie-ratings", type=int, default=5)
    p.add_argument("--min-tag-count", type=int, default=2)
    p.add_argument("--max-tags", type=int, default=32)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--tag-dim", type=int, default=32)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.mode == "train":
        train(args)
    if args.user_id is not None:
        recommend(args.user_id, args.top_k, args.rec_min_ratings, args.rerank_alpha) # recommend based on a user id, top k recommendations, and minimum ratings
    elif args.mode == "recommend":
        p.error("recommend requires --user-id")


if __name__ == "__main__":
    main()
