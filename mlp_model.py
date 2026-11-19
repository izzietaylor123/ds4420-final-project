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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = Path(__file__).parent / "data"
CKPT = Path(__file__).parent / "mlp_model.pt"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(sample_users, min_movie_ratings, min_tag_count, max_tags, seed, keep_users=()):
    """Read the CSVs and build index maps plus per-movie feature tables."""
    rng = np.random.default_rng(seed)

    print("Reading rating.csv ...")
    ratings = pd.read_csv(
        DATA / "rating.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    if sample_users and sample_users < ratings["userId"].nunique():
        users = ratings["userId"].unique()
        chosen = set(rng.choice(users, sample_users, replace=False).tolist()) | set(keep_users)
        ratings = ratings[ratings["userId"].isin(chosen)]

    counts = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(counts[counts >= min_movie_ratings].index)]
    ratings = ratings.reset_index(drop=True)

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    ratings["u"] = ratings["userId"].map(user2idx).astype(np.int64)
    ratings["m"] = ratings["movieId"].map(movie2idx).astype(np.int64)
    n_movies = len(movie_ids)

    # --- genres: multi-hot -------------------------------------------------
    movies = pd.read_csv(DATA / "movie.csv").set_index("movieId").loc[movie_ids]
    genre_lists = movies["genres"].str.split("|")
    genres = sorted({g for gl in genre_lists for g in gl if g != "(no genres listed)"})
    g2i = {g: i for i, g in enumerate(genres)}
    genre_mat = np.zeros((n_movies, len(genres)), dtype=np.float32)
    for row, gl in enumerate(genre_lists):
        for g in gl:
            if g in g2i:
                genre_mat[row, g2i[g]] = 1.0

    # --- tags: padded index matrix (0 = padding) ----------------------------
    tags = pd.read_csv(DATA / "tag.csv", usecols=["movieId", "tag"]).dropna()
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags = tags[tags["movieId"].isin(movie2idx)]
    tag_counts = tags["tag"].value_counts()
    vocab = tag_counts[tag_counts >= min_tag_count].index
    t2i = {t: i + 1 for i, t in enumerate(vocab)}  # 0 reserved for padding
    tags = tags[tags["tag"].isin(t2i)]
    # per-movie most frequent tags first, capped at max_tags
    per_movie = tags.groupby(["movieId", "tag"]).size().rename("n").reset_index()
    per_movie = per_movie.sort_values(["movieId", "n"], ascending=[True, False])
    per_movie["pos"] = per_movie.groupby("movieId").cumcount()
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
        ratings=ratings, user_ids=user_ids, movie_ids=movie_ids, user2idx=user2idx,
        titles=movies["title"].to_numpy(), genre_names=genres, genre_mat=genre_mat,
        tag_mat=tag_mat, n_tags=len(vocab) + 1,
        movie_counts=ratings["m"].value_counts().reindex(range(n_movies), fill_value=0).to_numpy(),
    )


def build_seen(ratings, n_users):
    """CSR-style (indptr, items) of the movies each user has already rated."""
    r = ratings.sort_values("u")
    indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(r["u"].to_numpy(), minlength=n_users), out=indptr[1:])
    return indptr, r["m"].to_numpy()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RecMLP(nn.Module):
    def __init__(self, n_users, n_movies, genre_mat, tag_mat, n_tags,
                 emb_dim=32, tag_dim=32, hidden=(128, 64), dropout=0.2, global_mean=0.0):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        # Average tag embedding: padding (id 0) is masked out of the mean, and
        # movies with no tags yield a zero vector.
        self.tag_emb = nn.Embedding(n_tags, tag_dim, padding_idx=0)
        self.register_buffer("genre_mat", torch.as_tensor(genre_mat))
        self.register_buffer("tag_mat", torch.as_tensor(tag_mat))
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        layers, d = [], emb_dim * 2 + genre_mat.shape[1] + tag_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)
        for e in (self.user_emb, self.movie_emb, self.tag_emb):
            nn.init.normal_(e.weight, std=0.05)
        with torch.no_grad():
            self.tag_emb.weight[0].zero_()

    def avg_tag_embedding(self, m):
        ids = self.tag_mat[m]                          # (B, max_tags)
        mask = (ids > 0).unsqueeze(-1).float()
        summed = (self.tag_emb(ids) * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1.0)

    def forward(self, u, m):
        x = torch.cat(
            [self.user_emb(u), self.movie_emb(m), self.genre_mat[m], self.avg_tag_embedding(m)],
            dim=1,
        )
        return self.mlp(x).squeeze(1) + self.global_mean  # logit (like) or rating


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_target(r, target, thr):
    return (r >= thr).float() if target == "like" else r


@torch.no_grad()
def evaluate(model, u, m, r, target, thr, bs=65536):
    model.eval()
    out = torch.cat([model(u[i:i + bs], m[i:i + bs]) for i in range(0, len(u), bs)])
    if target == "like":
        y = (r >= thr).float()
        bce = nn.functional.binary_cross_entropy_with_logits(out, y).item()
        acc = ((out > 0).float() == y).float().mean().item()
        return {"bce": bce, "acc": acc}
    pred = out.clamp(0.5, 5.0)
    return {"rmse": ((pred - r) ** 2).mean().sqrt().item(), "mae": (pred - r).abs().mean().item()}


def train(args):
    torch.manual_seed(args.seed)
    d = load_data(args.sample_users, args.min_movie_ratings, args.min_tag_count,
                  args.max_tags, args.seed, keep_users=[args.user_id] if args.user_id else ())
    df = d["ratings"]
    device = pick_device()
    print("device:", device)

    # random 80/10/10 split
    perm = np.random.default_rng(args.seed).permutation(len(df))
    n_val = n_test = int(0.1 * len(df))
    idx = {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}
    t = {k: (torch.as_tensor(df["u"].to_numpy()[v], device=device),
             torch.as_tensor(df["m"].to_numpy()[v], device=device),
             torch.as_tensor(df["rating"].to_numpy()[v], device=device)) for k, v in idx.items()}

    train_mean = float(t["train"][2].mean())
    base = train_mean if args.target == "rating" else 0.0
    model = RecMLP(len(d["user_ids"]), len(d["movie_ids"]), d["genre_mat"], d["tag_mat"], d["n_tags"],
                   args.emb_dim, args.tag_dim, tuple(args.hidden), args.dropout, base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if args.target == "like" else nn.MSELoss()

    # baseline for context
    vu, vm, vr = t["val"]
    if args.target == "rating":
        print(f"baseline (predict train mean) val RMSE: {((vr - train_mean) ** 2).mean().sqrt():.4f}")

    u, m, r = t["train"]
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(u), device=device)
        total = 0.0
        for i in range(0, len(u), args.batch_size):
            b = order[i:i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(u[b], m[b]), make_target(r[b], args.target, args.like_threshold))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val = evaluate(model, vu, vm, vr, args.target, args.like_threshold)
        score = val["rmse"] if args.target == "rating" else val["bce"]
        print(f"epoch {epoch:02d} train_loss {total / len(u):.4f} val {val}")
        if score < best - 1e-4:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    print("test:", evaluate(model, *t["test"], args.target, args.like_threshold))

    indptr, items = build_seen(df, len(d["user_ids"]))
    torch.save({
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": dict(n_users=len(d["user_ids"]), n_movies=len(d["movie_ids"]), n_tags=d["n_tags"],
                       emb_dim=args.emb_dim, tag_dim=args.tag_dim, hidden=tuple(args.hidden),
                       dropout=args.dropout, target=args.target, like_threshold=args.like_threshold),
        "user_ids": d["user_ids"], "movie_ids": d["movie_ids"], "titles": d["titles"],
        "genre_mat": d["genre_mat"], "tag_mat": d["tag_mat"], "movie_counts": d["movie_counts"],
        "seen_indptr": indptr, "seen_items": items,
    }, CKPT)
    print("saved", CKPT)


# --------------------------------------------------------------------------- #
# Recommendation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def recommend(user_id, top_k, min_count):
    ck = torch.load(CKPT, weights_only=False)
    c = ck["config"]
    model = RecMLP(c["n_users"], c["n_movies"], ck["genre_mat"], ck["tag_mat"], c["n_tags"],
                   c["emb_dim"], c["tag_dim"], c["hidden"], c["dropout"])
    model.load_state_dict(ck["state"])
    model.eval()

    pos = np.searchsorted(ck["user_ids"], user_id)
    if pos >= len(ck["user_ids"]) or ck["user_ids"][pos] != user_id:
        raise SystemExit(f"user {user_id} was not in the training sample; retrain with --user-id {user_id}")
    seen = ck["seen_items"][ck["seen_indptr"][pos]:ck["seen_indptr"][pos + 1]]

    m = torch.arange(c["n_movies"])
    scores = model(torch.full_like(m, pos), m)
    if c["target"] == "like":
        scores = torch.sigmoid(scores)
    scores = scores.numpy()
    scores[seen] = -np.inf                       # only unseen movies
    scores[ck["movie_counts"] < min_count] = -np.inf  # skip obscure titles
    label = "P(like)" if c["target"] == "like" else "pred rating"
    print(f"\nTop {top_k} recommendations for user {user_id} ({len(seen)} movies already rated):")
    for rank, i in enumerate(np.argsort(-scores)[:top_k], 1):
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
        recommend(args.user_id, args.top_k, args.rec_min_ratings) # recommend based on a user id, top k recommendations, and minimum ratings
    elif args.mode == "recommend":
        p.error("recommend requires --user-id")


if __name__ == "__main__":
    main()
