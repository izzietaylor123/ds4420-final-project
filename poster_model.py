import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from loading_data import DATA, load_data

RESULTS = Path(__file__).parent / "results"


def load_ratings(sample_users=20000, min_movie_ratings=5, seed=0): # Same data as the MLP and Bayesian model
    return load_data(sample_users, min_movie_ratings, min_tag_count=2, max_tags=32, seed=seed)


def split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    n_val = n_test = int(0.1 * n)
    return {"test": perm[:n_test], "val": perm[n_test:n_test + n_val], "train": perm[n_test + n_val:]}

KS = [5, 10, 20, 50, 100, 0]
LAMS = [0.0, 3.0, 10.0, 30.0, 100.0]
GRID = [(k, lam) for k in KS for lam in LAMS]

def genre_features(movie_ids):
    g = pd.read_csv(DATA / "movie.csv").set_index("movieId").loc[movie_ids, "genres"].str.get_dummies("|")
    g = g.drop(columns=[c for c in ("(no genres listed)", "IMAX") if c in g])
    return movie_ids, g.to_numpy(np.float32)


def load_features(movie_ids, source="poster_features_cnn.npz", shuffle_seed=None):
    if source.startswith("genre+"):
        Fg, hg = load_features(movie_ids, "genre")
        Fp, hp = load_features(movie_ids, source[len("genre+"):], shuffle_seed)
        has = hg & hp
        F = np.zeros((len(movie_ids), Fg.shape[1] + Fp.shape[1]), dtype=np.float32)
        F[has] = np.hstack([Fg[has], Fp[has]]) / np.sqrt(2)
        return F, has
    if source == "genre":
        ids, feats = genre_features(movie_ids)
    else:
        z = np.load(DATA / source)
        ids, feats = z["movie_ids"], z["features"]
    pos = np.searchsorted(movie_ids, ids)
    ok = (pos < len(movie_ids)) & (movie_ids[np.minimum(pos, len(movie_ids) - 1)] == ids)
    feats = feats[ok]
    if source == "genre":
        ok_any = feats.sum(1) > 0  # If no genre listed then no features
        pos, ok = pos[ok][ok_any], np.ones(ok_any.sum(), bool)
        feats = feats[ok_any]
    if shuffle_seed is not None:  # control -- randomly reassign posters to movies
        feats = feats[np.random.default_rng(shuffle_seed).permutation(len(feats))]
    F = np.zeros((len(movie_ids), feats.shape[1]), dtype=np.float32)
    F[pos[ok]] = feats
    has = np.zeros(len(movie_ids), dtype=bool)
    has[pos[ok]] = True
    F[has] -= F[has].mean(0)
    F[has] /= np.linalg.norm(F[has], axis=1, keepdims=True) + 1e-8
    return F, has


def fit_biases(u, m, r, n_users, n_movies, lam=10.0, iters=3):
    mu = r.mean()
    bu, bi = np.zeros(n_users), np.zeros(n_movies)
    nu = np.bincount(u, minlength=n_users)
    ni = np.bincount(m, minlength=n_movies)
    for _ in range(iters):
        bi = np.bincount(m, r - mu - bu[u], minlength=n_movies) / (lam + ni)
        bu = np.bincount(u, r - mu - bi[m], minlength=n_users) / (lam + nu)
    return mu, bu, bi


def group_by_user(u, n_users):
    order = np.argsort(u, kind="stable")
    ptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=n_users), out=ptr[1:])
    return order, ptr

class PosterSimilarity:
    def __init__(self, F, has, train_u, train_m, train_r, n_users, center="bias"):
        self.F, self.has, self.center = F, has, center
        n_movies = len(F)
        self.mu, self.bu, self.bi = fit_biases(train_u, train_m, train_r, n_users, n_movies)
        if center == "user":
            s = np.bincount(train_u, train_r, minlength=n_users)
            c = np.bincount(train_u, minlength=n_users)
            self.user_mean = np.where(c > 0, s / np.maximum(c, 1), train_r.mean())
        order, self.ptr = group_by_user(train_u, n_users)
        self.tm, self.tr = train_m[order], train_r[order]
        self.tres = self.tr - self.baseline(train_u[order], self.tm)

    def baseline(self, u, m):
        if self.center == "user":
            return self.user_mean[u]
        return self.mu + self.bu[u] + self.bi[m]

    def neighbours(self, u):
        sl = slice(self.ptr[u], self.ptr[u + 1])
        m, res = self.tm[sl], self.tres[sl]
        keep = self.has[m]
        return m[keep], res[keep]

    def predict(self, u, m, grid=GRID):
        base = self.baseline(u, m)
        out = np.repeat(base[:, None], len(grid), axis=1)
        order, ptr = group_by_user(u, len(self.ptr) - 1)
        for user in np.flatnonzero(np.diff(ptr)):
            rows = order[ptr[user]:ptr[user + 1]]
            rows = rows[self.has[m[rows]]]  # no poster -> keep baseline
            nm, nres = self.neighbours(user)
            if len(rows) == 0 or len(nm) == 0:
                continue
            S = np.maximum(self.F[m[rows]] @ self.F[nm].T, 0)  # (queries, neighbours), ignore dissimilar
            idx = np.argsort(-S, axis=1)
            S = np.take_along_axis(S, idx, 1)
            num = np.cumsum(S * nres[idx], axis=1)  # running weighted sums -> every k at once
            den = np.cumsum(S, axis=1)
            for j, (k, lam) in enumerate(grid):
                c = (len(nm) if k == 0 else min(k, len(nm))) - 1
                d = den[:, c] + lam
                out[rows, j] = base[rows] + np.where(d > 0, num[:, c] / np.maximum(d, 1e-12), 0)
        return np.clip(out, 0.5, 5.0)

    def score_all(self, user, k, lam=0.0):
        n = len(self.F)
        base = self.baseline(np.full(n, user), np.arange(n))
        nm, nres = self.neighbours(user)
        scores, because = base.copy(), np.full(n, -1)
        if len(nm):
            S = np.maximum(self.F @ self.F[nm].T, 0)
            S[~self.has] = 0
            kk = len(nm) if k == 0 else min(k, len(nm))
            top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            St = np.take_along_axis(S, top, 1)
            d = St.sum(1) + lam
            scores += np.where(d > 0, (St * nres[top]).sum(1) / np.maximum(d, 1e-12), 0)
            contrib = St * nres[top]
            because = nm[top[np.arange(n), contrib.argmax(1)]]
        return np.clip(scores, 0.5, 5.0), because

def rmse(p, r):
    return float(np.sqrt(np.mean((p - r) ** 2)))


def precision_at_k(model, k_nn, lam, rows_u, rows_m, rows_r, train_seen, n_eval, rng, top=10, thr=4.0):
    order, ptr = group_by_user(rows_u, len(model.ptr) - 1)
    users = [x for x in np.flatnonzero(np.diff(ptr))
             if (rows_r[order[ptr[x]:ptr[x + 1]]] >= thr).any()]
    users = rng.choice(users, min(n_eval, len(users)), replace=False)
    hits_model, hits_base = [], []
    for x in users:
        rr = order[ptr[x]:ptr[x + 1]]
        liked = set(rows_m[rr][rows_r[rr] >= thr])
        scores, _ = model.score_all(x, k_nn, lam)
        base = model.baseline(np.full(len(scores), x), np.arange(len(scores)))
        for s, hits in ((scores, hits_model), (base, hits_base)):
            s = s.copy()
            s[train_seen[x]] = -np.inf
            s[~model.has] = -np.inf
            rec = np.argpartition(-s, top)[:top]
            hits.append(len(liked.intersection(rec)) / top)
    return float(np.mean(hits_model)), float(np.mean(hits_base))


def evaluate(args):
    d = load_ratings(args.sample_users, args.min_movie_ratings, args.seed)
    df, n_users, n_movies = d["ratings"], len(d["user_ids"]), len(d["movie_ids"])
    u, m, r = df["u"].to_numpy(), df["m"].to_numpy(), df["rating"].to_numpy().astype(np.float64)
    sp = split(len(df), args.seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    F, has = load_features(d["movie_ids"], args.features)
    print(f"features: {args.features}")
    print(f"coverage: {has.mean():.1%} of movies, {has[m[te]].mean():.1%} of test ratings")

    mu, bu, bi = fit_biases(u[tr], m[tr], r[tr], n_users, n_movies)
    s = np.bincount(u[tr], r[tr], minlength=n_users); c = np.bincount(u[tr], minlength=n_users)
    umean = np.where(c > 0, s / np.maximum(c, 1), mu)
    print("\nbaselines (test RMSE):")
    print(f"  global mean              {rmse(np.full(len(te), mu), r[te]):.4f}")
    print(f"  user mean                {rmse(umean[u[te]], r[te]):.4f}")
    print(f"  mu + b_u + b_i           {rmse(np.clip(mu + bu[u[te]] + bi[m[te]], .5, 5), r[te]):.4f}")

    results = {}
    out = {"features": args.features, "center": args.center,
           "baselines": {"global_mean": rmse(np.full(len(te), mu), r[te]),
                         "user_mean": rmse(umean[u[te]], r[te]),
                         "bias": rmse(np.clip(mu + bu[u[te]] + bi[m[te]], .5, 5), r[te])}}
    for name, sh in (("posters", None), ("shuffled posters (control)", args.seed + 1)):
        F_, has_ = (F, has) if sh is None else load_features(d["movie_ids"], args.features, shuffle_seed=sh)
        model = PosterSimilarity(F_, has_, u[tr], m[tr], r[tr], n_users, center=args.center)
        val = model.predict(u[va], m[va])
        val_rmse = np.array([rmse(val[:, j], r[va]) for j in range(len(GRID))])
        best = int(np.argmin(val_rmse))
        k, lam = GRID[best]
        test = model.predict(u[te], m[te], grid=[GRID[best]])[:, 0]
        results[name] = (model, k, lam)
        print(f"\n{name}  (center={args.center})")
        print("  val RMSE (rows = k, cols = lam " + " ".join(map(str, LAMS)) + ")")
        for i, kk in enumerate(KS):
            print(f"    k={kk or 'all':<4} " + "  ".join(f"{v:.4f}" for v in val_rmse[i * len(LAMS):(i + 1) * len(LAMS)]))
        print(f"  best k={k or 'all'}, lam={lam} -> test RMSE {rmse(test, r[te]):.4f}  "
              f"MAE {np.mean(np.abs(test - r[te])):.4f}")
        out["shuffled" if sh else "model"] = {
            "k": k, "lam": lam, "test_rmse": rmse(test, r[te]),
            "test_mae": float(np.mean(np.abs(test - r[te]))), "val_rmse_grid": val_rmse.tolist()}

    if args.precision_users:
        model, k, lam = results["posters"]
        order, ptr = group_by_user(u[tr], n_users)
        seen = [m[tr][order[ptr[x]:ptr[x + 1]]] for x in range(n_users)]
        p_model, p_base = precision_at_k(model, k, lam, u[te], m[te], r[te], seen,
                                         args.precision_users, np.random.default_rng(args.seed))
        print(f"\nprecision@10 on {args.precision_users} users (relevant = test rating >= 4):")
        print(f"  baseline ranking only    {p_base:.4f}")
        print(f"  baseline + posters       {p_model:.4f}")
        out["precision_at_10"] = {"baseline": p_base, "model": p_model, "n_users": args.precision_users}

    RESULTS.mkdir(exist_ok=True)
    tag = args.features.replace(".npz", "").replace("poster_features", "posters").replace("+", "_plus_")
    with open(RESULTS / f"eval_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved results/eval_{tag}.json")


def recommend(args):
    d = load_ratings(args.sample_users, args.min_movie_ratings, args.seed)
    df, n_users = d["ratings"], len(d["user_ids"])
    pos = np.searchsorted(d["user_ids"], args.user_id)
    if pos >= n_users or d["user_ids"][pos] != args.user_id:
        raise SystemExit(f"user {args.user_id} is not in the {args.sample_users}-user sample")
    F, has = load_features(d["movie_ids"], args.features)  # all of the user's ratings are used here
    model = PosterSimilarity(F, has, df["u"].to_numpy(), df["m"].to_numpy(),
                      df["rating"].to_numpy().astype(np.float64), n_users, center=args.center)
    scores, because = model.score_all(pos, args.k, args.lam)
    mine = df[df["u"] == pos].sort_values("rating", ascending=False)
    counts = np.bincount(df["m"], minlength=len(scores))
    scores[mine["m"].to_numpy()] = -np.inf
    scores[(counts < args.rec_min_ratings) | ~has] = -np.inf

    titles = d["titles"]
    print(f"\nUser {args.user_id}'s top-rated movies ({len(mine)} rated):")
    for t, rt in zip(titles[mine["m"].to_numpy()[:args.top_k]], mine["rating"][:args.top_k]):
        print(f"  {rt:.1f}  {t}")
    print(f"\nTop {args.top_k} recommendations:")
    for i in np.argsort(-scores)[:args.top_k]:
        print(f"  {scores[i]:.2f}  {titles[i]:<55} poster looks like: {titles[because[i]]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["evaluate", "recommend"])
    p.add_argument("--features", default="poster_features_cnn.npz",
                   help="feature file in data/ (own CNN by default; poster_features.npz = ResNet-50), "
                        "or 'genre' for the genre-only control")
    p.add_argument("--center", choices=["bias", "user"], default="bias",
                   help="bias: mu+b_u+b_i baseline; user: user's mean only (pure poster content)")
    p.add_argument("--sample-users", type=int, default=20000)
    p.add_argument("--min-movie-ratings", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision-users", type=int, default=1000, help="0 to skip precision@10")
    p.add_argument("--user-id", type=int)
    p.add_argument("--k", type=int, default=20, help="neighbours for recommend (0 = all)")
    p.add_argument("--lam", type=float, default=10.0, help="shrinkage for recommend")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--rec-min-ratings", type=int, default=50)
    args = p.parse_args()
    if args.mode == "evaluate":
        evaluate(args)
    else:
        if args.user_id is None:
            p.error("recommend requires --user-id")
        recommend(args)


if __name__ == "__main__":
    main()
