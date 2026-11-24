import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from cnn_data import DATA

POSTERS = DATA / "posters"
RESULTS = Path(__file__).parent / "results"
MODEL_PATH = Path(__file__).parent / "models" / "poster_cnn.keras"
DROP = ("(no genres listed)", "IMAX")  # not real genres


def load_posters(height, width):
    files = sorted(POSTERS.glob("*.jpg"), key=lambda f: int(f.stem))
    imgs, ids = [], []
    for f in files:
        try:
            imgs.append(np.asarray(Image.open(f).convert("RGB").resize((width, height), Image.BILINEAR)))
            ids.append(int(f.stem))
        except Exception as e:
            print(f"  skipping {f.name}: {e}")
    return np.stack(imgs), np.array(ids)


def genre_labels(movie_ids):
    g = pd.read_csv(DATA / "movie.csv").set_index("movieId").loc[movie_ids, "genres"].str.get_dummies("|")
    g = g.drop(columns=[c for c in DROP if c in g])
    return g.to_numpy(np.float32), list(g.columns)


def build_model(height, width, n_genres, emb_dim=64, dropout=0.3, emb_act="relu"):
    from tensorflow import keras
    from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU,
                                         MaxPooling2D, Rescaling)

    X = Input(shape=(height, width, 3))
    x = Rescaling(1 / 255)(X)  # MinMax scale pixels to [0, 1], as in lecture
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    if emb_act == "leaky": # For negative inputs, so units cant get stuck at 0
        emb = LeakyReLU(0.1, name="embedding")(Dense(emb_dim)(x))
    else:
        emb = Dense(emb_dim, activation="relu", name="embedding")(x)
    x = Dropout(dropout)(emb)
    out = Dense(n_genres, activation="sigmoid")(x)
    return keras.Model(X, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--height", type=int, default=96)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emb-act", choices=["relu", "leaky"], default="relu",
                   help="activation of the embedding layer (leaky = LeakyReLU, avoids dead units)")
    args = p.parse_args()

    from sklearn.metrics import roc_auc_score 
    from tensorflow import keras
    keras.utils.set_random_seed(args.seed)

    print("loading posters ...")
    X, movie_ids = load_posters(args.height, args.width)
    Y, genres = genre_labels(movie_ids)
    print(f"{len(X):,} posters | {args.height}x{args.width}x3 | {len(genres)} genres")

    perm = np.random.default_rng(args.seed).permutation(len(X))
    n_val = n_test = int(0.15 * len(X))
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    model = build_model(args.height, args.width, len(genres), args.emb_dim, emb_act=args.emb_act)
    suffix = "" if args.emb_act == "relu" else f"_{args.emb_act}"
    out_path = DATA / f"poster_features_cnn{suffix}.npz"
    model_path = MODEL_PATH.with_name(f"poster_cnn{suffix}.keras")
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(multi_label=True, name="auc")])
    model.summary()
    hist = model.fit(
        X[tr], Y[tr], validation_data=(X[va], Y[va]), epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)], verbose=2)

    # per-genre test vs. random guessing
    P = model.predict(X[te], verbose=0)
    aucs = {g: float(roc_auc_score(Y[te, j], P[:, j])) for j, g in enumerate(genres)
            if 0 < Y[te, j].sum() < len(te)}
    prevalence = {g: float(Y[:, j].mean()) for j, g in enumerate(genres)}
    print("\ntest AUC by genre (random = 0.50):")
    for g, a in sorted(aucs.items(), key=lambda t: -t[1]):
        print(f"  {g:<12} {a:.3f}   ({prevalence[g]:.1%} of movies)")
    print(f"  macro-average {np.mean(list(aucs.values())):.3f}")

    # embeddings for every poster
    embed = keras.Model(model.input, model.get_layer("embedding").output)
    feats = embed.predict(X, batch_size=256, verbose=0).astype(np.float32)
    dead = int((feats.max(0) <= 0).sum())
    np.savez_compressed(out_path, movie_ids=movie_ids, features=feats, arch=f"own_cnn{suffix}")
    print(f"\nsaved embeddings {feats.shape} -> {out_path} ({dead} of {feats.shape[1]} units never active)")

    model_path.parent.mkdir(exist_ok=True)
    model.save(model_path)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"cnn_training{suffix}.json", "w") as f:
        json.dump({"history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
                   "test_auc": aucs, "macro_auc": float(np.mean(list(aucs.values()))),
                   "prevalence": prevalence, "args": vars(args), "dead_units": dead,
                   "n_train": len(tr), "n_val": len(va), "n_test": len(te)}, f, indent=2)
    print(f"saved results/cnn_training{suffix}.json and", model_path)


if __name__ == "__main__":
    main()
