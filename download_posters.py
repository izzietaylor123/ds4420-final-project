import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from cnn_data import DATA, load_ratings

POSTERS = DATA / "posters"
API = "https://api.themoviedb.org/3/movie/{}"
IMG = "https://image.tmdb.org/t/p/{}{}"


def load_dotenv(path=Path(__file__).parent / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def make_session(key):
    s = requests.Session()
    if len(key) > 40:
        s.headers["Authorization"] = f"Bearer {key}"
    else:             
        s.params = {"api_key": key}
    return s

# Download one poster - returns movie_id & status
def fetch_one(session, movie_id, tmdb_id, size, retries=4):
    out = POSTERS / f"{movie_id}.jpg"
    if out.exists():
        return movie_id, "exists"
    for attempt in range(retries):
        try:
            r = session.get(API.format(tmdb_id), timeout=15)
            if r.status_code == 429:  # hit rate limit
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return movie_id, "not_on_tmdb"
            r.raise_for_status()
            path = r.json().get("poster_path")
            if not path:
                return movie_id, "no_poster"
            img = requests.get(IMG.format(size, path), timeout=30)
            img.raise_for_status()
            out.write_bytes(img.content)
            return movie_id, "ok"
        except requests.RequestException as e:
            if attempt == retries - 1:
                return movie_id, f"error: {e.__class__.__name__}"
            time.sleep(2 ** attempt)
    return movie_id, "rate_limited"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all-movies", action="store_true", help="download for all movies, not just the sample")
    p.add_argument("--sample-users", type=int, default=20000)
    p.add_argument("--min-movie-ratings", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="w185", help="TMDb image size (w92, w154, w185, w342, ...)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    load_dotenv()
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        raise SystemExit("Set TMDB_API_KEY in .env first (see the docstring at the top of this file).")

    links = pd.read_csv(DATA / "link.csv").dropna(subset=["tmdbId"])
    if not args.all_movies:
        movie_ids = load_ratings(args.sample_users, args.min_movie_ratings, args.seed)["movie_ids"]
        links = links[links["movieId"].isin(movie_ids)]
    POSTERS.mkdir(parents=True, exist_ok=True)
    print(f"{len(links):,} movies to fetch -> {POSTERS}")

    session = make_session(key)
    results = []
    with ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(fetch_one, session, int(m), int(t), args.size)
                for m, t in zip(links["movieId"], links["tmdbId"])]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 500 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,}")

    res = pd.DataFrame(results, columns=["movieId", "status"])
    print(res["status"].value_counts().to_string())
    res[~res["status"].isin(["ok", "exists"])].to_csv(POSTERS / "missing.csv", index=False)


if __name__ == "__main__":
    main()
