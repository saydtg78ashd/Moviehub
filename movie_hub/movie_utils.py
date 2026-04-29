from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "ml-latest-small"
MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"


@dataclass
class MovieData:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    movie_stats: pd.DataFrame
    merged: pd.DataFrame
    genre_matrix: pd.DataFrame
    similarity: np.ndarray
    index_to_title: list[str]


def _ensure_data_exists() -> None:
    missing = [str(p) for p in [MOVIES_CSV, RATINGS_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MovieLens data files: " + ", ".join(missing) + "\nRun: python download_data.py"
        )


def load_movie_data() -> MovieData:
    _ensure_data_exists()
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)

    movie_stats = (
        ratings.groupby("movieId")
        .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"))
        .reset_index()
    )

    merged = movies.merge(movie_stats, on="movieId", how="left")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype(int)
    merged["rating_mean"] = merged["rating_mean"].fillna(0).round(2)
    merged["year"] = merged["title"].str.extract(r"\((\d{4})\)$").astype(float)

    genre_matrix = build_genre_matrix(merged)
    similarity = cosine_similarity(genre_matrix.values)
    index_to_title = merged["title"].tolist()

    return MovieData(movies, ratings, movie_stats, merged, genre_matrix, similarity, index_to_title)

def _find_title_index(df: pd.DataFrame, title: str) -> int:
    matches = df.index[df["title"] == title].tolist()
    if matches:
        return matches[0]
    lowered = title.lower()
    candidates = df.index[df["title"].str.lower().str.contains(lowered, na=False)].tolist()
    if not candidates:
        raise ValueError(f"Could not find movie title: {title}")
    return candidates[0]

def build_genre_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # ===== 1. TEXT (genre) =====
    text = df["genres"].fillna("Unknown").str.replace("|", " ", regex=False)
    tfidf = TfidfVectorizer()
    text_matrix = tfidf.fit_transform(text).toarray()
    # ===== 2. NUMERIC (year + rating) =====
    numeric = df[["year", "rating_mean", "rating_count"]].fillna(0)
    scaler = MinMaxScaler()
    numeric_scaled = scaler.fit_transform(numeric)
    # ===== 3. COMBINE =====
    import numpy as np
    final_matrix = np.hstack([text_matrix, numeric_scaled])
    return pd.DataFrame(final_matrix, index=df.index)

def recommend_by_title(df: pd.DataFrame, similarity: np.ndarray, title: str, n: int = 10) -> pd.DataFrame:
    idx = _find_title_index(df, title)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    movie_indices = [i for i, _ in sim_scores]
    recs = df.iloc[movie_indices][["movieId", "title", "genres", "rating_count", "rating_mean", "year"]].copy()
    recs["similarity"] = [round(float(score), 3) for _, score in sim_scores]
    return recs.reset_index(drop=True)


def recommend_by_preferences(df: pd.DataFrame, genre: str, min_votes: int = 20, min_rating: float = 3.5, n: int = 10) -> pd.DataFrame:
    filtered = df.copy()
    if genre != "All":
        filtered = filtered[filtered["genres"].str.contains(genre, case=False, na=False)]
    filtered = filtered[(filtered["rating_count"] >= min_votes) & (filtered["rating_mean"] >= min_rating)].copy()
    filtered = filtered.sort_values(by=["rating_mean", "rating_count", "title"], ascending=[False, False, True]).head(n)
    return filtered[["movieId", "title", "genres", "rating_count", "rating_mean", "year"]].reset_index(drop=True)


def top_by_rating(df: pd.DataFrame, n: int = 10, min_votes: int = 50) -> pd.DataFrame:
    filtered = df[df["rating_count"] >= min_votes].copy()
    filtered = filtered.sort_values(by=["rating_mean", "rating_count"], ascending=[False, False]).head(n)
    return filtered[["movieId", "title", "genres", "rating_count", "rating_mean", "year"]].reset_index(drop=True)


def genre_breakdown(df: pd.DataFrame) -> pd.Series:
    return df["genres"].fillna("Unknown").str.get_dummies(sep="|").sum().sort_values(ascending=False)


def decade_breakdown(df: pd.DataFrame) -> pd.Series:
    years = df["year"].dropna().astype(int)
    if years.empty:
        return pd.Series(dtype=int)
    decades = (years // 10) * 10
    return decades.value_counts().sort_index()


def user_activity_summary(ratings: pd.DataFrame) -> dict[str, float]:
    ratings_per_user = ratings.groupby("userId")["rating"].count()
    return {
        "users": float(ratings["userId"].nunique()),
        "ratings": float(len(ratings)),
        "avg_ratings_per_user": float(ratings_per_user.mean()),
        "median_ratings_per_user": float(ratings_per_user.median()),
    }