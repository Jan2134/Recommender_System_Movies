"""Data loading, preprocessing, EDA, and train/test splitting.

This is the foundation module: everything downstream consumes the DataFrames and
the train/test split produced here. It is fully implemented so the pipeline runs
end-to-end from day one (agile: "basic evaluation from the beginning").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --- Loading ----------------------------------------------------------------
def load_ratings(path=config.RATINGS_PATH) -> pd.DataFrame:
    """Load the ratings file and validate its schema."""
    df = pd.read_csv(path)
    required = {config.USER_COL, config.ITEM_COL, config.RATING_COL, config.TIMESTAMP_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ratings file missing columns: {missing}")
    return df


def load_movies(path=config.MOVIES_PATH) -> pd.DataFrame:
    """Load movie metadata (movieId, title, genres)."""
    df = pd.read_csv(path)
    required = {config.ITEM_COL, config.TITLE_COL, config.GENRES_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"movies file missing columns: {missing}")
    return df


def load_tags(path=config.TAGS_PATH) -> pd.DataFrame:
    """Load free-text tags (used to enrich content-based features)."""
    return pd.read_csv(path)


def build_item_metadata(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a per-item metadata frame with a combined text field for content-based.

    Genres are pipe-separated in MovieLens; we turn them into space-separated tokens
    and optionally append the most relevant free-text tags per movie.
    """
    items = movies.copy()
    items["genres_clean"] = (
        items[config.GENRES_COL]
        .replace("(no genres listed)", "", regex=False)
        .str.replace("|", " ", regex=False)
        .str.lower()
    )

    if tags is not None and len(tags):
        agg = (
            tags.dropna(subset=[config.TAG_COL])
            .assign(tag=lambda d: d[config.TAG_COL].astype(str).str.lower())
            .groupby(config.ITEM_COL)["tag"]
            .apply(lambda s: " ".join(s))
            .rename("tags_text")
        )
        items = items.merge(agg, on=config.ITEM_COL, how="left")
    else:
        items["tags_text"] = ""

    items["tags_text"] = items["tags_text"].fillna("")
    items["content"] = (items["genres_clean"] + " " + items["tags_text"]).str.strip()
    return items


# --- EDA --------------------------------------------------------------------
def describe_dataset(ratings: pd.DataFrame, items: pd.DataFrame | None = None,
                     verbose: bool = True) -> dict:
    """Compute the EDA statistics required by the assignment and return them as a dict."""
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_items)

    rating_dist = ratings[config.RATING_COL].value_counts().sort_index()
    most_active_users = ratings[config.USER_COL].value_counts().head(10)
    most_popular_items = ratings[config.ITEM_COL].value_counts().head(10)

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "density": 1 - sparsity,
        "mean_rating": ratings[config.RATING_COL].mean(),
        "ratings_per_user_median": ratings[config.USER_COL].value_counts().median(),
        "ratings_per_item_median": ratings[config.ITEM_COL].value_counts().median(),
        "rating_distribution": rating_dist,
        "most_active_users": most_active_users,
        "most_popular_items": most_popular_items,
    }

    if verbose:
        print(f"Users:            {n_users:,}")
        print(f"Items:            {n_items:,}")
        print(f"Ratings:          {n_ratings:,}")
        print(f"Sparsity:         {sparsity:.4%}")
        print(f"Mean rating:      {stats['mean_rating']:.3f}")
        print(f"Median ratings/user: {stats['ratings_per_user_median']:.0f}")
        print("Rating distribution:")
        print(rating_dist.to_string())

    return stats


# --- Splitting --------------------------------------------------------------
def filter_min_ratings(ratings: pd.DataFrame, min_user=config.MIN_RATINGS_PER_USER,
                       min_item=config.MIN_RATINGS_PER_ITEM) -> pd.DataFrame:
    """Drop users/items with too few ratings so every test user has history."""
    df = ratings
    item_counts = df[config.ITEM_COL].value_counts()
    df = df[df[config.ITEM_COL].isin(item_counts[item_counts >= min_item].index)]
    user_counts = df[config.USER_COL].value_counts()
    df = df[df[config.USER_COL].isin(user_counts[user_counts >= min_user].index)]
    return df.reset_index(drop=True)


def train_test_split_temporal(ratings: pd.DataFrame, test_fraction=config.TEST_FRACTION):
    """Per-user temporal holdout: each user's most recent ``test_fraction`` ratings go to test.

    This is more realistic than a random split (you predict the future from the past)
    and guarantees every test user is also present in train.
    """
    ratings = ratings.sort_values([config.USER_COL, config.TIMESTAMP_COL])
    train_parts, test_parts = [], []
    for _, grp in ratings.groupby(config.USER_COL, sort=False):
        n_test = max(1, int(round(len(grp) * test_fraction)))
        test_parts.append(grp.iloc[-n_test:])
        train_parts.append(grp.iloc[:-n_test])
    train = pd.concat(train_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def train_test_split_random(ratings: pd.DataFrame, test_fraction=config.TEST_FRACTION,
                            random_state=config.RANDOM_STATE):
    """Simple random holdout (kept as a comparison/ablation against the temporal split)."""
    test = ratings.sample(frac=test_fraction, random_state=random_state)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings: pd.DataFrame, user_id) -> set:
    """Set of item IDs a user has already rated (to exclude from recommendations)."""
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])


def build_relevant_items(test: pd.DataFrame, threshold=config.RELEVANCE_THRESHOLD) -> dict:
    """Map each user -> set of relevant test items (rating >= threshold). Ranking ground truth."""
    rel = test[test[config.RATING_COL] >= threshold]
    return {u: set(g[config.ITEM_COL]) for u, g in rel.groupby(config.USER_COL)}
