"""Streamlit prototype UI (targets the 20% User Experience grade).

Lets a grader pick a user and an algorithm, then shows the user's taste profile
(recently liked movies) next to the top-N recommendations with scores, so the
behaviour of each method is visible and comparable.

Run:
    streamlit run app/streamlit_app.py

Status: SCAFFOLD — the layout/caching is set up; wire in models as you implement
them. It already works end-to-end with the baseline recommenders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config, data
from src.baselines import (
    MostPopularRecommender,
    HighestAverageRatingRecommender,
    RandomRecommender,
)
from src.collaborative import ItemItemCF, UserUserCF
from src.content_based import ContentBasedRecommender
from src.matrix_factorization import FunkSVD

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load():
    ratings = data.filter_min_ratings(data.load_ratings())
    movies = data.load_movies()
    items = data.build_item_metadata(movies, data.load_tags())
    train, _ = data.train_test_split_temporal(ratings)
    return ratings, movies, items, train


@st.cache_resource
def get_models(_train, _items):
    models = {
        "Most Popular": MostPopularRecommender().fit(_train),
        "Highest Average (min 20)": HighestAverageRatingRecommender(min_ratings=20).fit(_train),
        "Random": RandomRecommender().fit(_train),
        "Item-Item CF": ItemItemCF(k=20).fit(_train),
        "User-User CF": UserUserCF(k=30).fit(_train),
        "Content-Based": ContentBasedRecommender().fit(_train, _items),
        "Matrix Factorization": FunkSVD(n_factors=50, n_epochs=20).fit(_train),
    }
    return models


def title_for(movies, item_id):
    row = movies.loc[movies[config.ITEM_COL] == item_id, config.TITLE_COL]
    return row.iloc[0] if len(row) else str(item_id)


def main():
    st.title("🎬 Movie Recommender Prototype")
    ratings, movies, items, train = load()
    models = get_models(train, items)

    with st.sidebar:
        st.header("Controls")
        user_id = st.selectbox("User", sorted(ratings[config.USER_COL].unique()))
        algo = st.selectbox("Algorithm", list(models.keys()))
        n = st.slider("Number of recommendations", 5, 20, config.TOP_K)

    left, right = st.columns(2)

    with left:
        st.subheader("What this user already liked")
        liked = (
            train[(train[config.USER_COL] == user_id) & (train[config.RATING_COL] >= 4)]
            .sort_values(config.RATING_COL, ascending=False)
            .head(10)
        )
        liked = liked.merge(movies, on=config.ITEM_COL)
        st.dataframe(liked[[config.TITLE_COL, config.RATING_COL, config.GENRES_COL]],
                     hide_index=True, use_container_width=True)

    with right:
        st.subheader(f"Top {n} — {algo}")
        recs = models[algo].recommend(user_id, n=n, exclude_seen=True)
        rec_df = pd.DataFrame(
            [{"title": title_for(movies, i), "score": round(float(s), 3)} for i, s in recs]
        )
        st.dataframe(rec_df, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
