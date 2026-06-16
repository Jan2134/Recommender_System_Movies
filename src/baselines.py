"""Non-personalized baseline recommenders (fully implemented).

These set the floor that personalized methods must beat, and provide the
"basic evaluation from the beginning" agile increment. All recommenders share
the interface:

    model.fit(train_df, items_df=None) -> self
    model.recommend(user_id, n=10, exclude_seen=True) -> list[(item_id, score)]
"""

from __future__ import annotations

import numpy as np

from . import config


class _BaseRecommender:
    name = "base"

    def __init__(self):
        self._seen_by_user = {}

    def _fit_seen(self, train):
        self._seen_by_user = {
            u: set(g[config.ITEM_COL])
            for u, g in train.groupby(config.USER_COL)
        }
        return self

    def _filter(self, ranked_items, user_id, n, exclude_seen):
        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        out = [(i, s) for i, s in ranked_items if i not in seen]
        return out[:n]


class MostPopularRecommender(_BaseRecommender):
    """Rank items by number of ratings (popularity). Same list for every user."""

    name = "most_popular"

    def fit(self, train, items=None):
        self._fit_seen(train)
        counts = train[config.ITEM_COL].value_counts()
        self.ranking_ = list(zip(counts.index.tolist(), counts.values.tolist()))
        return self

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        return self._filter(self.ranking_, user_id, n, exclude_seen)


class HighestAverageRatingRecommender(_BaseRecommender):
    """Rank by mean rating among items with at least ``min_ratings`` ratings."""

    name = "highest_average"

    def __init__(self, min_ratings=20):
        super().__init__()
        self.min_ratings = min_ratings

    def fit(self, train, items=None):
        self._fit_seen(train)
        grp = train.groupby(config.ITEM_COL)[config.RATING_COL]
        stats = grp.agg(["mean", "count"])
        stats = stats[stats["count"] >= self.min_ratings].sort_values("mean", ascending=False)
        self.ranking_ = list(zip(stats.index.tolist(), stats["mean"].tolist()))
        return self

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        return self._filter(self.ranking_, user_id, n, exclude_seen)


class RandomRecommender(_BaseRecommender):
    """Sample random unseen items. Lower bound / sanity check."""

    name = "random"

    def __init__(self, random_state=config.RANDOM_STATE):
        super().__init__()
        self.rng = np.random.default_rng(random_state)

    def fit(self, train, items=None):
        self._fit_seen(train)
        self.items_ = train[config.ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        candidates = [i for i in self.items_ if i not in seen]
        chosen = self.rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(i), 0.0) for i in chosen]
