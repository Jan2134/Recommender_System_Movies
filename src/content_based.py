"""Content-based recommender (from scratch).

1. Build an item-feature matrix from metadata (genres + free-text tags) with TF-IDF
   (or raw counts, as an ablation — see ``use_tfidf``).
2. Build a user profile as the centered-rating-weighted sum of the vectors of items
   the user rated:   profile(u) = sum_i (r_ui - mean_u) * v_i
   Centering matters: an uncentered profile is dominated by genres the user rates a
   lot (even mediocrely), since pure rating-weighted sums can't express "dislike".
   Centering lets low ratings push the profile *away* from those features.
3. Score unseen items by cosine similarity between the profile and item vectors.

Interface:
    fit(train_df, items_df) -> self
    recommend(user_id, n=10, exclude_seen=True) -> list[(item_id, score)]
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


class ContentBasedRecommender:
    name = "content_based"

    def __init__(self, feature_col="content", use_tfidf=True):
        self.feature_col = feature_col   # "content" = genres + tags, see data.build_item_metadata
        self.use_tfidf = use_tfidf       # ablation hook: TF-IDF vs raw counts
        self.vectorizer = None
        self.item_matrix_ = None         # (n_items x n_features), sparse
        self.item_ids_ = None            # row index -> movieId
        self.item_index_ = None          # movieId -> row index
        self.user_means_ = None
        self.global_mean_ = None
        self._train = None
        self._seen_by_user = {}

    def fit(self, train, items):
        text = items[self.feature_col].fillna("")
        self.vectorizer = TfidfVectorizer() if self.use_tfidf else CountVectorizer()
        self.item_matrix_ = self.vectorizer.fit_transform(text)

        self.item_ids_ = items[config.ITEM_COL].to_numpy()
        self.item_index_ = {item_id: i for i, item_id in enumerate(self.item_ids_)}

        self._train = train
        self.user_means_ = train.groupby(config.USER_COL)[config.RATING_COL].mean()
        self.global_mean_ = train[config.RATING_COL].mean()
        self._seen_by_user = {
            u: set(g[config.ITEM_COL]) for u, g in train.groupby(config.USER_COL)
        }
        return self

    def build_user_profile(self, user_id) -> np.ndarray:
        sub = self._train[self._train[config.USER_COL] == user_id]
        mean_u = self.user_means_.get(user_id, self.global_mean_)

        rows, weights = [], []
        for item_id, rating in zip(sub[config.ITEM_COL], sub[config.RATING_COL]):
            idx = self.item_index_.get(item_id)
            if idx is not None:
                rows.append(idx)
                weights.append(rating - mean_u)

        n_features = self.item_matrix_.shape[1]
        if not rows:
            return np.zeros(n_features)

        weights = np.asarray(weights)
        profile = self.item_matrix_[rows].T @ weights  # (n_features,)
        return np.asarray(profile).ravel()

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        profile = self.build_user_profile(user_id)
        if not np.any(profile):
            return []

        scores = cosine_similarity(profile.reshape(1, -1), self.item_matrix_).ravel()

        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if scores[idx] <= 0:
                break
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            results.append((int(item_id), float(scores[idx])))
            if len(results) >= n:
                break
        return results

    def similar_items(self, item_id, n=config.TOP_K):
        """Item-to-item content similarity ("more like this")."""
        idx = self.item_index_.get(item_id)
        if idx is None:
            return []
        scores = cosine_similarity(self.item_matrix_[idx], self.item_matrix_).ravel()
        scores[idx] = -1  # exclude itself
        order = np.argsort(-scores)[:n]
        return [(int(self.item_ids_[i]), float(scores[i])) for i in order if scores[i] > 0]
