"""Neighborhood collaborative filtering (from scratch).

Build a sparse user-item matrix (scipy CSR), mean-center each user's ratings to
remove user bias, then compute cosine similarity on the centered matrix:

Item-item:  score(u, i) = sum_{j in topk(i, rated(u))} sim(i, j) * r_uj
                          / sum_{j in topk(i, rated(u))} |sim(i, j)|
            topk(i, rated(u)) = the k items rated by u most similar to candidate i,
            chosen at recommend time. (A global top-k per item, fixed independently
            of the user, was tried first and rejected: with ~9.7K items a user's
            history almost never intersects a small fixed neighbor set in more than
            one item, which collapses the "weighted average" to a single neighbor's
            raw rating. Restricting top-k to the user's own rated items avoids that.)

User-user:  score(u, i) = mean_u + sum_{v in N_k(u), v rated i} sim(u, v) * (r_vi - mean_v)
                          / sum_{v in N_k(u), v rated i} |sim(u, v)|
            N_k(u) = global top-k similar users, fixed at fit time. With only ~600
            users this set is broad enough to reliably overlap with item raters, so
            the global-then-intersect approach (rejected for items, above) is fine here.

Interface:
    fit(train_df) -> self
    recommend(user_id, n=10, exclude_seen=True) -> list[(item_id, score)]
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def _build_user_item_matrix(train):
    """Map raw userId/movieId to contiguous indices and build a CSR ratings matrix."""
    user_ids = np.sort(train[config.USER_COL].unique())
    item_ids = np.sort(train[config.ITEM_COL].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {m: i for i, m in enumerate(item_ids)}

    rows = train[config.USER_COL].map(user_index).to_numpy()
    cols = train[config.ITEM_COL].map(item_index).to_numpy()
    vals = train[config.RATING_COL].to_numpy(dtype=np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return matrix, user_ids, item_ids, user_index, item_index


def _user_means(matrix):
    """Per-user mean rating, computed only over observed (nonzero) entries."""
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    counts = np.diff(matrix.indptr)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _center_by_user(matrix, means):
    """Subtract each user's mean from their observed entries (sparsity preserved)."""
    centered = matrix.copy().astype(np.float32)
    for u in range(matrix.shape[0]):
        start, end = centered.indptr[u], centered.indptr[u + 1]
        centered.data[start:end] -= means[u]
    return centered


def _shrink_similarity(sim, binary_matrix, axis, alpha=config.CF_SHRINKAGE):
    """Significance-weight similarity by co-occurrence support.

    Without this, two items/users that happen to share a single rater can get
    cosine similarity == 1.0, which then dominates scoring (the classic
    sparse-CF artifact). Shrink by sim *= n_common / (n_common + alpha), so
    pairs with little overlap evidence are pulled toward zero.

    axis="items": binary_matrix is users x items -> co_counts = binary.T @ binary (items x items)
    axis="users": binary_matrix is users x items -> co_counts = binary @ binary.T (users x users)
    """
    binary = binary_matrix
    co_counts = (binary.T @ binary) if axis == "items" else (binary @ binary.T)
    co_counts = np.asarray(co_counts.todense(), dtype=np.float32)
    weight = co_counts / (co_counts + alpha)
    return sim * weight


def _topk_similarity(sim, k):
    """Zero out everything but the top-k neighbors per row (excluding self). Square matrix."""
    n = sim.shape[0]
    np.fill_diagonal(sim, 0.0)
    if k >= n - 1:
        return sim
    return _topk_rows(sim, k)


def _topk_rows(matrix, k):
    """Zero out everything but the top-k columns per row. Works on any (rows x cols) matrix."""
    n_rows, n_cols = matrix.shape
    if k >= n_cols:
        return matrix
    pruned = np.zeros_like(matrix)
    idx = np.argpartition(-matrix, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_rows), k)
    pruned[rows, idx.ravel()] = matrix[rows, idx.ravel()]
    return pruned


class ItemItemCF:
    name = "item_item_cf"

    def __init__(self, k=20):
        self.k = k
        self.user_index_ = None
        self.item_ids_ = None
        self.matrix_ = None      # raw user-item CSR (scoring uses actual ratings)
        self.sim_ = None         # item-item similarity, shrunk, dense, full (not pre-pruned)
        self._seen_by_user = {}

    def fit(self, train):
        matrix, user_ids, item_ids, user_index, item_index = _build_user_item_matrix(train)
        centered = _center_by_user(matrix, _user_means(matrix))
        binary = (matrix != 0).astype(np.float32)

        sim = cosine_similarity(centered.T, dense_output=True).astype(np.float32)
        np.fill_diagonal(sim, 0.0)
        self.sim_ = _shrink_similarity(sim, binary, axis="items")

        self.matrix_ = matrix.tocsr()
        self.user_index_ = user_index
        self.item_ids_ = item_ids
        self._seen_by_user = {
            u: set(g[config.ITEM_COL]) for u, g in train.groupby(config.USER_COL)
        }
        return self

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        if user_id not in self.user_index_:
            return []
        u = self.user_index_[user_id]
        user_row = self.matrix_.getrow(u)
        rated_cols, rated_ratings = user_row.indices, user_row.data
        if len(rated_cols) == 0:
            return []

        sim_sub = self.sim_[:, rated_cols]                  # n_items x n_rated
        sim_sub = _topk_rows(sim_sub, self.k)                # keep only the k most similar rated items per candidate
        numer = sim_sub @ rated_ratings
        denom = np.abs(sim_sub).sum(axis=1)
        scores = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)

        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if scores[idx] <= 0:
                break  # argsort descending: once we hit <=0 nothing better remains
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            results.append((int(item_id), float(scores[idx])))
            if len(results) >= n:
                break
        return results


class UserUserCF:
    """User-user neighborhood CF (for the item-item vs user-user comparison)."""

    name = "user_user_cf"

    def __init__(self, k=30):
        self.k = k
        self.user_index_ = None
        self.item_ids_ = None
        self.matrix_ = None        # raw user-item CSR
        self.centered_ = None      # mean-centered CSR (same sparsity as matrix_)
        self.user_means_ = None
        self.sim_ = None           # user-user similarity, top-k pruned, dense
        self._seen_by_user = {}

    def fit(self, train):
        matrix, user_ids, item_ids, user_index, item_index = _build_user_item_matrix(train)
        means = _user_means(matrix)
        centered = _center_by_user(matrix, means)
        binary = (matrix != 0).astype(np.float32)

        sim = cosine_similarity(centered, dense_output=True).astype(np.float32)
        sim = _shrink_similarity(sim, binary, axis="users")
        self.sim_ = _topk_similarity(sim, self.k)

        self.matrix_ = matrix.tocsr()
        self.centered_ = centered
        self.user_means_ = means
        self.user_index_ = user_index
        self.item_ids_ = item_ids
        self._seen_by_user = {
            u: set(g[config.ITEM_COL]) for u, g in train.groupby(config.USER_COL)
        }
        return self

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        if user_id not in self.user_index_:
            return []
        u = self.user_index_[user_id]
        sim_row = self.sim_[u]
        neighbors = np.nonzero(sim_row)[0]
        if len(neighbors) == 0:
            return []
        neighbor_sims = sim_row[neighbors]

        sub_centered = self.centered_[neighbors]                       # n_neighbors x n_items
        sub_mask = (self.matrix_[neighbors] != 0).astype(np.float32)   # who rated what

        numer = sub_centered.T @ neighbor_sims                         # n_items
        denom = sub_mask.T @ np.abs(neighbor_sims)                     # n_items
        scores = self.user_means_[u] + np.divide(
            numer, denom, out=np.zeros_like(numer), where=denom > 0
        )

        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        rated_mask = denom > 0  # only score items at least one neighbor actually rated
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if not rated_mask[idx]:
                continue
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            results.append((int(item_id), float(scores[idx])))
            if len(results) >= n:
                break
        return results
