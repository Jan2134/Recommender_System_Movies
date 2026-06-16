"""Matrix factorization via SGD — FunkSVD with biases (from scratch).

Model
-----
    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

Learn p_u, q_i (latent factors) and biases b_u, b_i by minimizing regularized
squared error over observed ratings with stochastic gradient descent:

    e_ui = r_ui - r_hat(u, i)
    b_u += lr * (e_ui - reg * b_u)
    b_i += lr * (e_ui - reg * b_i)
    p_u += lr * (e_ui * q_i - reg * p_u)
    q_i += lr * (e_ui * p_u - reg * q_i)

Implementing this from scratch (rather than scikit-surprise) is what the professor
rewards. scikit-surprise can stay as a sanity-check baseline if needed.

Interface:
    fit(train_df) -> self
    predict(user_id, item_id) -> float
    recommend(user_id, n=10, exclude_seen=True) -> list[(item_id, score)]
"""

from __future__ import annotations

import numpy as np

from . import config


class FunkSVD:
    name = "matrix_factorization"

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02,
                 random_state=config.RANDOM_STATE, verbose=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose
        self.mu_ = None
        self.b_u_ = None
        self.b_i_ = None
        self.P_ = None
        self.Q_ = None
        self.user_index_ = None
        self.item_index_ = None
        self.item_ids_ = None
        self.train_rmse_ = []
        self._seen_by_user = {}

    def fit(self, train):
        rng = np.random.default_rng(self.random_state)

        user_ids = np.sort(train[config.USER_COL].unique())
        item_ids = np.sort(train[config.ITEM_COL].unique())
        self.user_index_ = {u: i for i, u in enumerate(user_ids)}
        self.item_index_ = {m: i for i, m in enumerate(item_ids)}
        self.item_ids_ = item_ids
        n_users, n_items = len(user_ids), len(item_ids)

        u_idx = train[config.USER_COL].map(self.user_index_).to_numpy()
        i_idx = train[config.ITEM_COL].map(self.item_index_).to_numpy()
        ratings = train[config.RATING_COL].to_numpy(dtype=np.float64)

        self.mu_ = ratings.mean()
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        scale = 0.1
        self.P_ = rng.normal(0, scale, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0, scale, size=(n_items, self.n_factors))

        n_obs = len(ratings)
        order = np.arange(n_obs)
        self.train_rmse_ = []

        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sq_err_sum = 0.0
            for k in order:
                u, i, r = u_idx[k], i_idx[k], ratings[k]
                pred = self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i]
                err = r - pred
                sq_err_sum += err * err

                p_u, q_i = self.P_[u].copy(), self.Q_[i].copy()
                self.b_u_[u] += self.lr * (err - self.reg * self.b_u_[u])
                self.b_i_[i] += self.lr * (err - self.reg * self.b_i_[i])
                self.P_[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q_[i] += self.lr * (err * p_u - self.reg * q_i)

            rmse = np.sqrt(sq_err_sum / n_obs)
            self.train_rmse_.append(rmse)
            if self.verbose:
                print(f"  epoch {epoch + 1}/{self.n_epochs}  train RMSE={rmse:.4f}")

        self._seen_by_user = {
            u: set(g[config.ITEM_COL]) for u, g in train.groupby(config.USER_COL)
        }
        return self

    def predict(self, user_id, item_id) -> float:
        u = self.user_index_.get(user_id)
        i = self.item_index_.get(item_id)
        if u is None or i is None:
            return float(self.mu_)
        return float(self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i])

    def recommend(self, user_id, n=config.TOP_K, exclude_seen=True):
        u = self.user_index_.get(user_id)
        if u is None:
            return []

        scores = self.mu_ + self.b_u_[u] + self.b_i_ + self.Q_ @ self.P_[u]

        seen = self._seen_by_user.get(user_id, set()) if exclude_seen else set()
        order = np.argsort(-scores)
        results = []
        for idx in order:
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            results.append((int(item_id), float(scores[idx])))
            if len(results) >= n:
                break
        return results
