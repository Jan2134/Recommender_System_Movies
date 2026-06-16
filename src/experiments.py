"""Tuning sweeps, ablations, and popularity-bias analysis.

Separate from main.py (which trains the "production" config of each model once)
because these functions retrain a model many times over a hyperparameter grid.
Each function returns a tidy DataFrame; ``run_experiments.py`` drives them and
saves CSVs + figures to results/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, evaluation
from .collaborative import ItemItemCF, UserUserCF
from .content_based import ContentBasedRecommender
from .matrix_factorization import FunkSVD


def _eval(model, relevant, catalog, item_pop, n_users, k=config.TOP_K):
    return evaluation.evaluate_ranking(
        model, relevant, catalog, k=k, item_popularity=item_pop, n_users=n_users
    )


def sweep_cf_k(train, relevant, catalog, item_pop, n_users, ks=(5, 10, 20, 30, 50, 80)):
    """Precision/coverage tradeoff as the neighborhood size k grows, for both CF variants."""
    rows = []
    for k in ks:
        for name, cls in [("item_item_cf", ItemItemCF), ("user_user_cf", UserUserCF)]:
            model = cls(k=k).fit(train)
            metrics = _eval(model, relevant, catalog, item_pop, n_users)
            rows.append({"model": name, "k": k, **metrics})
    return pd.DataFrame(rows)


def _test_rating_metrics(model, test):
    """Held-out rating-prediction RMSE/MAE — separate from train RMSE, which can't reveal overfitting."""
    preds = [model.predict(u, i) for u, i in zip(test[config.USER_COL], test[config.ITEM_COL])]
    truths = test[config.RATING_COL].to_numpy()
    return {"test_rmse": evaluation.rmse(preds, truths), "test_mae": evaluation.mae(preds, truths)}


def sweep_mf_factors(train, test, relevant, catalog, item_pop, n_users,
                     n_factors_grid=(10, 20, 50, 100), n_epochs=15):
    rows = []
    for n_factors in n_factors_grid:
        model = FunkSVD(n_factors=n_factors, n_epochs=n_epochs).fit(train)
        metrics = _eval(model, relevant, catalog, item_pop, n_users)
        rating_metrics = _test_rating_metrics(model, test)
        rows.append({
            "n_factors": n_factors, "train_rmse": model.train_rmse_[-1],
            **rating_metrics, **metrics,
        })
    return pd.DataFrame(rows)


def sweep_mf_reg(train, test, relevant, catalog, item_pop, n_users,
                 reg_grid=(0.005, 0.02, 0.05, 0.1), n_factors=50, n_epochs=15):
    rows = []
    for reg in reg_grid:
        model = FunkSVD(n_factors=n_factors, n_epochs=n_epochs, reg=reg).fit(train)
        metrics = _eval(model, relevant, catalog, item_pop, n_users)
        rating_metrics = _test_rating_metrics(model, test)
        rows.append({
            "reg": reg, "train_rmse": model.train_rmse_[-1],
            **rating_metrics, **metrics,
        })
    return pd.DataFrame(rows)


def ablation_content_features(train, items, relevant, catalog, item_pop, n_users):
    """TF-IDF vs raw counts; genres-only vs genres+tags."""
    rows = []
    items_genres_only = items.copy()
    items_genres_only["content"] = items_genres_only["genres_clean"]

    configs = [
        ("tfidf_genres_tags", True, items),
        ("raw_counts_genres_tags", False, items),
        ("tfidf_genres_only", True, items_genres_only),
    ]
    for name, use_tfidf, item_frame in configs:
        model = ContentBasedRecommender(use_tfidf=use_tfidf).fit(train, item_frame)
        metrics = _eval(model, relevant, catalog, item_pop, n_users)
        rows.append({"config": name, **metrics})
    return pd.DataFrame(rows)


def popularity_bias_table(fitted_models: dict, relevant_users, item_pop, k=config.TOP_K):
    """For each model, the mean training-set popularity rank of its recommended items.

    Lower mean popularity-percentile == the model leans on obscure items;
    higher == it leans on blockbusters. Compared against the catalog's own
    popularity distribution to see which models amplify vs counteract it.
    """
    pop_series = pd.Series(item_pop)
    pop_percentile = pop_series.rank(pct=True)  # 1.0 = most popular item in train

    rows = []
    for name, model in fitted_models.items():
        percentiles = []
        for user_id in relevant_users:
            recs = model.recommend(user_id, n=k, exclude_seen=True)
            for item_id, _ in recs:
                if item_id in pop_percentile.index:
                    percentiles.append(pop_percentile.loc[item_id])
        rows.append({
            "model": name,
            "mean_popularity_percentile": float(np.mean(percentiles)) if percentiles else np.nan,
            "n_recommendations": len(percentiles),
        })
    return pd.DataFrame(rows).sort_values("mean_popularity_percentile", ascending=False)
