"""Pipeline orchestrator: load -> EDA -> split -> fit -> evaluate -> save.

Runs today with the implemented modules (data, baselines, evaluation) and
automatically picks up the personalized models as you implement them — any model
still raising NotImplementedError is skipped with a note.

Usage:
    python main.py
"""

from __future__ import annotations

import pandas as pd

from src import config, data, evaluation
from src.baselines import (
    MostPopularRecommender,
    HighestAverageRatingRecommender,
    RandomRecommender,
)
from src.content_based import ContentBasedRecommender
from src.collaborative import ItemItemCF, UserUserCF
from src.matrix_factorization import FunkSVD


def build_models():
    """Registry of (model, needs_items) — extend as you implement modules."""
    return [
        (MostPopularRecommender(), False),
        (HighestAverageRatingRecommender(min_ratings=20), False),
        (RandomRecommender(), False),
        (ContentBasedRecommender(), True),
        (ItemItemCF(k=20), False),
        (UserUserCF(k=30), False),
        (FunkSVD(n_factors=50, n_epochs=20), False),
    ]


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== Loading data ==")
    ratings = data.load_ratings()
    movies = data.load_movies()
    tags = data.load_tags()
    items = data.build_item_metadata(movies, tags)

    print("\n== EDA ==")
    data.describe_dataset(ratings, items)

    print("\n== Preprocess + split (per-user temporal holdout) ==")
    ratings = data.filter_min_ratings(ratings)
    train, test = data.train_test_split_temporal(ratings)
    relevant = data.build_relevant_items(test)
    catalog = ratings[config.ITEM_COL].unique()
    item_pop = ratings[config.ITEM_COL].value_counts().to_dict()
    n_users = ratings[config.USER_COL].nunique()
    print(f"train={len(train):,}  test={len(test):,}  eval users={len(relevant):,}")

    print("\n== Fit + evaluate ==")
    rows = []
    for model, needs_items in build_models():
        try:
            model.fit(train, items) if needs_items else model.fit(train)
            metrics = evaluation.evaluate_ranking(
                model, relevant, catalog, k=config.TOP_K,
                item_popularity=item_pop, n_users=n_users,
            )
            metrics = {"model": model.name, **metrics}
            rows.append(metrics)
            print(f"[ok]      {model.name}: P@{config.TOP_K}={metrics[f'precision@{config.TOP_K}']:.4f} "
                  f"NDCG={metrics[f'ndcg@{config.TOP_K}']:.4f}")
        except NotImplementedError:
            print(f"[skip]    {model.name}: not implemented yet")

    if rows:
        out = config.RESULTS_DIR / "metrics.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
