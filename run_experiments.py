"""Tuning sweeps, ablations, and popularity-bias analysis.

Separate from main.py: main.py trains each model once at its chosen config;
this script retrains models across hyperparameter grids and produces the
comparison tables + figures for the report (results/*.csv, results/figures/*.png).

Usage:
    python run_experiments.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, data, evaluation, experiments
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender
from src.collaborative import ItemItemCF, UserUserCF
from src.content_based import ContentBasedRecommender
from src.matrix_factorization import FunkSVD

FIG = config.FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  figure -> {path}")


def eda_figures(ratings):
    dist = ratings[config.RATING_COL].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(dist.index.astype(str), dist.values, color="#4472C4")
    plt.xlabel("Rating"); plt.ylabel("Count"); plt.title("Rating distribution")
    savefig("eda_rating_distribution.png")

    per_user = ratings[config.USER_COL].value_counts()
    plt.figure(figsize=(6, 4))
    plt.hist(per_user.values, bins=50, color="#70AD47")
    plt.yscale("log")
    plt.xlabel("Ratings per user"); plt.ylabel("Number of users (log)")
    plt.title("Ratings-per-user distribution")
    savefig("eda_ratings_per_user.png")


def cf_k_sweep_figure(df):
    plt.figure(figsize=(7, 4))
    for name, grp in df.groupby("model"):
        plt.plot(grp["k"], grp[f"precision@{config.TOP_K}"], marker="o", label=name)
    plt.xlabel("k (neighbors)"); plt.ylabel(f"Precision@{config.TOP_K}")
    plt.title("CF: precision vs neighborhood size"); plt.legend()
    savefig("tuning_cf_k_precision.png")

    plt.figure(figsize=(7, 4))
    for name, grp in df.groupby("model"):
        plt.plot(grp["k"], grp[f"coverage@{config.TOP_K}"], marker="o", label=name)
    plt.xlabel("k (neighbors)"); plt.ylabel(f"Catalog coverage@{config.TOP_K}")
    plt.title("CF: coverage vs neighborhood size"); plt.legend()
    savefig("tuning_cf_k_coverage.png")


def mf_sweep_figure(df, x_col, fname, title):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df[x_col], df[f"precision@{config.TOP_K}"], marker="o", color="#4472C4", label="Precision@10")
    ax1.set_xlabel(x_col); ax1.set_ylabel("Precision@10", color="#4472C4")
    ax2 = ax1.twinx()
    ax2.plot(df[x_col], df["train_rmse"], marker="s", color="#C00000", label="Train RMSE")
    ax2.plot(df[x_col], df["test_rmse"], marker="^", color="#ED7D31", label="Test RMSE")
    ax2.set_ylabel("RMSE")
    fig.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    plt.title(title)
    savefig(fname)


def content_ablation_figure(df):
    plt.figure(figsize=(7, 4))
    plt.bar(df["config"], df[f"precision@{config.TOP_K}"], color="#FFC000")
    plt.ylabel(f"Precision@{config.TOP_K}"); plt.title("Content-based: feature ablation")
    plt.xticks(rotation=20, ha="right")
    savefig("ablation_content_features.png")


def popularity_bias_figure(df):
    plt.figure(figsize=(7, 4))
    plt.barh(df["model"], df["mean_popularity_percentile"], color="#8064A2")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Mean popularity percentile of recommended items")
    plt.title("Popularity bias by model (1.0 = most popular item in train)")
    savefig("popularity_bias.png")


def main():
    print("== Load + split ==")
    ratings = data.filter_min_ratings(data.load_ratings())
    movies = data.load_movies()
    items = data.build_item_metadata(movies, data.load_tags())
    train, test = data.train_test_split_temporal(ratings)
    relevant = data.build_relevant_items(test)
    catalog = ratings[config.ITEM_COL].unique()
    item_pop = ratings[config.ITEM_COL].value_counts().to_dict()
    n_users = ratings[config.USER_COL].nunique()

    print("== EDA figures ==")
    eda_figures(ratings)

    print("== CF k-sweep ==")
    cf_df = experiments.sweep_cf_k(train, relevant, catalog, item_pop, n_users)
    cf_df.to_csv(config.RESULTS_DIR / "tuning_cf_k.csv", index=False)
    cf_k_sweep_figure(cf_df)
    print(cf_df[["model", "k", f"precision@{config.TOP_K}", f"coverage@{config.TOP_K}"]].to_string(index=False))

    print("\n== MF n_factors sweep ==")
    mf_factors_df = experiments.sweep_mf_factors(train, test, relevant, catalog, item_pop, n_users)
    mf_factors_df.to_csv(config.RESULTS_DIR / "tuning_mf_factors.csv", index=False)
    mf_sweep_figure(mf_factors_df, "n_factors", "tuning_mf_factors.png", "MF: precision & RMSE vs n_factors")
    print(mf_factors_df.to_string(index=False))

    print("\n== MF reg sweep ==")
    mf_reg_df = experiments.sweep_mf_reg(train, test, relevant, catalog, item_pop, n_users)
    mf_reg_df.to_csv(config.RESULTS_DIR / "tuning_mf_reg.csv", index=False)
    mf_sweep_figure(mf_reg_df, "reg", "tuning_mf_reg.png", "MF: precision & RMSE vs regularization")
    print(mf_reg_df.to_string(index=False))

    print("\n== Content-based feature ablation ==")
    content_df = experiments.ablation_content_features(train, items, relevant, catalog, item_pop, n_users)
    content_df.to_csv(config.RESULTS_DIR / "ablation_content_features.csv", index=False)
    content_ablation_figure(content_df)
    print(content_df.to_string(index=False))

    print("\n== Popularity bias ==")
    fitted = {
        "most_popular": MostPopularRecommender().fit(train),
        "highest_average": HighestAverageRatingRecommender(min_ratings=20).fit(train),
        "random": RandomRecommender().fit(train),
        "content_based": ContentBasedRecommender().fit(train, items),
        "item_item_cf": ItemItemCF(k=20).fit(train),
        "user_user_cf": UserUserCF(k=30).fit(train),
        "matrix_factorization": FunkSVD(n_factors=50, n_epochs=20).fit(train),
    }
    bias_df = experiments.popularity_bias_table(fitted, list(relevant.keys()), item_pop)
    bias_df.to_csv(config.RESULTS_DIR / "popularity_bias.csv", index=False)
    popularity_bias_figure(bias_df)
    print(bias_df.to_string(index=False))

    print("\nAll tuning/ablation results saved under results/ and results/figures/")


if __name__ == "__main__":
    main()
