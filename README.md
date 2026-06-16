# Movie Recommender System

Individual project for **Recommender Systems** (ESADE, Prof. Marc Torrens).

A movie recommender built **from scratch** (no recommender libraries — every model,
similarity computation, and SGD update loop is hand-written) on the
[MovieLens latest-small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
dataset (610 users, 9,742 movies, 100,836 ratings). Seven recommenders are compared
under a single offline evaluation protocol, with a Streamlit app for interactive,
qualitative exploration.

**Live demo:** _add your Streamlit Cloud URL here once deployed_

## Models implemented

| Family | Models |
|---|---|
| Non-personalized | Most Popular, Highest Average Rating (min-ratings threshold), Random |
| Content-based | TF-IDF over genres + tags, cosine similarity against a centered-rating user profile |
| Collaborative filtering | Item-item and user-user neighborhood CF, with significance-weighted ("shrunk") cosine similarity |
| Matrix factorization | FunkSVD (`r̂ = μ + b_u + b_i + pᵤ·qᵢ`), trained by hand-rolled SGD |

## Evaluation protocol

- **Split**: per-user temporal holdout — each user's most recent 20% of ratings
  go to test, the rest to train. Realistic (predict the future from the past)
  and guarantees every evaluated user has training history.
- **Relevance**: a test rating ≥ 4.0 counts as "relevant" for ranking metrics.
- **Ranking metrics**: Precision/Recall/NDCG/MAP/MRR/Hit Rate @10.
- **Beyond-accuracy**: catalog coverage, novelty, intra-list diversity, and a
  **popularity-bias analysis** (how strongly each model skews toward blockbusters).
- **Rating-prediction metrics**: RMSE/MAE, evaluated on held-out ratings (used to
  catch matrix-factorization overfitting that train-only RMSE would hide — see
  the n_factors/regularization sweep in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)).

### Headline findings

- **Popularity baselines win on raw accuracy** (`most_popular` P@10 ≈ 0.058) —
  expected in offline holdout evaluation, since future interactions skew toward
  broadly-liked items. FunkSVD is the strongest *personalized* model (P@10 ≈ 0.033).
- **Item-item CF shows negative popularity bias** (mean popularity-percentile of
  its recommendations: 0.27, vs. 0.99 for `most_popular` and 0.87 for FunkSVD) —
  it systematically surfaces below-average-popularity items, which the
  accuracy-only metrics above don't reward but a real product might.
- **FunkSVD overfits past `n_factors≈20` / under-regularized settings**: train
  RMSE keeps falling with more capacity, but held-out test RMSE stays flat —
  only visible once test-set RMSE/MAE were tracked separately from train RMSE.

Full write-up of all tuning/ablation results: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

`data/raw/ml-latest-small/` is committed directly (it's ~3MB) so the project and
the Streamlit Cloud deployment work with no separate download step. To refresh it:

```bash
curl -L -o /tmp/ml.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -o /tmp/ml.zip -d data/raw/
```

> Source: F. M. Harper & J. A. Konstan (2015), *The MovieLens Datasets: History and Context*,
> ACM TiiS. Used under the GroupLens research license. The full ml-32m dump is kept
> locally for reference but is not used — it's too large for the from-scratch,
> non-vectorized CF/SGD code in this repo to handle interactively.

## Run

```bash
python main.py                      # full pipeline -> results/metrics.csv
python run_experiments.py           # hyperparameter sweeps, ablations, popularity-bias analysis -> results/*.csv, results/figures/*.png
streamlit run app/streamlit_app.py  # interactive prototype
```

## Deploying the Streamlit app

This repo is structured to deploy on [Streamlit Community Cloud](https://streamlit.io/cloud)
with no extra configuration:

1. Push this repo to GitHub (already public/private as needed).
2. On share.streamlit.io, create a new app pointing at this repo, branch `main`,
   main file path `app/streamlit_app.py`.
3. Streamlit Cloud installs `requirements.txt` and clones the repo as-is — since
   `data/raw/ml-latest-small/` is committed (not gitignored), the app has everything
   it needs at deploy time, with no download or setup step.

## Layout

```
src/        from-scratch implementations
  config.py                 paths, column names, defaults
  data.py                   loading, EDA, preprocessing, train/test split
  evaluation.py             ranking + beyond-accuracy + rating metrics
  baselines.py               most-popular / highest-average / random
  collaborative.py          item-item & user-user neighborhood CF
  content_based.py          TF-IDF item vectors + user profiles
  matrix_factorization.py   FunkSVD (SGD) with biases
  experiments.py            hyperparameter sweeps, ablations, popularity-bias analysis
app/        streamlit_app.py — interactive demo UI
data/raw/   MovieLens latest-small (committed)
notebooks/  exploratory + reporting notebooks
results/    metrics.csv, tuning/ablation CSVs, figures (regenerated by main.py / run_experiments.py)
reference/  course PDF + professor's placeholder template (gitignored, not used)
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the week-by-week build log,
the grading-criteria map, and the full results discussion.
