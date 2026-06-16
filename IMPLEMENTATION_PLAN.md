# Implementation Plan — Movie Recommender (from scratch)

ESADE Recommender Systems · Individual Project · Prof. Marc Torrens

## 0. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Track | **Movies — MovieLens latest-small** (~100K ratings, 610 users, 9.7K movies) | Professor's recommended "main" small dataset; light enough to run from-scratch CF/MF and tune quickly. ml-32m kept locally but not used (too heavy for hand-written dense ops). |
| Code structure | **Flat repo** (`src/`, `app/`, `notebooks/`, `data/`, `results/`) | Becomes a clean GitHub repo. |
| Implementation | **From scratch** (own design, not filling the professor's TODOs) | Professor said this scores higher. scikit-surprise kept only as an optional cross-check. |
| UI | **Streamlit app** | Targets the 20% User Experience grade. |
| Eval protocol | **Per-user temporal holdout** (last 20% of each user's ratings), relevance = rating ≥ 4 | Realistic (predict future from past); guarantees every test user has training history. |

## Grading map (what each % needs)

- **Technical implementations — 50%** → §3–§6: baselines, content-based, item-item + user-user CF, FunkSVD MF, all hand-written and explained.
- **Evaluation — 30%** → §7: Precision/Recall/NDCG/MAP/MRR@K, coverage + novelty + diversity, RMSE/MAE, plus tuning and ablations.
- **User Experience — 20%** → §8: Streamlit prototype + recommendation examples + the slide deck.

## Agile principle

Keep a runnable pipeline at all times (`python main.py`). `main.py` already runs the
baselines and skips not-yet-implemented models, so every increment is measurable
immediately ("basic evaluation from the beginning").

---

## Status

- [x] **Foundation** — repo layout, data download, config
- [x] **Data + EDA + split** (`src/data.py`)
- [x] **Evaluation metrics** (`src/evaluation.py`)
- [x] **Non-personalized baselines** (`src/baselines.py`)
- [x] **Orchestrator + Streamlit shell** (`main.py`, `app/streamlit_app.py`)
- [x] **Collaborative filtering** (`src/collaborative.py`) — item-item + user-user, wired into `main.py` and the Streamlit app
- [x] **Content-based** (`src/content_based.py`) — TF-IDF over genres+tags, centered user profiles, wired in
- [x] **Matrix factorization** (`src/matrix_factorization.py`) — FunkSVD/SGD with biases, wired in
- [x] **Tuning, ablations, beyond-accuracy analysis** (`src/experiments.py`, `run_experiments.py`) — CF k-sweep, MF n_factors/reg sweep with held-out test RMSE/MAE, content-feature ablation, popularity-bias table
- [ ] Notebooks + figures + slide deck *(next)*

---

## Week-by-week

### Week 1 — Setup + data + EDA *(done)*
- Repo, venv, dataset download, `config.py`.
- `data.py`: load ratings/movies/tags, build item metadata (genres + tags → `content`),
  `describe_dataset` (users, items, ratings, sparsity, rating distribution, most active
  users, most popular items), filtering, temporal & random splits.
- Deliverable: `notebooks/01_eda.ipynb` with charts saved to `results/figures/`.

### Week 2 — Non-personalized + evaluation harness *(done)*
- `baselines.py`: MostPopular, HighestAverage (min ratings), Random.
- `evaluation.py`: ranking + beyond-accuracy + rating metrics, `evaluate_ranking` loop.
- `main.py` produces `results/metrics.csv`. Baselines are the floor to beat.

### Week 3 — Collaborative filtering *(done)*
- `collaborative.py`: sparse user-item matrix, mean-centering, cosine similarity.
- Implemented **item-item** and **user-user**, both with significance-weighted
  ("shrunk") similarity: `sim *= n_common / (n_common + alpha)` — without this,
  item/user pairs that share a single rater get cosine similarity 1.0 and
  dominate scoring, a classic sparse-CF artifact (caught via qualitative
  inspection: every recommendation scored exactly 5.0 before the fix).
- Item-item neighbors are selected **dynamically per recommendation**, restricted
  to the target user's rated items, rather than precomputed globally. With
  ~9.7K items, a small fixed global top-k almost never overlaps a user's history
  in more than one item, which collapses the weighted average to a single
  neighbor's raw rating. User-user keeps a global top-k (only ~600 users, so the
  same failure mode doesn't apply).
- Added `config.MIN_RATINGS_PER_ITEM` (drop items with <5 ratings before
  building the similarity matrix) as a complementary noise reduction.
- **Finding for the report**: both CF variants score *below* `most_popular` on
  Precision/NDCG@10 in this offline temporal-holdout protocol (item-item P@10
  ≈0.013 vs popularity ≈0.058). This is expected, not a bug — popularity
  baselines are well known to win accuracy metrics because held-out future
  interactions skew toward broadly-liked items. Confirmed CF recommendations
  are qualitatively sound (e.g., a Star Wars/Gladiator fan gets Seven Samurai,
  Psycho, 12 Angry Men). This motivates the **popularity-bias analysis**
  extension (Week 6): show CF's coverage/novelty advantage over popularity even
  though raw precision is lower.
- Remaining extension: sweep `k` for both variants and report the tradeoff.

### Week 4 — Content-based *(done)*
- `content_based.py`: TF-IDF over `content` (genres + tags); user profile =
  Σ (r_ui − mean_u)·v_i; cosine scoring; `similar_items` for "more like this".
- Verified `similar_items` is sound: Toy Story → Bug's Life, Toy Story 2, Antz.
- Verified `recommend` for a typical user (54 ratings, liked Spider-Man/Dark
  Knight/Star Wars/Mission Impossible) returns coherent Action/Sci-Fi/IMAX picks.
- **Finding for the report**: for one extreme power-user (1,706 ratings spanning
  almost every genre), the profile degenerates toward whichever genres are
  *least common in their history*, because Σ(r−mean) over hundreds of ratings of
  common genres (comedy, action) accumulates large net-negative magnitude purely
  from volume, drowning out the genuine preference signal in this user's case.
  This is a known weakness of the plain centered-weighted-sum profile (the
  formula specified in the assignment) for prolific raters — worth a paragraph
  in "Discussion of limitations," not something to special-case in code.
- Extension/ablation still open: **TF-IDF vs raw genre counts** (`use_tfidf=False`
  is already wired as a constructor flag — just needs a comparison run);
  genres-only vs genres+tags.

### Week 5 — Matrix factorization *(done)*
- `matrix_factorization.py`: **FunkSVD** `r̂ = μ + b_u + b_i + p_u·q_i` trained by
  hand-rolled SGD (plain Python loop over shuffled observations each epoch — full
  pipeline incl. this still runs in ~18s on this dataset size, no vectorization
  trick needed).
- Verified convergence: train RMSE drops monotonically 0.957 → 0.719 over 20
  epochs (`n_factors=50, lr=0.005, reg=0.02`). `predict()` falls back to the
  global mean for unknown users/items.
- Verified qualitatively: a Comedy/Drama/Romance fan (Forrest Gump, Mrs.
  Doubtfire, Emma) gets Shawshank, Casablanca, Princess Bride, Amadeus — sharp,
  well-known picks in the right register.
- **Result**: FunkSVD (P@10≈0.032) clearly beats every other *personalized*
  method (content-based ≈0.005, item-item CF ≈0.013, user-user CF ≈0.004),
  though still below `most_popular` (≈0.058) — consistent with the
  popularity-bias pattern noted in Weeks 3–4.
- Remaining extension: tune `n_factors`/`lr`/`reg`; optional scikit-surprise SVD
  cross-check.

### Week 6 — Evaluation, analysis, UI polish *(tuning/ablations/bias done)*
- `src/experiments.py` + `run_experiments.py`: hyperparameter sweeps, ablations,
  and popularity-bias table, all saved to `results/*.csv` + `results/figures/*.png`.
- **CF k-sweep**: precision peaks at small k for both variants (item-item best
  at k=10, P@10≈0.0152; user-user best at k=5, P@10≈0.0193 — beating item-item
  at that one setting). Coverage falls monotonically as k grows for both (more
  neighbors → recommendations converge on the same popular items), the
  classic precision/diversity tradeoff.
- **MF n_factors / reg sweep — overfitting finding**: train RMSE drops sharply
  with more factors (0.815→0.718) or less regularization (0.811→0.745), but
  **held-out test RMSE barely moves (≈0.883–0.888) and even ticks up slightly**.
  Train RMSE alone (what Week 5 reported) hid this — it falls monotonically and
  looks like "more capacity is always better." Wiring `evaluation.rmse`/`mae`
  against the test set (previously implemented but unused) exposed that this
  model overfits past `n_factors≈10-20` / `reg≈0.005-0.02`; ranking quality
  (P@10) confirms it, peaking at the same small/well-regularized settings
  rather than improving with capacity.
- **Content-based ablation**: TF-IDF (genres+tags) beats raw counts (P@10
  0.0047 vs 0.0032) — TF-IDF's IDF down-weighting of ubiquitous genres helps.
  Genres-only TF-IDF is close to genres+tags (0.0046 vs 0.0047) but with much
  higher coverage (0.91 vs 0.85), suggesting tags add little signal here while
  narrowing variety.
- **Popularity bias** (mean popularity-percentile of recommended items, 1.0 =
  most popular item in train): `most_popular` 0.997, `matrix_factorization`
  0.871, `highest_average` 0.854 — all skew heavily to blockbusters.
  `content_based` 0.537, `user_user_cf` 0.516, `random` 0.491 sit near neutral.
  **`item_item_cf` is the outlier at 0.270 — *negative* popularity bias**,
  meaning it systematically surfaces below-average-popularity items. Combined
  with its low raw precision (Week 3), this is the report's strongest case
  that accuracy-only evaluation undersells item-item CF's practical value
  (novelty/diversity it provides that popularity-chasing models don't).
- Remaining: recommendation examples for ≥3 users (qualitative comparison
  across methods) consolidated into the report; Streamlit polish (algorithm
  switch already works — could add "why recommended" explanations).

### Week 7 — Report + slide deck
- Slide deck: technical challenges, method comparison, final remarks (per the brief).
- Written report following the guideline structure (intro → dataset → EDA → algorithms →
  protocol → results table → examples → limitations → conclusion).

---

## Shared interface (every recommender)

```python
model.fit(train_df, items_df=None) -> self
model.recommend(user_id, n=10, exclude_seen=True) -> list[(item_id, score)]
```

This uniformity is what lets `evaluate_ranking` and the Streamlit app treat all
methods interchangeably.

## Extensions targeted (for a higher grade)

User-user vs item-item · TF-IDF vs raw genres · k / latent-factor tuning ·
novelty + diversity metrics · popularity-bias analysis · Streamlit interface ·
temporal vs random split ablation.
