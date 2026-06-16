"""Evaluation metrics and the offline evaluation loop.

Ranking metrics treat a recommendation as a ranked list of item IDs and compare it
against the user's set of relevant (held-out) items. Beyond-accuracy metrics
(coverage, novelty, intra-list diversity) capture what accuracy misses.

All per-user functions take ``recommended`` (ranked list of item IDs) and
``relevant`` (a set of relevant item IDs). Fully implemented.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from . import config


# --- Per-user ranking metrics ----------------------------------------------
def precision_at_k(recommended, relevant, k=config.TOP_K) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=config.TOP_K) -> float:
    if not relevant:
        return 0.0
    topk = recommended[:k]
    hits = sum(1 for i in topk if i in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k=config.TOP_K) -> float:
    return 1.0 if any(i in relevant for i in recommended[:k]) else 0.0


def average_precision_at_k(recommended, relevant, k=config.TOP_K) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def dcg_at_k(relevances, k=config.TOP_K) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(recommended, relevant, k=config.TOP_K) -> float:
    if not relevant:
        return 0.0
    gains = [1.0 if i in relevant else 0.0 for i in recommended[:k]]
    idcg = dcg_at_k([1.0] * min(len(relevant), k), k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def reciprocal_rank(recommended, relevant, k=config.TOP_K) -> float:
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


# --- Beyond-accuracy (aggregate over all users) -----------------------------
def catalog_coverage(all_recommended, catalog_items) -> float:
    """Fraction of the catalog that appears in at least one user's recommendations."""
    recommended_unique = set()
    for recs in all_recommended:
        recommended_unique.update(recs)
    return len(recommended_unique) / len(set(catalog_items))


def novelty(all_recommended, item_popularity, n_users) -> float:
    """Mean self-information -log2(p(item)) of recommended items. Higher = more novel."""
    total, count = 0.0, 0
    for recs in all_recommended:
        for item in recs:
            p = item_popularity.get(item, 1) / n_users
            total += -math.log2(p) if p > 0 else 0.0
            count += 1
    return total / count if count else 0.0


def intra_list_diversity(recommended, item_vectors) -> float:
    """1 - mean pairwise cosine similarity within one user's list (needs item vectors)."""
    items = [i for i in recommended if i in item_vectors]
    if len(items) < 2:
        return 0.0
    sims = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            va, vb = item_vectors[items[a]], item_vectors[items[b]]
            denom = (np.linalg.norm(va) * np.linalg.norm(vb))
            sims.append(float(va @ vb) / denom if denom else 0.0)
    return 1.0 - (sum(sims) / len(sims))


# --- Rating-prediction metrics ----------------------------------------------
def rmse(predictions, truths) -> float:
    p, t = np.asarray(predictions, float), np.asarray(truths, float)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(predictions, truths) -> float:
    p, t = np.asarray(predictions, float), np.asarray(truths, float)
    return float(np.mean(np.abs(p - t)))


# --- Offline evaluation loop ------------------------------------------------
def evaluate_ranking(model, relevant_by_user, catalog_items, k=config.TOP_K,
                     item_popularity=None, n_users=None) -> dict:
    """Run top-k recommendation for every test user and average the metrics.

    ``model.recommend(user_id, n, exclude_seen=True)`` must return a list of
    (item_id, score) tuples or item_ids.
    """
    per_user = {"precision": [], "recall": [], "ndcg": [], "map": [], "mrr": [], "hit": []}
    all_recommended = []

    for user_id, relevant in relevant_by_user.items():
        recs = model.recommend(user_id, n=k, exclude_seen=True)
        rec_ids = [r[0] if isinstance(r, (tuple, list)) else r for r in recs]
        all_recommended.append(rec_ids)

        per_user["precision"].append(precision_at_k(rec_ids, relevant, k))
        per_user["recall"].append(recall_at_k(rec_ids, relevant, k))
        per_user["ndcg"].append(ndcg_at_k(rec_ids, relevant, k))
        per_user["map"].append(average_precision_at_k(rec_ids, relevant, k))
        per_user["mrr"].append(reciprocal_rank(rec_ids, relevant, k))
        per_user["hit"].append(hit_rate_at_k(rec_ids, relevant, k))

    results = {f"{m}@{k}": float(np.mean(v)) if v else 0.0 for m, v in per_user.items()}
    results[f"coverage@{k}"] = catalog_coverage(all_recommended, catalog_items)
    if item_popularity is not None and n_users:
        results[f"novelty@{k}"] = novelty(all_recommended, item_popularity, n_users)
    return results
