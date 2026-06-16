"""Generates the project slide deck as a PDF (results/slides.pdf).

One-off script, not part of the main pipeline: run after results/ has been
populated by main.py and run_experiments.py.

Usage:
    python make_slides.py
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src import config

OUT_PATH = config.RESULTS_DIR / "slides.pdf"
FIG = config.FIGURES_DIR

NAVY = "#1F2A44"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
RED = "#C00000"
GREEN = "#70AD47"
GRAY = "#595959"
LIGHT = "#F2F2F2"

W, H = 13.333, 7.5  # 16:9


def new_slide():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    return fig


def header(fig, title, subtitle=None):
    fig.text(0.06, 0.90, title, fontsize=28, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.06, 0.835, subtitle, fontsize=15, color=GRAY)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.80, 0.80], color=BLUE, linewidth=2,
                               transform=fig.transFigure))


def footer(fig, page, total):
    fig.text(0.06, 0.03, "Movie Recommender System, from scratch", fontsize=10, color=GRAY)
    fig.text(0.94, 0.03, f"{page} / {total}", fontsize=10, color=GRAY, ha="right")


def _wrap(text, fontsize, width_in, indent=""):
    chars_per_line = max(10, int(width_in / (fontsize / 72 * 0.50)))
    lines = textwrap.wrap(text, width=chars_per_line)
    return lines or [""]


def bullets(fig, items, x=0.08, y=0.70, fontsize=16, color="black", right=0.94, gap=0.025):
    width_in = (right - x) * W
    sub_fontsize = fontsize - 3
    line_h = (fontsize / 72) / H * 1.35
    sub_line_h = (sub_fontsize / 72) / H * 1.35
    cur_y = y
    for item in items:
        if isinstance(item, tuple):
            text, sub = item
        else:
            text, sub = item, None
        text_lines = _wrap(f"•  {text}", fontsize, width_in)
        for j, line in enumerate(text_lines):
            fig.text(x, cur_y - j * line_h, line, fontsize=fontsize, color=color)
        cur_y -= len(text_lines) * line_h
        if sub:
            sub_lines = _wrap(sub, sub_fontsize, width_in - 0.03 * W)
            cur_y -= 0.012
            for j, line in enumerate(sub_lines):
                fig.text(x + 0.03, cur_y - j * sub_line_h, line, fontsize=sub_fontsize, color=GRAY)
            cur_y -= len(sub_lines) * sub_line_h
        cur_y -= gap


def add_image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    if path.exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, f"(missing: {path.name})", ha="center", va="center")


slides = []
TOTAL = 13

# 1. Title
fig = new_slide()
fig.patch.set_facecolor(NAVY)
fig.text(0.5, 0.60, "Movie Recommender System", fontsize=40, fontweight="bold",
          color="white", ha="center")
fig.text(0.5, 0.50, "A from scratch comparison of recommendation methods on MovieLens",
          fontsize=18, color="#CFE0F5", ha="center")
fig.text(0.5, 0.30, "Recommender Systems, Individual Project", fontsize=16, color="white", ha="center")
fig.text(0.5, 0.25, "Prof. Marc Torrens, ESADE", fontsize=14, color="#CFE0F5", ha="center")
fig.text(0.5, 0.10, "Jan Erik Sternberg", fontsize=14, color="white", ha="center")
slides.append(fig)

# 2. Problem & approach
fig = new_slide()
header(fig, "Goal and approach", "What the assignment asks for, and the choice made here")
bullets(fig, [
    "Build a working movie recommender prototype and evaluate it properly",
    "Grading: 50% technical implementation, 30% evaluation, 20% user experience",
    ("Every model is implemented from scratch",
     "no recommender library calls: similarity, SGD updates, and ranking are hand written"),
    ("Dataset: MovieLens latest small (610 users, 9,742 movies, 100,836 ratings)",
     "the full 32M rating dump was kept locally but not used, it is too heavy for the non vectorized code here"),
    ("Deliverable: a Streamlit prototype plus this slide deck",
     "targets the user experience component directly"),
])
footer(fig, 2, TOTAL)
slides.append(fig)

# 3. Dataset / EDA
fig = new_slide()
header(fig, "Dataset at a glance", "Rating distribution and activity skew")
add_image(fig, FIG / "eda_rating_distribution.png", [0.05, 0.15, 0.42, 0.60])
add_image(fig, FIG / "eda_ratings_per_user.png", [0.53, 0.15, 0.42, 0.60])
fig.text(0.06, 0.10, "Ratings skew positive (most ratings are 3.5 to 5.0), and a small number of", fontsize=13, color=GRAY)
fig.text(0.06, 0.065, "very active users account for a large share of all ratings, a typical long tail pattern.", fontsize=13, color=GRAY)
footer(fig, 3, TOTAL)
slides.append(fig)

# 4. Evaluation protocol
fig = new_slide()
header(fig, "Evaluation protocol", "One shared protocol for every model")
bullets(fig, [
    ("Per user temporal holdout split",
     "each user's most recent 20% of ratings go to test, the rest to train"),
    ("Relevance threshold: a test rating of 4.0 or higher counts as relevant",
     "used for ranking metrics"),
    "Ranking metrics: Precision, Recall, NDCG, MAP, MRR, Hit Rate, all at K=10",
    "Beyond accuracy: catalog coverage, novelty, intra list diversity, popularity bias",
    "Rating prediction: RMSE and MAE on held out ratings (matrix factorization only)",
])
footer(fig, 4, TOTAL)
slides.append(fig)

# 5. Models overview
fig = new_slide()
header(fig, "Seven models, one interface", "Every model implements fit() and recommend()")
rows = [
    ("Non personalized", "Most Popular, Highest Average Rating, Random", BLUE),
    ("Content based", "TF-IDF over genres and tags, centered rating weighted user profile, cosine scoring", GREEN),
    ("Collaborative filtering", "Item item and user user neighborhood CF with significance weighted (shrunk) cosine similarity", ORANGE),
    ("Matrix factorization", "FunkSVD: r-hat = mu + b_u + b_i + p_u . q_i, trained with hand rolled SGD", RED),
]
y = 0.68
for name, desc, color in rows:
    fig.add_artist(plt.Rectangle((0.06, y - 0.01), 0.02, 0.10, transform=fig.transFigure,
                                  facecolor=color, edgecolor="none"))
    fig.text(0.10, y + 0.055, name, fontsize=17, fontweight="bold", color=NAVY)
    fig.text(0.10, y + 0.015, desc, fontsize=13, color=GRAY)
    y -= 0.17
footer(fig, 5, TOTAL)
slides.append(fig)

# 6. CF debugging story
fig = new_slide()
header(fig, "Technical challenge: collaborative filtering artifacts", "Two sparse-data bugs found by inspecting real recommendations, not just metrics")
bullets(fig, [
    ("Bug 1: items or users sharing a single rater get cosine similarity 1.0",
     "fixed with significance weighted shrinkage: sim *= n_common / (n_common + alpha)"),
    ("Bug 2: every score collapsed to exactly 5.0",
     "a fixed global top-k of neighbors rarely overlapped a user's own rated items with ~9.7K items, so the weighted average reduced to one rating"),
    ("Fix: item item CF selects its top-k neighbors dynamically per recommendation,",
     "restricted to the items the target user actually rated, instead of a global top-k computed once at fit time"),
    ("User user CF keeps a global top-k",
     "with only about 610 users this failure mode does not apply"),
])
footer(fig, 6, TOTAL)
slides.append(fig)

# 7. CF k sweep
fig = new_slide()
header(fig, "Tuning: neighborhood size k", "Precision and catalog coverage trade off as k grows")
add_image(fig, FIG / "tuning_cf_k_precision.png", [0.05, 0.15, 0.42, 0.60])
add_image(fig, FIG / "tuning_cf_k_coverage.png", [0.53, 0.15, 0.42, 0.60])
fig.text(0.06, 0.10, "Smaller neighborhoods give both better precision and higher coverage:", fontsize=13, color=GRAY)
fig.text(0.06, 0.065, "user user peaks at k=5, item item at k=10. Larger k narrows recommendations toward the same popular items.", fontsize=13, color=GRAY)
footer(fig, 7, TOTAL)
slides.append(fig)

# 8. MF overfitting
fig = new_slide()
header(fig, "Matrix factorization: finding overfitting", "Train RMSE alone hides it, held out test RMSE reveals it")
add_image(fig, FIG / "tuning_mf_factors.png", [0.06, 0.15, 0.42, 0.60])
add_image(fig, FIG / "tuning_mf_reg.png", [0.53, 0.15, 0.42, 0.60])
fig.text(0.06, 0.10, "Train RMSE keeps falling with more latent factors or less regularization (0.81 to 0.72),", fontsize=13, color=GRAY)
fig.text(0.06, 0.065, "but test RMSE stays flat near 0.88, and ranking quality peaks at small, well regularized settings.", fontsize=13, color=GRAY)
footer(fig, 8, TOTAL)
slides.append(fig)

# 9. Content based ablation
fig = new_slide()
header(fig, "Content based: feature ablation", "TF-IDF versus raw counts, genres only versus genres and tags")
add_image(fig, FIG / "ablation_content_features.png", [0.10, 0.13, 0.55, 0.62])
bullets(fig, [
    ("TF-IDF beats raw counts", "0.0047 vs 0.0032 precision at 10"),
    ("Genres only is close to genres and tags", "0.0046 vs 0.0047"),
    ("but with much higher coverage", "0.91 vs 0.85"),
    "Tags add little signal here while narrowing variety",
], x=0.68, y=0.62, fontsize=14)
footer(fig, 9, TOTAL)
slides.append(fig)

# 10. Results table
fig = new_slide()
header(fig, "Overall results", "Precision, recall, and NDCG at K=10, full test set")
col_labels = ["Model", "Precision@10", "Recall@10", "NDCG@10", "Coverage@10"]
data = [
    ["Most Popular", "0.058", "0.050", "0.076", "0.026"],
    ["Highest Average", "0.023", "0.018", "0.031", "0.011"],
    ["Matrix Factorization", "0.031", "0.026", "0.038", "0.060"],
    ["Item Item CF", "0.013", "0.004", "0.013", "0.377"],
    ["Content Based", "0.005", "0.006", "0.007", "0.851"],
    ["User User CF", "0.004", "0.005", "0.005", "0.232"],
    ["Random", "0.004", "0.004", "0.004", "0.792"],
]
ax = fig.add_axes([0.06, 0.15, 0.88, 0.62])
ax.axis("off")
table = ax.table(cellText=data, colLabels=col_labels, loc="center", cellLoc="center")
table.scale(1, 2.0)
table.auto_set_font_size(False)
table.set_fontsize(13)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor(LIGHT)
fig.text(0.06, 0.10, "Popularity baselines win on raw accuracy, expected in offline evaluation since future", fontsize=13, color=GRAY)
fig.text(0.06, 0.065, "interactions skew toward broadly liked items. FunkSVD is the strongest personalized model.", fontsize=13, color=GRAY)
footer(fig, 10, TOTAL)
slides.append(fig)

# 11. Popularity bias
fig = new_slide()
header(fig, "Beyond accuracy: popularity bias", "Mean popularity percentile of recommended items, 1.0 is the most popular item in train")
add_image(fig, FIG / "popularity_bias.png", [0.08, 0.13, 0.52, 0.64])
bullets(fig, [
    ("Most Popular: 0.997, Matrix Factorization: 0.871", "lean heavily toward blockbusters"),
    ("Content based, user user CF, and random sit near 0.5", "roughly neutral"),
    ("Item item CF: 0.270, the only model with negative bias",
     "it systematically surfaces below average popularity items"),
    "This is invisible to precision alone, but matters for a real product",
], x=0.65, y=0.62, fontsize=14)
footer(fig, 11, TOTAL)
slides.append(fig)

# 12. Streamlit / UX
fig = new_slide()
header(fig, "Prototype: Streamlit app", "Targets the user experience grading component")
bullets(fig, [
    "Pick any user and see their liked movies alongside live recommendations",
    "Switch between all seven models and compare scores side by side",
    "Adjustable recommendation list length",
    "Deployed and publicly reachable, no setup required to try it",
])
fig.text(0.06, 0.30, "Live demo: https://recommendersystemmovies.streamlit.app/", fontsize=16,
          color=BLUE, fontweight="bold")
footer(fig, 12, TOTAL)
slides.append(fig)

# 13. Limitations & conclusion
fig = new_slide()
header(fig, "Limitations and final remarks", None)
bullets(fig, [
    ("Content based profiles degrade for extreme power users",
     "thousands of ratings across nearly every genre wash out the taste signal in the centered weighted sum"),
    ("Offline accuracy metrics favor popularity",
     "they do not capture novelty or discovery value, which is why beyond accuracy metrics matter"),
    ("Matrix factorization needs careful regularization",
     "more capacity alone does not generalize better, confirmed by held out test RMSE"),
    ("Overall: building every model from scratch surfaced concrete, debuggable failure modes",
     "that using a library would likely have hidden"),
])
footer(fig, 13, TOTAL)
slides.append(fig)


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        for fig in slides:
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved {len(slides)} slides -> {OUT_PATH}")


if __name__ == "__main__":
    main()
