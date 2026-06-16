"""Project-wide configuration: paths, column names, and defaults.

Flat-repo layout: this file lives in ``src/`` and the project root is its parent.
"""

from pathlib import Path

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# MovieLens "latest-small" (the track chosen for this project).
DATASET_DIR = RAW_DATA_DIR / "ml-latest-small"
RATINGS_PATH = DATASET_DIR / "ratings.csv"
MOVIES_PATH = DATASET_DIR / "movies.csv"
TAGS_PATH = DATASET_DIR / "tags.csv"
LINKS_PATH = DATASET_DIR / "links.csv"

# --- Column names -----------------------------------------------------------
USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL = "title"
GENRES_COL = "genres"
TAG_COL = "tag"

# --- Defaults ---------------------------------------------------------------
TOP_K = 10                 # default recommendation list length
RANDOM_STATE = 42
RELEVANCE_THRESHOLD = 4.0  # a test rating >= this counts as a "relevant" item
TEST_FRACTION = 0.2        # per-user temporal holdout fraction
MIN_RATINGS_PER_USER = 5   # users below this are dropped before splitting
MIN_RATINGS_PER_ITEM = 5   # items below this are dropped (avoids spurious CF similarity)
CF_SHRINKAGE = 10          # significance-weighting constant for neighborhood CF similarity
