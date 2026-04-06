"""
Henkel BBS Safety Classifier — FastAPI Backend
===============================================
- Trains on BBSWA_All_Merged.xlsx at startup (fixed training data).
- Accepts a new monthly Excel file via POST /analyze.
- Runs the full 3-layer cascade pipeline on the new file.
- Returns predicted labels, leaderboard, distribution, and downloadable Excel.

Also exposes POST /prepare to merge raw monthly Excel files into the training file.
"""

import re
import io
import os
import warnings
import base64
import unicodedata
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
MERGED_FILE  = os.path.join(BASE_DIR, "BBSWA_All_Merged.xlsx")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ─────────────────────────────────────────────────────────────
# COLUMN NAMES
# ─────────────────────────────────────────────────────────────
OBSERVER_COL    = "Observer"
OBSERVATION_COL = "Observationcomment"
LABEL_COL       = "label"
MONTH_COL       = "Month"

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
FUZZY_THRESHOLD          = 90
UNCLASSIFIED_WEIGHT      = 3.0
ML_DECISION_THRESHOLD    = 0.35
BEST_C_VALUE             = 2.0
SHORT_FILTER_ML_OVERRIDE = 0.40
ARABIC_CHAR_THRESHOLD    = 0.30
RANDOM_STATE             = 42
SPLIT_B_THRESHOLD        = 0.55

# Score mapping used by /prepare endpoint (mirrors Data_Preparation.ipynb)
COLUMNS_MAP = {
    "BBSWA de faible qualite \n0 point ":           0.0,
    "BBSWA de qualite moderee \n1 point ":           1.0,
    "BBSWA de qualite acceptable\n1.5 point":        1.5,
    "BBSWA de qualite excellente \n2 points ":       2.0,
}

# ─────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────
_DIGIT_GLUE = re.compile(r'\d+([a-zàâäéèêëîïôùûüç])', re.IGNORECASE)
_ARABIC_RE  = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

def normalise(text: str) -> str:
    """Lowercase and glue digit-letter boundaries."""
    return _DIGIT_GLUE.sub(r'\1', str(text).strip().lower())

def is_non_french(text: str) -> bool:
    """Returns True if the text is predominantly Arabic script."""
    chars = str(text).replace(" ", "")
    if not chars:
        return False
    arabic_len = sum(len(m) for m in _ARABIC_RE.findall(str(text)))
    return (arabic_len / len(chars)) >= ARABIC_CHAR_THRESHOLD

# ─────────────────────────────────────────────────────────────
# RULE DICTIONARIES
# ─────────────────────────────────────────────────────────────
FUZZY_TARGETS = ["epi", "loto", "bavette", "casque", "gilet", "harnais"]

ROLE_PATTERNS = [
    r"\bquelqu'un[a-z]*\b", r"\bindividu[a-z]*\b", r"\bhomme[a-z]*\b",
    r"\bpreparateur[a-z]*\b", r"\bpréparateur[a-z]*\b", r"\bmanutentionnaire[a-z]*\b",
    r"\bconducteur[a-z]*\b", r"\bchauffeur[a-z]*\b", r"\bclark[a-z]*\b",
    r"\belectricien[a-z]*\b", r"\bélectricien[a-z]*\b", r"\bsoudeur[a-z]*\b",
    r"\bmaintenance\b", r"\bintervenant[a-z]*\b", r"\bouvrier[a-z]*\b",
    r"\bemployé[a-z]*\b", r"\bemploy[eé][a-z]*\b", r"\bsalarié[a-z]*\b",
    r"\bprestataire[a-z]*\b", r"\bvisiteur[a-z]*\b", r"\bchef[a-z]*\b",
    r"\bsuperviseur[a-z]*\b", r"\bmanager[a-z]*\b", r"\bresponsable[a-z]*\b",
    r"\bstagiaire[a-z]*\b", r"\bintérimaire[a-z]*\b", r"\binterimaire[a-z]*\b",
    r"\bpersonnel[a-z]*\b", r"\bcollègue[a-z]*\b", r"\bcollegu[a-z]*\b",
]

ACTION_PATTERNS = [
    r"\bsensibilis[a-zéèê]*\b", r"\baprès\s+sensibilis", r"\bsensibilisation\b",
    r"\bon\s+lui\s+a\b", r"\bnous\s+avons\b", r"\bon\s+a\b", r"\bbon\s+travail\b",
]

INTERACTION_PATTERNS_A = [
    r"\bsensibilis[a-zéèê]*\b", r"\baprès\s+sensibilis",
    r"\bon\s+lui\s+a\b", r"\bnous\s+avons\b",
    r"\bon\s+a\s+(dit|inform|demand|averti|rappel|expliqu|conseil)",
    r"\bavert[ia][a-z]*\b", r"\binformé[es]*\b", r"\bexpliqu[eéè][a-z]*\b",
    r"\bdiscut[eéè][a-z]*\b", r"\béchang[eéè][a-z]*\b", r"\bprévenu[es]*\b",
    r"\bconscientis[eéè][a-z]*\b", r"\bintervenu\b", r"\bcorrigé[es]*\b",
    r"\bstoppé[es]*\b", r"\barrêté[es]*\b", r"\bmis\s+en\s+place\b",
    r"\bremis[e]*\b", r"\brangé[es]*\b", r"\bréparé[es]*\b", r"\bfixé[es]*\b",
    r"\bdone\b", r"\brappelé[es]*\b", r"\bdemandé[es]*\b",
]

CORRECTION_PATTERNS_B = [
    r"\bcorrig[eéè][a-z]*\b", r"\bstoppé[a-z]*\b", r"\barrêt[eéè][a-z]*\b",
    r"\bmis\s+en\s+place\b", r"\bremis[a-z]*\b", r"\brangé[a-z]*\b",
    r"\bréparé[a-z]*\b", r"\bfixé[a-z]*\b", r"\béquipé[a-z]*\b",
    r"\bsécurisé[a-z]*\b", r"\beffectué[a-z]*\b", r"\bport[eéè][a-z]*\b",
    r"\bobliger[a-z]*\b", r"\bréalis[eéè][a-z]*\b", r"\bdone\b",
    r"\bchangé[a-z]*\b", r"\bmodif[a-z]*\b", r"\brepris[a-z]*\b",
    r"\benlev[eéè][a-z]*\b", r"\binstall[eéè][a-z]*\b", r"\bmis\s+(son|sa|les|le|un)\b",
]

INTERACTION_ONLY_PATTERNS_B = [
    r"\brefus[eéè][a-z]*\b", r"\bpas\s+fait\b", r"\bavertissement\s+verbal\b",
    r"\bsensibilis[eéè][a-z]*\s+uniquement\b", r"\bpromis\s+de\b",
    r"\bn'a\s+pas\b", r"\bsans\s+suite\b",
]

# ─────────────────────────────────────────────────────────────
# OBSERVER NAME NORMALISATION & DEDUPLICATION
# ─────────────────────────────────────────────────────────────
# All variants of the trailing zone suffix written by observers:
#   "A/z", "/Az", "/AZ", "SaberA/z", "Yousef/Az", " /AZ", "A/Z" …
# Strip it entirely — it is pure noise.
_SUFFIX_RE = re.compile(
    r"[\s/]*[Aa]?[\s]*/[\s]*[Aa]?[\s]*[Zz][\s]*$",
    re.IGNORECASE,
)

# Strip role/function garbage tags: "(funct)", "(func)", "(function)" …
_ROLE_TAG_RE = re.compile(r"\s*\(func\w*\)\s*", re.IGNORECASE)

# Strip leading alphanumeric garbage prefixes before a "(funct)" tag,
# e.g. "REGLQ1_2 (funct)ighit Saber" → "ighit Saber"
_PREFIX_GARBAGE_RE = re.compile(r"^[\w\s]+\(func\w*\)", re.IGNORECASE)


def _clean_observer(name: str) -> str:
    """
    Deterministic normalisation that collapses every known variant of
    an observer name to one canonical string.  No fuzzy matching needed —
    after these steps, identical people produce identical strings.

    Transformations (applied in order):
      1. Strip surrounding whitespace, collapse internal runs to one space.
      2. Remove leading garbage prefix ("REGLQ1_2 (funct)").
      3. Remove role tags ("(funct)") anywhere in the string.
      4. Remove the trailing A/z zone suffix in ALL its spacing/case variants.
      5. Strip accents: î→i, é→e, â→a, …
      6. Remove every non-alphanumeric character (slashes, underscores, …).
      7. Collapse spaces, lower-case.

    Verified against every variant seen in production data:
      "Ighit Saber A/z" → "ighit saber"
      "Ighit SaberA/z"  → "ighit saber"   ✓
      "ighit Saber"     → "ighit saber"   ✓
      "REGLQ1_2 (funct)ighit Saber" → "ighit saber"  ✓
      "Nabil Alîk A/z"  → "nabil alik"
      "Nabil Alik /AZ"  → "nabil alik"    ✓
      "Mazrari Yousef/Az" → "mazrari yousef"
      "Mazrari Yousef /Az"→ "mazrari yousef" ✓
      "Kerfah Rabah/Az" → "kerfah rabah"  ✓
    """
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    s = _PREFIX_GARBAGE_RE.sub("", s).strip()
    s = _ROLE_TAG_RE.sub(" ", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()
    # strip accents
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    # remove all non-alphanumeric characters
    s = re.sub(r"[^a-zA-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalise_observer_column(series: pd.Series) -> pd.Series:
    """
    Replace every raw observer name with the cleaned canonical form.

    We use the canonical string itself as the display name (title-cased)
    so that the leaderboard shows tidy names like "Ighit Saber" instead
    of whichever messy variant happened to appear first.

    Steps:
      1. Compute canonical key for every row.
      2. Build a display name per canonical key = title-case of the key.
      3. Map every row to its display name.
    """
    canon = series.map(_clean_observer)
    display = canon.map(lambda k: k.title())   # "ighit saber" → "Ighit Saber"
    return display


# ─────────────────────────────────────────────────────────────
# LAYER 1 — INFRASTRUCTURE vs. HUMAN RISK
# ─────────────────────────────────────────────────────────────
def get_rule_block(obs: str):
    """
    Returns (block_name, rule_pred).
    rule_pred: 1.0 = human risk, 0.0 = infra, None = defer to ML.
    """
    if is_non_french(obs):
        return "NonFrench", None
    text = normalise(obs)
    for word in text.split():
        for target in FUZZY_TARGETS:
            if fuzz.ratio(word, target) >= FUZZY_THRESHOLD:
                return "Fuzzy", 1.0
    if len(text.split()) <= 2:
        return "Short", 0.0
    for pat in ROLE_PATTERNS:
        if re.search(pat, text):
            return "Role", 1.0
    for pat in ACTION_PATTERNS:
        if re.search(pat, text):
            return "Action", 1.0
    return "Unclassified", None

def build_l1() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=2, max_features=8000, sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            C=BEST_C_VALUE, class_weight="balanced",
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])

def predict_l1(texts: list, ml_model) -> list:
    """Layer-1 predictions for a list of raw observation strings."""
    preds = []
    for obs in texts:
        block, rule_p = get_rule_block(obs)
        ml_prob = ml_model.predict_proba([normalise(obs)])[0, 1]
        fp = rule_p
        if block == "Short" and ml_prob >= SHORT_FILTER_ML_OVERRIDE:
            fp = None
        if fp is None:
            fp = 1.0 if ml_prob >= ML_DECISION_THRESHOLD else 0.0
        preds.append(fp)
    return preds

# ─────────────────────────────────────────────────────────────
# LAYER 2 — OBSERVATION vs. ACTION TAKEN  (Split A)
# ─────────────────────────────────────────────────────────────
def build_l2(cw=None) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            min_df=2, max_features=6000, sublinear_tf=True,
        )),
        ("lr", LogisticRegression(
            C=2.0, class_weight=cw,
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])

def split_a_rule(text: str):
    """Returns 1 (action taken) or None (defer to ML)."""
    t = normalise(text)
    for pat in INTERACTION_PATTERNS_A:
        if re.search(pat, t):
            return 1
    return None

# ─────────────────────────────────────────────────────────────
# LAYER 3 — INTERACTION vs. CORRECTION  (Split B)
# ─────────────────────────────────────────────────────────────
def build_l3() -> Pipeline:
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                min_df=2, max_features=3000, sublinear_tf=True,
            )),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 3),
                min_df=2, max_features=3000, sublinear_tf=True,
            )),
        ])),
        ("lr", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE,
        )),
    ])

def split_b_rule(text: str):
    """Returns 1 (corrected), 0 (interaction only), or None (defer to ML)."""
    t = normalise(text)
    for pat in CORRECTION_PATTERNS_B:
        if re.search(pat, t):
            return 1
    for pat in INTERACTION_ONLY_PATTERNS_B:
        if re.search(pat, t):
            return 0
    return None

# ─────────────────────────────────────────────────────────────
# CASCADE  — combine all 3 layers for a single observation
# ─────────────────────────────────────────────────────────────
def cascade(obs_text: str, l1_pred: float, ml_a, ml_b) -> float:
    if l1_pred == 0.0:
        return 0.0

    a = split_a_rule(obs_text)
    if a is None:
        a = ml_a.predict([normalise(obs_text)])[0]
    if a == 0:
        return 1.0

    b = split_b_rule(obs_text)
    if b is None:
        prob = ml_b.predict_proba([normalise(obs_text)])[0, 1]
        b = 1 if prob >= SPLIT_B_THRESHOLD else 0

    return 2.0 if b == 1 else 1.5

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE  — train on history, predict on new file
# ─────────────────────────────────────────────────────────────
def run_pipeline(history_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains the 3-layer cascade on `history_df` (labelled),
    predicts labels for every row in `new_df`.
    Returns `new_df` enriched with Predicted_Label and Categorie columns.
    """
    # Prepare labelled training set
    hist = history_df.dropna(subset=[LABEL_COL, OBSERVATION_COL]).copy()
    hist[LABEL_COL] = hist[LABEL_COL].astype(float)
    hist["bin"] = hist[LABEL_COL].replace({1.5: 1.0, 2.0: 1.0})
    hist = hist[~hist[OBSERVATION_COL].astype(str).apply(is_non_french)].reset_index(drop=True)

    X = hist[OBSERVATION_COL].astype(str)
    y = hist["bin"]

    # Layer 1
    X_norm  = X.apply(normalise)
    blocks  = X.apply(lambda x: get_rule_block(x)[0])
    weights = blocks.apply(lambda b: UNCLASSIFIED_WEIGHT if b == "Unclassified" else 1.0).values
    ml1 = build_l1()
    ml1.fit(X_norm, y, lr__sample_weight=weights)
    hist["l1"] = predict_l1(X.tolist(), ml1)

    # Layer 2
    tr_a = hist[(hist["l1"] == 1.0) & hist[LABEL_COL].isin([1.0, 1.5, 2.0])].copy()
    tr_a["norm"] = tr_a[OBSERVATION_COL].astype(str).apply(normalise)
    tr_a["ya"]   = (tr_a[LABEL_COL] != 1.0).astype(int)
    cw_arr = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr_a["ya"].values)
    ml2 = build_l2({0: cw_arr[0], 1: cw_arr[1]})
    ml2.fit(tr_a["norm"].values, tr_a["ya"].values)

    # Layer 3
    tr_b = tr_a[tr_a[LABEL_COL].isin([1.5, 2.0])].copy()
    tr_b["yb"] = (tr_b[LABEL_COL] == 2.0).astype(int)
    ml3 = build_l3()
    ml3_ready = len(tr_b) >= 4
    if ml3_ready:
        ml3.fit(tr_b["norm"].values, tr_b["yb"].values)

    # Predict new file
    out   = new_df.copy()
    texts = out[OBSERVATION_COL].astype(str).tolist()
    l1s   = predict_l1(texts, ml1)

    labels = []
    for t, l1 in zip(texts, l1s):
        if l1 == 0.0:
            labels.append(0.0)
        elif ml3_ready:
            labels.append(cascade(t, l1, ml2, ml3))
        else:
            labels.append(1.5)

    out["Predicted_Label"] = labels
    out["Categorie"] = out["Predicted_Label"].map({
        0.0: "Infra (0.0)",
        1.0: "Risque (1.0)",
        1.5: "Interaction (1.5)",
        2.0: "Corrige (2.0)",
    })
    return out

# ─────────────────────────────────────────────────────────────
# DATA PREPARATION HELPER  (mirrors Data_Preparation.ipynb)
# ─────────────────────────────────────────────────────────────
def prepare_single_file(content: bytes, month_name: str) -> pd.DataFrame:
    """
    Reads one raw monthly Excel file, maps quality columns to numeric labels,
    returns a clean DataFrame ready for merging into the training set.
    """
    df = pd.read_excel(io.BytesIO(content))
    df["label"] = None
    for col, value in COLUMNS_MAP.items():
        if col in df.columns:
            df.loc[df[col].notna(), "label"] = value
    df = df[[OBSERVER_COL, OBSERVATION_COL, "label"]].dropna(subset=["label"])
    df[MONTH_COL] = month_name
    return df

# ─────────────────────────────────────────────────────────────
# PRELOAD TRAINING DATA AT STARTUP
# ─────────────────────────────────────────────────────────────
if not os.path.exists(MERGED_FILE):
    raise RuntimeError(
        f"Training file not found: {MERGED_FILE}\n"
        "Place BBSWA_All_Merged.xlsx next to main.py before launching the server."
    )

HISTORY_DF = pd.read_excel(MERGED_FILE)
print(f"Training data loaded — {len(HISTORY_DF):,} rows from {MERGED_FILE}")

# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Henkel BBS Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /analyze ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload a new monthly Excel file (needs Observer + Observationcomment columns).
    Returns: category distribution, top-10 observer leaderboard, base64 Excel result.
    """
    content = await file.read()

    try:
        new_df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot read Excel file: {e}")

    for col in [OBSERVER_COL, OBSERVATION_COL]:
        if col not in new_df.columns:
            raise HTTPException(
                400,
                f"Required column '{col}' not found. "
                f"Columns in file: {list(new_df.columns)}"
            )

    result = run_pipeline(HISTORY_DF, new_df)

    # Distribution
    counts = result["Predicted_Label"].value_counts().to_dict()
    dist = {
        "cat0":  int(counts.get(0.0, 0)),
        "cat1":  int(counts.get(1.0, 0)),
        "cat15": int(counts.get(1.5, 0)),
        "cat2":  int(counts.get(2.0, 0)),
    }

    # ── Leaderboard (mirrors Cell-4 notebook logic, predicted-only) ──────────

    # ── Step 1 — normalise observer names (strips /Az suffix, accents, typos)
    result[OBSERVER_COL] = normalise_observer_column(result[OBSERVER_COL])

    # Step 2 — total predicted points per observer (0.0 / 1.0 / 1.5 / 2.0)
    pts = (
        result.groupby(OBSERVER_COL)["Predicted_Label"]
        .sum()
        .reset_index(name="Pred_Points")
    )

    # Step 3 — total observation count per observer
    obs_count = (
        result.groupby(OBSERVER_COL)
        .size()
        .reset_index(name="Total_Obs")
    )

    # Step 4 — per-category observation *counts* (not point sums)
    cats = (
        result.groupby(OBSERVER_COL)["Predicted_Label"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Guarantee all four label columns exist even if a category is absent
    for lbl in [0.0, 1.0, 1.5, 2.0]:
        if lbl not in cats.columns:
            cats[lbl] = 0

    # Step 5 — merge everything
    lb = pts.merge(obs_count, on=OBSERVER_COL).merge(cats, on=OBSERVER_COL)

    # Step 6 — rank by predicted points, ties get the same (min) rank
    lb["Pred_Rank"] = (
        lb["Pred_Points"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Step 7 — sort by rank, keep top 15 (matches notebook head(15))
    lb = lb.sort_values("Pred_Rank").head(15).reset_index(drop=True)

    observers = []
    for _, row in lb.iterrows():
        observers.append({
            "name":      str(row[OBSERVER_COL]),
            "total":     float(row["Pred_Points"]),
            "rank":      int(row["Pred_Rank"]),
            "total_obs": int(row["Total_Obs"]),
            "cat0":      int(row[0.0]),
            "cat1":      int(row[1.0]),
            "cat15":     int(row[1.5]),
            "cat2":      int(row[2.0]),
        })

    # Excel download
    buf = io.BytesIO()
    result.to_excel(buf, index=False)
    excel_b64 = base64.b64encode(buf.getvalue()).decode()

    month = file.filename.replace(".xlsx", "").replace(".xls", "").strip()

    return JSONResponse({
        "month":     month,
        "total":     len(result),
        "dist":      dist,
        "observers": observers,
        "excel_b64": excel_b64,
        "filename":  f"BBS_Predit_{month}.xlsx",
    })


# ── POST /prepare ─────────────────────────────────────────────
@app.post("/prepare")
async def prepare(files: list[UploadFile] = File(...)):
    """
    Upload one or more raw monthly Excel files (with quality columns).
    Merges them into BBSWA_All_Merged.xlsx and reloads training data in memory.
    """
    global HISTORY_DF

    dfs = []
    for f in files:
        content = await f.read()
        month = f.filename.replace("Analyse BBSWA ", "").replace(".xlsx", "").strip()
        try:
            df = prepare_single_file(content, month)
            dfs.append(df)
        except Exception as e:
            raise HTTPException(400, f"Error processing {f.filename}: {e}")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_excel(MERGED_FILE, index=False)
    HISTORY_DF = merged
    print(f"Training data refreshed — {len(HISTORY_DF):,} rows")

    return JSONResponse({
        "status": "ok",
        "rows":   len(merged),
        "months": sorted(merged[MONTH_COL].unique().tolist()),
    })


# ── GET /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "training_rows": len(HISTORY_DF)}


# ── Static frontend — MUST be last (catch-all) ────────────────
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")