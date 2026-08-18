<p align="center">
  <img src="frontend/henkel-logo.png" width="200"/>
</p>

# Henkel BBS Safety Classifier

A production-grade NLP pipeline that automatically classifies French-language safety observations written by Henkel warehouse staff. Built to replace manual scoring, the system predicts a severity label for each observation and generates a monthly leaderboard of observers.

---

## The Problem

Safety officers walk the warehouse floor and write short observations in French — things like *"employé sans gilet"* or *"on a sensibilisé l'opérateur"*. These are then scored manually on a 4-point scale. The observations are highly messy: severe typos, abbreviations, missing accents, often only 2–3 words long.

The goal was to automate this scoring reliably, with full visibility into *why* each prediction was made.

---

## The Dataset

Each observation carries one of four labels, derived from the original BBSWA scoring columns:

| Label | Meaning | Points |
|-------|---------|--------|
| `0.0` | Infrastructural / Environmental Risk | 0 |
| `1.0` | Human Risk — Observed (no interaction) | 1 |
| `1.5` | Human Risk — Interacted verbally | 1.5 |
| `2.0` | Human Risk — Behaviour corrected | 2 |

Raw files arrive as monthly Excel exports with columns including `Observer`, `Observationcomment`, and four quality scoring columns. A preparation step maps these to a unified `label` column and merges all months into a single training file (`BBSWA_All_Merged.xlsx`).

---

## Architecture — Why a Cascade and Not a Single Model

Early experiments with a single multi-class model (including fine-tuned CamemBERT) revealed a critical problem: **class 1.5 was almost never predicted correctly**. With ~70–80 samples out of 784 total Phase 2 rows, it was systematically absorbed into class 2.0, giving 0.00 F1 for "Interacted" despite 89%+ overall accuracy.

The core insight was that treating this as a single 4-class problem was wrong. The classes have a natural **decision hierarchy**, and splitting that into sequential binary decisions makes each individual problem far easier — and makes failures immediately locatable.

The pipeline has **3 layers**, each with its own rule block and ML fallback. This means at any point you can inspect exactly which layer made a decision and which signal triggered it. There is no black box.

---

## Pipeline — Layer by Layer

```
Raw Observation Text
        │
        ▼
  ┌─────────────────────────────────────────┐
  │  LAYER 1 — Infrastructure vs Human Risk │
  │                                         │
  │  Rules (fire first):                    │
  │  · Non-French / Arabic text → defer ML  │
  │  · Fuzzy PPE match (casque, gilet…) → 1 │
  │  · ≤ 2 words → 0 (infra)               │
  │  · Role keyword (employé, chef…) → 1    │
  │  · Action keyword (sensibilisé…) → 1    │
  │                                         │
  │  ML fallback:                           │
  │  · TF-IDF char n-grams (2–4)            │
  │  · Logistic Regression, threshold 0.35  │
  │  · Unclassified rows get 3× sample weight│
  └───────────────┬─────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
       0.0               Human Risk
    (done)                   │
                             ▼
  ┌─────────────────────────────────────────┐
  │  LAYER 2 — Observed vs Action Taken     │
  │  (Split A)                              │
  │                                         │
  │  Question: did the observer interact    │
  │  with the person, or just watch?        │
  │                                         │
  │  Rules: any active verb fires → Interact│
  │  (sensibilisé, corrigé, arrêté, …)      │
  │                                         │
  │  ML fallback:                           │
  │  · TF-IDF char n-grams (2–4)            │
  │  · Logistic Regression, class-weighted  │
  └───────────────┬─────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
       1.0               Interaction
    (done)             confirmed
                             │
                             ▼
  ┌─────────────────────────────────────────┐
  │  LAYER 3 — Interacted vs Corrected      │
  │  (Split B)                              │
  │                                         │
  │  Question: did the person's behaviour   │
  │  actually change?                       │
  │                                         │
  │  Rules:                                 │
  │  · Physical correction verb → 2.0       │
  │    (corrigé, rangé, mis en place, …)    │
  │  · Verbal-only signal → 1.5             │
  │    (refusé, sans suite, promis de…)     │
  │                                         │
  │  ML fallback:                           │
  │  · TF-IDF char + word n-grams (FeatureUnion)│
  │  · Logistic Regression, threshold 0.55  │
  └───────────────┬─────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
       1.5                  2.0
  (Interacted)          (Corrected)
```

### Why Rules First, ML Second

Each layer runs deterministic regex patterns before the ML model sees the text. If a rule fires, the ML is skipped entirely. This serves three purposes:

1. **Precision on known signals** — terms like *corrigé* or *sensibilisé* are unambiguous in context. A regex is 100% precise; a model might waver.
2. **Debuggability** — every prediction carries a `method` tag (`rule:pattern` or `model`). You always know exactly why the system decided what it decided.
3. **Data efficiency** — the ML only needs to learn the ambiguous cases. The easy cases are already handled, which means the model trains on a harder, more informative subset.

### Why TF-IDF + Logistic Regression and Not a Transformer

CamemBERT was tried for Phase 2. It achieved 72% accuracy on the severity test set but **completely failed on class 1.5** (F1 = 0.00) — because with ~80 samples of 1.5 in the entire dataset, even a large pretrained model cannot learn to distinguish it from 1.5 vs 2.0 when both classes share many of the same words.

TF-IDF on character n-grams (2–4) handles the messy French naturally: typos, missing accents, and fused words all produce overlapping character sequences that still match. Logistic Regression on top gives a calibrated probability (used for the thresholds in Layers 1 and 3) and trains in milliseconds on this dataset size.

The cascaded binary approach solves the 1.5 problem structurally: Split B only ever sees confirmed-interaction rows, which dramatically reduces ambiguity. The rule layer for Split B handles the unambiguous correction verbs, leaving the ML to deal only with genuinely uncertain cases.

---

## Text Preprocessing

All text goes through a shared normalisation step before any layer sees it:

- Lowercase
- Digit-letter boundary glueing (`2ouvriers` → `ouvriers`)
- Arabic script detection (filtered out — non-French entries are skipped)

Observer names are normalised separately to handle the many variants seen in production:
- Trailing zone suffixes stripped (`A/z`, `/Az`, `/AZ`, `A/Z`, …)
- Role garbage tags removed (`(funct)`, `REGLQ1_2 (funct)…`)
- Accents stripped, non-alphanumeric characters removed
- Result is deterministic — identical people always produce identical canonical strings

---

## Project Structure

```
Henkel-main/
├── main.py                  # FastAPI backend — all pipeline logic + API routes
├── BBSWA_All_Merged.xlsx    # Training data (not in repo — add locally)
├── requirements.txt
├── render.yaml              # Render.com deployment config
└── frontend/
    ├── index.html
    ├── app.js
    └── style.css
```

---

## API Endpoints

### `POST /analyze`
Upload a new monthly Excel file. The server trains on the full history, predicts labels for every row in the new file, and returns:
- Category distribution (counts per label)
- Top-15 observer leaderboard (ranked by predicted points)
- Downloadable Excel with `Predicted_Label` and `Categorie` columns

**Required columns:** `Observer`, `Observationcomment`

### `POST /prepare`
Upload one or more raw monthly Excel files (with the original BBSWA quality scoring columns). Merges them into `BBSWA_All_Merged.xlsx` and reloads the training data in memory without restarting the server.

### `GET /health`
Returns server status and current training row count.

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-username/henkel-bbs.git
cd henkel-bbs

pip install -r requirements.txt

# Place your training file next to main.py
cp /path/to/BBSWA_All_Merged.xlsx .

uvicorn main:app --reload
```

Open `http://localhost:8000` for the frontend.

---

## Deployment

The project includes a `render.yaml` for one-click deploy to [Render](https://render.com):

```yaml
services:
  - type: web
    name: henkel-bbs
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    plan: free
```

Note: `BBSWA_All_Merged.xlsx` is not committed to the repo (it contains internal data). Add it as a [Render Disk](https://render.com/docs/disks) or via the `/prepare` endpoint on first deploy.

---

## Results Summary

| Layer | Task | Method | Accuracy |
|-------|------|--------|----------|
| Layer 1 | Infra vs Human | Rules + TF-IDF LR | ~91–92% |
| Layer 2 | Observed vs Interact | Rules + TF-IDF LR | reported per run |
| Layer 3 | Interacted vs Corrected | Rules + TF-IDF LR (char+word) | reported per run |
| Full system | All 4 classes | Cascade | reported per run |

Cross-validation (5-fold, stratified) is run on each layer independently during notebook development, with separate held-out test sets to measure true generalisation. The decision path (which rule or model fired at which layer) is logged for every prediction.

---

## Notebooks

Development and evaluation were done in Jupyter:

- **Cell 0 / Phase 1** — Layer 1 training, cross-validation, hyperparameter search
- **Cell 1** — Phase 2 exploration with CamemBERT (archived; revealed the 1.5 data problem)
- **Cell 2** — Cascaded binary Phase 2 (Layers 2 + 3), full CV and test reports per layer

---

## Built With

- [scikit-learn](https://scikit-learn.org/) — TF-IDF, Logistic Regression, cross-validation
- [FastAPI](https://fastapi.tiangolo.com/) — API backend
- [thefuzz](https://github.com/seatgeek/thefuzz) — fuzzy PPE term matching in Layer 1
- [pandas](https://pandas.pydata.org/) / [openpyxl](https://openpyxl.readthedocs.io/) — data handling
