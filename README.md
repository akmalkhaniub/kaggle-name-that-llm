# Kaggle — Name That LLM (Prompt-Disjoint LLM Authorship Attribution)

Starter & competitive baseline workspace for the **[Name That LLM Competition](https://www.kaggle.com/competitions/name-that-llm)** on Kaggle.

> **Action Needed Before Pushing / Downloading Data:** Competition rules must be manually accepted on the Kaggle Web UI ([`kaggle.com/competitions/name-that-llm`](https://www.kaggle.com/competitions/name-that-llm)) before attaching competition data or submitting via CLI (`403 Forbidden` until joined).

---

## Competition Contract & Validation Design

- **Input Schema:** `train.csv` (`id`, `text`, `model`, `prompt_id`), `test.csv` (`id`, `text`, `prompt_id`).
- **Evaluation Metric:** **Balanced Accuracy** (`sklearn.metrics.balanced_accuracy_score`) evaluated on responses to **unseen `prompt_id`s**.
- **Leak-Free CV Strategy:** `GroupKFold(n_splits=5)` grouped strictly by `prompt_id` so prompt-specific topic words never leak across folds.
- **Required Output File:** `/kaggle/working/submission.parquet` with exact columns `['id', 'model']`.

---

## Upgraded Architecture (`baseline_tfidf_stylometry_logreg.ipynb`)

Beyond the official starter notebook (`Word (1,2)` + `Char (3,5)` + single `LogisticRegression(C=2.0)`), this notebook implements a prompt-invariant multi-view linear ensemble:

1. **Multi-View Sparse Feature Union (`FeatureUnion`):**
   - **Word + Punctuation Tokens (`(1, 3)` n-grams, `45k` dims):** Explicit tokenization retaining markdown delimiters, em-dashes, semicolons, and formatting cues alongside phrases.
   - **Subword Character N-Grams (`char_wb`, `(3, 5)` n-grams, `50k` dims):** Captures tokenization preferences, spacing, contraction patterns, and spelling morphology invariant to prompt topic.
   - **17 Structural Stylometry Features (`StylometryTransformer` + `StandardScaler`):** Character/word counts, lexical diversity (type-token ratio), sentence length mean/std, bullet list & numbered list ratios, markdown header/bold ratios, code-fence counts, punctuation densities, and capitalization/digit rates.
2. **Diverse 5-Fold `GroupKFold` Probability Ensemble:**
   - Averages out-of-fold probabilities across **`LogisticRegression(C=2.5, class_weight="balanced")`** (high macro-recall sensitivity) and **`LogisticRegression(C=1.5, class_weight=None)`** (calibrated multinomial log-odds).
3. **Out-of-Fold Balanced-Accuracy Prior Power Calibration ($\gamma$):**
   - Directly optimizes Balanced Accuracy (`mean` of per-class recalls) by rescaling OOF probabilities $P(y=c \mid x) / \pi_c^\gamma$ across candidate exponents $\gamma \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ validated cleanly with nested `GroupKFold`.

---

## Local Verification & Build

```powershell
& 'G:\ReplitProjects\enterprise-banking-ds-platform\.venv\Scripts\python.exe' build_baseline_nb.py
```

Generates `kaggle_notebook/baseline_tfidf_stylometry_logreg.ipynb` and runs an offline AST + synthetic grouped-CV + Stylometry feature-union smoke test.
