"""
eDNA species classifier training pipeline.

1. Load & validate the eDNA dataset (sequence, species columns).
2. Extract mean-pooled nucleotide-transformer embeddings, cached to disk and
   auto-invalidated if the source CSV changes.
3. Compare several classifier heads (Logistic Regression, Random Forest, MLP,
   and RBF-SVM on smaller datasets) with stratified cross-validation and
   randomized hyperparameter search; keep the best one.
4. Evaluate the selected model on a held-out test split: accuracy, top-3
   accuracy, macro-F1, a per-class report, and a confusion matrix.
5. Save the classifier + label encoder (same filenames as before, so the
   Streamlit app keeps working) plus a classification report and confusion
   matrix image for inspection.
"""

import contextlib
import hashlib
import json
import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.svm import SVC
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- CONFIGURATION ---
DATASET_CSV = "edna_clean_dataset.csv"
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
CLASSIFIER_OUT = "edna_classifier.joblib"
ENCODER_OUT = "species_encoder.joblib"
CACHE_X = "edna_embeddings_X.npy"
CACHE_Y = "edna_labels_y.npy"
CACHE_META = "edna_embeddings_meta.json"
REPORT_OUT = "classification_report.txt"
CONFUSION_MATRIX_OUT = "confusion_matrix.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42
SEARCH_ITER = 15
MIN_SAMPLES_PER_CLASS = 2


# --- 0. DATA LOADING & VALIDATION ---
def load_and_validate_data() -> pd.DataFrame:
    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(
            f"Could not find '{DATASET_CSV}'. Update DATASET_CSV or place the file "
            f"next to this script."
        )
    df = pd.read_csv(DATASET_CSV)
    for col in ("sequence", "species"):
        if col not in df.columns:
            raise ValueError(f"Expected a '{col}' column in {DATASET_CSV}.")

    before = len(df)
    df = df.dropna(subset=["sequence", "species"]).copy()
    df["sequence"] = (
        df["sequence"].astype(str).apply(lambda s: s.strip().upper())
    )
    df = df[df["sequence"].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        print(
            f"[!] Dropped {dropped} rows with missing/empty sequence or species."
        )

    counts = df["species"].value_counts()
    rare = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(rare):
        print(
            f"[!] Dropping {len(rare)} species with < {MIN_SAMPLES_PER_CLASS} samples "
            f"(too few to split into train/test): {list(rare.index)}"
        )
        df = df[~df["species"].isin(rare.index)]

    seq_len = df["sequence"].str.len()
    print(
        f"[i] {len(df)} sequences across {df['species'].nunique()} species. "
        f"Length (nt): min={seq_len.min()}, mean={seq_len.mean():.0f}, max={seq_len.max()}"
    )
    if seq_len.max() > MAX_LENGTH * 5:
        print(
            f"[!] Longest sequence is {seq_len.max()} nt; with MAX_LENGTH={MAX_LENGTH} tokens "
            f"(~{MAX_LENGTH * 5}-{MAX_LENGTH * 6} nt for this tokenizer) some sequences may be "
            f"truncated. Consider raising MAX_LENGTH if that matters for your marker."
        )
    print(
        f"[i] Samples per species: min={counts.min()}, median={int(counts.median())}, "
        f"max={counts.max()}"
    )
    return df.reset_index(drop=True)


# --- 1. EMBEDDING EXTRACTION WITH DISK CACHING (auto-invalidated) ---
def _fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(
        df[["sequence", "species"]], index=False
    ).to_numpy()
    return hashlib.sha256(payload.tobytes()).hexdigest()


def extract_and_cache_embeddings(df: pd.DataFrame):
    fingerprint = _fingerprint(df)
    cache_exists = os.path.exists(CACHE_X) and os.path.exists(CACHE_Y)

    if cache_exists:
        stale = False
        has_meta = os.path.exists(CACHE_META)
        if has_meta:
            with open(CACHE_META) as f:
                meta = json.load(f)
            stale = meta.get("fingerprint") != fingerprint
        if not stale:
            note = (
                ""
                if has_meta
                else " [cache predates fingerprint tracking - trusting it once]"
            )
            print(
                f"[✓] Loading cached embeddings from disk ({CACHE_X}){note}..."
            )
            X = np.load(CACHE_X)
            y_labels = np.load(CACHE_Y, allow_pickle=True)
            if not has_meta:
                with open(CACHE_META, "w") as f:
                    json.dump(
                        {"fingerprint": fingerprint, "shape": list(X.shape)}, f
                    )
            return X, y_labels
        print("[!] Dataset changed since embeddings were cached — recomputing.")

    print(
        f"[*] Extracting embeddings with {MODEL_NAME.split('/')[-1]} on {DEVICE.upper()}..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if DEVICE == "cuda"
        else contextlib.nullcontext()
    )

    sequences = df["sequence"].tolist()
    all_embeddings = []

    batch_starts = range(0, len(sequences), BATCH_SIZE)
    for i in tqdm(
        batch_starts, total=len(batch_starts), desc="Embedding batches"
    ):
        batch = sequences[i : i + BATCH_SIZE]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad(), autocast_ctx:
            outputs = model(**tokens, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            summed = (last_hidden_state * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            all_embeddings.append(mean_pooled.float().cpu().numpy())

    X = np.vstack(all_embeddings)
    y_labels = np.asarray(df["species"].to_numpy(), dtype=str)

    np.save(CACHE_X, X)
    np.save(CACHE_Y, y_labels)
    with open(CACHE_META, "w") as f:
        json.dump({"fingerprint": fingerprint, "shape": list(X.shape)}, f)
    print(f"[✓] Extracted & cached feature matrix: {X.shape}")
    return X, y_labels


# --- 2. MODEL SELECTION ACROSS SEVERAL CLASSIFIER FAMILIES ---
def build_candidate_models(n_train: int):
    candidates = {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {"C": loguniform(1e-3, 1e2)},
        ),
        "RandomForest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
            ),
            {
                "n_estimators": randint(200, 600),
                "max_depth": [None, 10, 20, 30],
                "min_samples_leaf": randint(1, 5),
            },
        ),
        "MLP": (
            MLPClassifier(
                max_iter=500, early_stopping=True, random_state=RANDOM_STATE
            ),
            {
                "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
                "alpha": loguniform(1e-4, 1e-1),
                "learning_rate_init": loguniform(1e-4, 1e-2),
            },
        ),
    }
    if n_train <= 4000:
        candidates["SVM (RBF)"] = (
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {"C": loguniform(1e-1, 1e2), "gamma": loguniform(1e-4, 1e-1)},
        )
    return candidates


def select_best_model(X_train, y_train, cv_splits: int):
    cv = StratifiedKFold(
        n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE
    )
    candidates = build_candidate_models(len(X_train))

    print(
        f"\n[*] Model selection with {cv_splits}-fold stratified CV (scoring: macro-F1)..."
    )
    results = {}
    for name, (estimator, param_dist) in candidates.items():
        search = RandomizedSearchCV(
            estimator,
            param_dist,
            n_iter=SEARCH_ITER,
            cv=cv,
            scoring="f1_macro",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        results[name] = search
        print(
            f"    {name:<16s} macro-F1 = {search.best_score_:.4f}  (best params: {search.best_params_})"
        )

    best_name = max(results, key=lambda k: results[k].best_score_)
    print(
        f"[✓] Selected {best_name} (macro-F1 = {results[best_name].best_score_:.4f})"
    )
    return results[best_name].best_estimator_, best_name


# --- 3. DIAGNOSTICS ---
def diagnose_fit(train_acc: float, test_acc: float) -> None:
    gap = (train_acc - test_acc) * 100
    if train_acc < 0.85 and gap >= 15:
        verdict = (
            "weak on both counts — training accuracy itself is low *and* there's a large "
            "train/test gap. Check per-class distributions and sequence length validity."
        )
    elif train_acc < 0.85:
        verdict = "underfitting — model lacks capacity or data requires higher feature resolution."
    elif gap >= 15:
        verdict = "overfitting — training accuracy is significantly higher than test accuracy."
    elif gap < 10:
        verdict = "reasonably well-balanced."
    else:
        verdict = "borderline variance — acceptable for multi-class biology."
    print(f" • Train/test gap:            {gap:.2f}%  ->  {verdict}")


# --- 4. TRAIN, EVALUATE, REPORT ---
def train_and_evaluate():
    df = load_and_validate_data()
    X_raw, y_raw = extract_and_cache_embeddings(df)
    X = normalize(X_raw, norm="l2")

    encoder = LabelEncoder()
    y = np.asarray(encoder.fit_transform(y_raw))

    min_class_count = int(df["species"].value_counts().min())
    cv_splits = max(2, min(5, min_class_count))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    best_model, best_name = select_best_model(X_train, y_train, cv_splits)

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))
    k = min(3, y_test_proba.shape[1])
    top3_acc = float(
        top_k_accuracy_score(
            y_test, y_test_proba, k=k, labels=best_model.classes_
        )
    )
    macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))

    print("\n" + "=" * 55)
    print(f"   MODEL PERFORMANCE METRICS — {best_name}")
    print("=" * 55)
    print(f" • Training Accuracy:        {train_acc * 100:.2f}%")
    print(f" • Test Accuracy (Top-1):    {test_acc * 100:.2f}%")
    print(f" • Test Accuracy (Top-{k}):    {top3_acc * 100:.2f}%")
    print(f" • Test Macro-F1:            {macro_f1 * 100:.2f}%")
    diagnose_fit(train_acc, test_acc)
    print("=" * 55)

    present_labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = encoder.inverse_transform(present_labels)

    report = str(
        classification_report(
            y_test,
            y_test_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        )
    )
    with open(REPORT_OUT, "w") as f:
        f.write(report)
    print(f"[✓] Wrote per-class report to `{REPORT_OUT}`")

    cm = confusion_matrix(y_test, y_test_pred, labels=present_labels)
    side = max(8, len(present_labels) * 0.4)
    fig, ax = plt.subplots(figsize=(side, side))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=target_names
    ).plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_OUT, dpi=150)
    plt.close(fig)
    print(f"[✓] Wrote confusion matrix to `{CONFUSION_MATRIX_OUT}`")

    joblib.dump(best_model, CLASSIFIER_OUT)
    joblib.dump(encoder, ENCODER_OUT)
    print(f"[✓] Serialized model to: `{CLASSIFIER_OUT}`")
    print(f"[✓] Serialized label encoder to: `{ENCODER_OUT}`")


if __name__ == "__main__":
    train_and_evaluate()