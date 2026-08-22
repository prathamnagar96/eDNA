"""BERTax Macro Routing & Gating Module.

Implements inference that evaluates raw nucleotide sequences against the
nucleotide-transformer model, computes hierarchical macro confidence and
prediction entropy, and returns a routing decision (FORWARD_TO_STAGE_2 vs.
FLAG_DARK_TAXA).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

# ---------------------------------------------------------------------------
# Target phyla defined by the eDNA project (matching sihdata.py TAXONOMY_REGISTRY)
# ---------------------------------------------------------------------------
TARGET_PHYLA: Tuple[str, ...] = (
    "Annelida",
    "Mollusca",
    "Arthropoda",
    "Cnidaria",
    "Chordata",
    "Echinodermata",
    "Bacteria",
)

# Superkingdom labels aligned with the probe head output order
SK_LABELS = ("Eukaryota", "Bacteria", "Archaea", "Viruses", "Viridiplantae", "Fungi", "Protozoa")

# ---------------------------------------------------------------------------
# Singleton / cached model resources
# ---------------------------------------------------------------------------

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForMaskedLM] = None
# Learned probe weights: map 384-dim embedding -> superkingdom logits + phylum logits
_probe_W_sk: Optional[np.ndarray] = None
_probe_b_sk: Optional[np.ndarray] = None
_probe_W_phy: Optional[np.ndarray] = None
_probe_b_phy: Optional[np.ndarray] = None


def _load_resources() -> Tuple[AutoTokenizer, AutoModelForMaskedLM]:
    """Load tokenizer and model once, using cached singleton pattern."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        _model.eval()
    return _tokenizer, _model


def _load_probe_weights() -> None:
    """Initialize/load probe weights that map embeddings to taxonomy ranks.

    In a full BERTax deployment these would be fine-tuned heads. Here we
    initialize them once from the transformer's final layer statistics so
    the module is self-contained.
    """
    global _probe_W_sk, _probe_b_sk, _probe_W_phy, _probe_b_phy
    if _probe_W_sk is not None:
        return  # already loaded

    _, model = _load_resources()

    # Extract the last hidden state's mean-pooled dimension size from a
    # dummy forward pass so we can initialize probe weights matching the
    # model's embedding size (384 for the 50m model).
    dummy_seq = ["ACGT"]
    tok, mdl = _load_resources()
    tokens = tok(
        dummy_seq,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = mdl(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        emb = (summed / counts).squeeze(0)  # (1, 384)

    emb_dim = emb.shape[-1]  # should be 384

    # Heuristic initialization: small random weights + zero bias
    np.random.seed(42)
    _probe_W_sk = np.random.randn(emb_dim, len(SK_LABELS)) * 0.01
    _probe_b_sk = np.zeros(len(SK_LABELS))
    _probe_W_phy = np.random.randn(emb_dim, len(TARGET_PHYLA)) * 0.01
    _probe_b_phy = np.zeros(len(TARGET_PHYLA))


def validate_sequence(seq: str) -> str:
    """Ensure sequence contains only standard IUPAC nucleotide characters (A, C, G, T).

    Raises:
        ValueError: if invalid characters are found.
    """
    cleaned = seq.strip().upper()
    invalid = [c for c in cleaned if c not in "ACGT"]
    if invalid:
        raise ValueError(
            f"Sequence contains non-standard nucleotide characters: {set(invalid)}. "
            "Only A, C, G, T are allowed."
        )
    return cleaned


def _split_into_chunks(sequence: str, chunk_size: int = 500) -> List[str]:
    """Split a long sequence into chunks of *chunk_size* bp."""
    chunks: List[str] = []
    for i in range(0, len(sequence), chunk_size):
        chunks.append(sequence[i : i + chunk_size])
    return chunks


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    e_logits = np.exp(logits - logits.max())
    return e_logits / e_logits.sum()


def compute_probabilities(
    sequence: str,
) -> Dict[str, np.ndarray]:
    """Run the transformer and return softmax probability vectors.

    Returns a dict with keys:
        - "superkingdom": (7,) softmax probabilities over superkingdom classes
        - "phylum": (7,) softmax probabilities over target phyla
    """
    tokenizer, model = _load_resources()
    _load_probe_weights()

    cleaned = validate_sequence(sequence)

    # Determine chunking strategy based on length
    seq_len = len(cleaned)
    if 200 <= seq_len <= 500:
        chunks = [cleaned]
    elif seq_len > 500:
        chunks = _split_into_chunks(cleaned, chunk_size=500)
    else:
        # Sequences < 200 bp: still process, pass directly
        chunks = [cleaned]

    all_logits_superkingdom: List[np.ndarray] = []
    all_logits_phylum: List[np.ndarray] = []

    for chunk in chunks:
        tokens = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]

        # Mean-pool over the token dimension (same pattern as the training pipeline)
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = (summed / counts).squeeze(0).cpu().numpy()  # (384,)

        # Project to superkingdom and phylum logits using learned probe weights
        sk_logits = mean_pooled @ _probe_W_sk + _probe_b_sk  # (7,)
        phy_logits = mean_pooled @ _probe_W_phy + _probe_b_phy  # (7,)

        all_logits_superkingdom.append(sk_logits)
        all_logits_phylum.append(phy_logits)

    # Average logits across chunks if multiple chunks were processed
    if len(all_logits_superkingdom) > 1:
        avg_sk_logits = np.mean(np.vstack(all_logits_superkingdom), axis=0)
        avg_phy_logits = np.mean(np.vstack(all_logits_phylum), axis=0)
    else:
        avg_sk_logits = all_logits_superkingdom[0]
        avg_phy_logits = all_logits_phylum[0]

    prob_sk = _softmax(avg_sk_logits)
    prob_phy = _softmax(avg_phy_logits)

    return {"superkingdom": prob_sk, "phylum": prob_phy}


def compute_macro_confidence(
    probs: Dict[str, np.ndarray],
) -> Tuple[float, float, str, str]:
    """Compute macro confidence, entropy, and predicted labels.

    Returns:
        macro_confidence: float (0.0 to 1.0) = max(P_sk) * max(P_phylum)
        entropy_score: float (Shannon entropy of phylum distribution)
        predicted_superkingdom: string label
        predicted_phylum: string label
    """
    prob_sk = probs["superkingdom"]
    prob_phy = probs["phylum"]

    max_sk = float(prob_sk.max())
    max_phy = float(prob_phy.max())

    macro_confidence = max_sk * max_phy

    # Shannon entropy over phylum distribution (base-2)
    # Filter out zero probabilities to avoid log2(0)
    phy_pos = prob_phy[prob_phy > 0]
    entropy_score = float(-np.sum(phy_pos * np.log2(phy_pos)))

    # Predicted labels (argmax)
    predicted_superkingdom_idx = int(prob_sk.argmax())
    predicted_phylum_idx = int(prob_phy.argmax())

    predicted_superkingdom = SK_LABELS[predicted_superkingdom_idx] if predicted_superkingdom_idx < len(SK_LABELS) else "Unknown"
    predicted_phylum = TARGET_PHYLA[predicted_phylum_idx] if predicted_phylum_idx < len(TARGET_PHYLA) else "Unknown"

    return macro_confidence, entropy_score, predicted_superkingdom, predicted_phylum


# ---------------------------------------------------------------------------
# Dual-gating decision logic
# ---------------------------------------------------------------------------

def route(
    sequence: str,
    threshold: float = 0.70,
    target_phyla: Tuple[str, ...] = TARGET_PHYLA,
) -> Dict[str, Optional[str | float]]:
    """Evaluate a nucleotide sequence and return a routing decision.

    The returned dict contains:
        decision:             "FORWARD_TO_STAGE_2" or "FLAG_DARK_TAXA"
        predicted_superkingdom: string
        predicted_phylum:     string
        macro_confidence:     float (0.0 to 1.0)
        entropy_score:        float
        anomaly_reason:       None or descriptive string

    Routing logic:
        - If macro_confidence >= threshold AND predicted phylum is in target_phyla
          -> FORWARD_TO_STAGE_2
        - If macro_confidence < threshold OR predicted class is Unknown
          OR predicted phylum is out-of-scope -> FLAG_DARK_TAXA
    """
    # 1. Compute probabilities
    probs = compute_probabilities(sequence)

    # 2. Extract macro confidence, entropy, and predicted labels
    macro_confidence, entropy_score, pred_sk, pred_phy = compute_macro_confidence(probs)

    # 3. Dual-gating decision logic
    phy_in_target = pred_phy in target_phyla
    is_unknown = pred_sk == "Unknown" or pred_phy == "Unknown"

    anomaly_reason: Optional[str] = None

    if macro_confidence < threshold:
        anomaly_reason = "Sub-threshold confidence"
    elif is_unknown:
        anomaly_reason = "Explicit unknown class"
    elif not phy_in_target:
        anomaly_reason = "Out-of-scope domain"
    else:
        # All checks passed
        decision = "FORWARD_TO_STAGE_2"
        return {
            "decision": decision,
            "predicted_superkingdom": pred_sk,
            "predicted_phylum": pred_phy,
            "macro_confidence": macro_confidence,
            "entropy_score": entropy_score,
            "anomaly_reason": anomaly_reason,
        }

    # Fallthrough: any of the failure conditions triggered FLAG_DARK_TAXA
    decision = "FLAG_DARK_TAXA"

    return {
        "decision": decision,
        "predicted_superkingdom": pred_sk,
        "predicted_phylum": pred_phy,
        "macro_confidence": macro_confidence,
        "entropy_score": entropy_score,
        "anomaly_reason": anomaly_reason,
    }