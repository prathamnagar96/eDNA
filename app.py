import io
import os
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
import streamlit as st
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Page configuration
st.set_page_config(
    page_title="BioSentinel — eDNA Intelligence Engine",
    page_icon="🧬",
    layout="wide",
)

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_PATH = "edna_classifier.joblib"
ENCODER_PATH = "species_encoder.joblib"
QUEUE_FILE = "dark_taxa_review_queue.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading Genomic Transformer & Classifier...")
def load_pipeline_components():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    clf = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return tokenizer, model, clf, encoder


def extract_embeddings(
    sequences: list[str], tokenizer, model, max_length: int = 128
) -> np.ndarray:
    tokens = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        mean_pooled = (last_hidden_state * attention_mask).sum(
            dim=1
        ) / attention_mask.sum(dim=1)

    return mean_pooled.cpu().numpy()


tokenizer, model, clf, encoder = load_pipeline_components()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🧬 BioSentinel")
st.sidebar.caption("MoES SIH-25042 • eDNA Classification Engine")

selected_role = st.sidebar.radio(
    "Select Perspective",
    [
        "🔬 Researcher Workbench",
        "🌾 Farmer / Field Advisory",
        "⚠️ Pathogen & Biosecurity Radar",
    ],
)

CONFIDENCE_THRESHOLD = st.sidebar.slider(
    "Dark Taxa / Novelty Gate", 0.50, 0.95, 0.70, 0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Backbone:** `{MODEL_NAME.split('/')[-1]}`\n\n**Device:** `{DEVICE.upper()}`"
)


# --- INFERENCE ENGINE ---
def run_dna_inference(seq_list: list[str], id_list: list[str] | None = None):
    if not id_list:
        id_list = [f"READ_{i+1:03d}" for i in range(len(seq_list))]

    embeddings = extract_embeddings(seq_list, tokenizer, model)
    probabilities = clf.predict_proba(embeddings)
    predicted_indices = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    predicted_species = encoder.inverse_transform(predicted_indices)

    results = []
    for seq_id, seq, species, conf, probs in zip(
        id_list, seq_list, predicted_species, confidences, probabilities
    ):
        is_dark = conf < CONFIDENCE_THRESHOLD
        results.append(
            {
                "Read ID": seq_id,
                "Sequence": seq,
                "Display Seq": seq[:30] + "..." if len(seq) > 30 else seq,
                "Class Call": "Novel Lineage / Uncatalogued"
                if is_dark
                else species,
                "Predicted Taxa": species,
                "Confidence": conf,
                "Status": "🚩 Review Required"
                if is_dark
                else "✅ High Confidence",
                "Probabilities": {
                    cls: float(p) for cls, p in zip(encoder.classes_, probs)
                },
            }
        )
    return results


# ==========================================
# 1. RESEARCHER WORKBENCH
# ==========================================
if "🔬 Researcher" in selected_role:
    st.title("🔬 Researcher & Authority Workbench")
    st.caption(
        "Direct FASTQ/FASTA ingestion, deep embedding classification, and active dark-taxa review loop."
    )

    tab1, tab2 = st.tabs(["Sequence Ingestion", "📋 Dark Taxa Review Queue"])

    with tab1:
        sample_seq = st.text_area(
            "Raw Nucleotide Read (DNA)",
            value="ATGCGTACGTTAGCCTAGGCTAGCTAGGCTACGTTAGCTAGGCTTACGATCGTAGCTAGCTAGGCTAGCTACGTTAGGCTAGCCTTAGGCTACGATCGTAGCTAGGCTAGCTAGGCTACGTTAGCTAGC",
            height=80,
        )

        if st.button("Run Genomic Pipeline", type="primary"):
            cleaned = sample_seq.strip().upper()
            res = run_dna_inference([cleaned], ["AMBIGUOUS_PROBE_01"])[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Call", res["Class Call"])
            col2.metric(
                "Confidence Score", f"{res['Confidence'] * 100:.2f}%"
            )
            col3.metric("Taxonomic Status", res["Status"])

            st.subheader("Softmax Class Probability Distribution")
            prob_df = pd.DataFrame(
                list(res["Probabilities"].items()),
                columns=["Taxon", "Softmax Probability"],
            )
            st.bar_chart(prob_df.set_index("Taxon"))

            # Interactive Human-in-the-Loop Review Box
            if "🚩" in res["Status"]:
                st.markdown("---")
                st.warning("### ⚠️ Ambiguous Read Intercepted")
                st.markdown(
                    "This sequence falls below the confidence threshold. Assign a provisional taxonomic classification to submit it to the retraining loop."
                )

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    expert_label = st.text_input(
                        "Provisional Linnaean Binomial / Clade Name",
                        value="Bathymodiolus aff. thermophilus",
                    )
                with col_b:
                    st.write("")
                    st.write("")
                    if st.button("✅ Commit to Retraining Queue"):
                        new_row = pd.DataFrame(
                            [
                                {
                                    "seq_id": res["Read ID"],
                                    "sequence": res["Sequence"],
                                    "original_model_call": res[
                                        "Predicted Taxa"
                                    ],
                                    "confidence": round(res["Confidence"], 4),
                                    "expert_assigned_taxa": expert_label,
                                    "timestamp": pd.Timestamp.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                            ]
                        )
                        if os.path.exists(QUEUE_FILE):
                            new_row.to_csv(
                                QUEUE_FILE,
                                mode="a",
                                header=False,
                                index=False,
                            )
                        else:
                            new_row.to_csv(QUEUE_FILE, index=False)
                        st.success(
                            f"Successfully appended {res['Read ID']} to `{QUEUE_FILE}`! Model active-learning dataset updated."
                        )

    with tab2:
        st.subheader("Human-in-the-Loop Retraining Queue")
        if os.path.exists(QUEUE_FILE):
            queue_df = pd.read_csv(QUEUE_FILE)
            st.dataframe(queue_df, use_container_width=True)
            st.download_button(
                "⬇️ Export Retraining Batch (CSV)",
                data=queue_df.to_csv(index=False),
                file_name="dark_taxa_retraining_set.csv",
                mime="text/csv",
            )
        else:
            st.info(
                "No novel taxa currently queued for review. Flagged reads will appear here automatically."
            )

# ==========================================
# 2. FARMER PORTAL
# ==========================================
elif "🌾 Farmer" in selected_role:
    st.title("🌾 Field Health Advisory & Early Warning")
    st.caption(
        "Automated eDNA soil/water pathogen screening with plain-language advisories."
    )

    test_preset = st.selectbox(
        "Select Soil Sample Probe Location",
        [
            "Plot A — Coastal Paddy (Fusarium Test)",
            "Plot B — Freshwater Aquaculture Pond (Carp Test)",
            "Plot C — Coldwater Stream Inlet (Salmon Test)",
        ],
    )
    preset_map = {
        "Plot A — Coastal Paddy (Fusarium Test)": "TCGATCGATCGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTATGT",
        "Plot B — Freshwater Aquaculture Pond (Carp Test)": "GCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGATCGATGC",
        "Plot C — Coldwater Stream Inlet (Salmon Test)": "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACTGT",
    }

    if st.button("Analyze Field Sample", type="primary"):
        seq = preset_map[test_preset]
        res = run_dna_inference([seq], [test_preset])[0]

        if "Fusarium" in res["Predicted Taxa"]:
            st.error("🔴 **High Risk Detected: Root Rot Pathogen (Fusarium)**")
            st.markdown(
                f"**Confidence:** `{res['Confidence']*100:.1f}%` | **Action Required**"
            )
            st.markdown("""
            **3-Step Advisory:**
            1. **Soil Drainage:** Eliminate standing water in affected sectors.
            2. **Bio-Control:** Apply *Trichoderma viride* formulation within 48h.
            3. **Quarantine:** Restrict movement of farm machinery across plots.
            """)
        else:
            st.success(
                f"🟢 **Field Status Healthy:** Indicator organism detected (`{res['Predicted Taxa']}`)."
            )

# ==========================================
# 3. REGIONAL PATHOGEN RADAR
# ==========================================
elif "⚠️ Pathogen" in selected_role:
    st.title("⚠️ Regional Pathogen & Biodiversity Radar")
    st.caption(
        "Geospatial threat monitoring and automated alert dispatch interface."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Monitored Sites", "14 Active Hubs", "+2 this week")
    col2.metric("Pathogen Alerts", "1 Active (Fusarium)", "Requires Action")
    col3.metric("Biodiversity Index", "0.84", "Stable")

    st.subheader("Live Environmental Monitoring Feed")
    demo_samples = [
        ("GB_MAR_SAL_001", "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACTGT"),
        ("GB_MAR_CYP_001", "GCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGATCGATGC"),
        ("GB_AGR_FUS_001", "TCGATCGATCGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTATGT"),
    ]
    radar_res = run_dna_inference(
        [s[1] for s in demo_samples], [s[0] for s in demo_samples]
    )
    radar_df = pd.DataFrame(radar_res).drop(
        columns=["Probabilities", "Sequence"]
    )
    st.dataframe(radar_df, use_container_width=True)

    if st.button("Dispatch Automated Alerts to District Officers"):
        st.success("✅ Broadcast dispatched to 12 extension officers.")