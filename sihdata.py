import gzip
import os
import random
import re
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split

# 25 Target Marine, Deep-Sea, and Pathogenic Taxa across 7 Phyla
TAXONOMY_REGISTRY = [
    # --- 1. ANNELIDA (Hydrothermal Vent & Abyssal Worms) ---
    {
        "taxonomy": "Eukaryota;Annelida;Polychaeta;Sabellida;Siboglinidae;Riftia;Riftia_pachyptila",
        "prefix": "ANN_RIF",
        "seed": "AACCGGTTGCAACGTTAGCTAGCTACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAACCGGTTGCAACGTTAGCTAGCTACGATCG",
    },
    {
        "taxonomy": "Eukaryota;Annelida;Polychaeta;Terebellida;Alvinellidae;Alvinella;Alvinella_pompejana",
        "prefix": "ANN_ALV",
        "seed": "AACCGGTTCCAACGTTAGCTAGCTAGGCTACGTTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAACCGGTTCCAACGTTAGCTAGCTAGGCTACGT",
    },
    {
        "taxonomy": "Eukaryota;Annelida;Polychaeta;Sabellida;Siboglinidae;Oasisia;Oasisia_alvinae",
        "prefix": "ANN_OAS",
        "seed": "AACCGGTTGCAACGTTAGCTAGCTACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTACCCGGTTGCAACGTTAGCTAGCTACGATCGTAGC",
    },
    # --- 2. MOLLUSCA (Vent Bivalves, Squids & Gastropods) ---
    {
        "taxonomy": "Eukaryota;Mollusca;Bivalvia;Mytilida;Mytilidae;Bathymodiolus;Bathymodiolus_thermophilus",
        "prefix": "MOL_BAT",
        "seed": "TTGGCCAACGTTAGCTAGCTAGGCTACGTTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAGGCTACGTTTGGCCAACGTTAGCTAGCTAGGCTACG",
    },
    {
        "taxonomy": "Eukaryota;Mollusca;Cephalopoda;Oegopsida;Architeuthidae;Architeuthis;Architeuthis_dux",
        "prefix": "MOL_ARC",
        "seed": "GGCCAATTACGATCGTAGCTAGCTACGATCGTAGCTAAACCGGTTAGCTAGCTACGATCGTAGCTAGCTAGGCCAATTACGATCGTAGCTAGCTACGAT",
    },
    {
        "taxonomy": "Eukaryota;Mollusca;Bivalvia;Venerida;Vesicomyidae;Calyptogena;Calyptogena_magnifica",
        "prefix": "MOL_CAL",
        "seed": "TTGGCCAACGTTAGCTAGCTACGATCGTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAGGCTACGTTTGGCCAACGTTAGCTAGCTACGATCGTAG",
    },
    {
        "taxonomy": "Eukaryota;Mollusca;Gastropoda;Neogastropoda;Conidae;Conus;Conus_geographus",
        "prefix": "MOL_CON",
        "seed": "TTGGCCAATTACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAGCTAACCGGTTATTGGCCAATTACGATCGTAGCTAGCTAACGTTA",
    },
    # --- 3. ARTHROPODA (Vent Shrimp, Copepods, Krill, Yeti Crab) ---
    {
        "taxonomy": "Eukaryota;Arthropoda;Malacostraca;Decapoda;Alvinocarididae;Rimicaris;Rimicaris_exoculata",
        "prefix": "ART_RIM",
        "seed": "CCGGTTAACGATCGTAGCTAGCTAGGCTAGCTACGTTAGGCTAGCCTTAGGCTACGATCGTAGCTAGGCTCCGGTTAACGATCGTAGCTAGCTAGGCTA",
    },
    {
        "taxonomy": "Eukaryota;Arthropoda;Copepoda;Calanoida;Calanidae;Calanus;Calanus_finmarchicus",
        "prefix": "ART_CAL",
        "seed": "AATTCCGGCTAGCTACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAACCGGTTAGCTAAATTCCGGCTAGCTACGATCGTAGCTAGC",
    },
    {
        "taxonomy": "Eukaryota;Arthropoda;Malacostraca;Euphausiacea;Euphausiidae;Euphausia;Euphausia_superba",
        "prefix": "ART_EUP",
        "seed": "CCGGTTAACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAGCTAACCGGTTAGCTAGCCCCGGTTAACGATCGTAGCTAGCTAACGT",
    },
    {
        "taxonomy": "Eukaryota;Arthropoda;Malacostraca;Decapoda;Kiwaidae;Kiwa;Kiwa_hirsuta",
        "prefix": "ART_KIW",
        "seed": "CCGGTTAACGATCGTAGCTAGCTAGGCTAGCTACGTTAGGCTACGATCGTAGCTAGGCTAGCTACGTTAGCCCGGTTAACGATCGTAGCTAGCTAGGC",
    },
    # --- 4. CNIDARIA (Cold-Water & Deep-Sea Corals) ---
    {
        "taxonomy": "Eukaryota;Cnidaria;Anthozoa;Scleractinia;Caryophylliidae;Lophelia;Lophelia_pertusa",
        "prefix": "CNI_LOP",
        "seed": "TTAACCGGACGTTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAGGCTACGTTAGCTAGCTAGGCTACGTTAACCGGACGTTAGCTAGGCTTACGAT",
    },
    {
        "taxonomy": "Eukaryota;Cnidaria;Anthozoa;Scleralcyonacea;Coralliidae;Corallium;Corallium_rubrum",
        "prefix": "CNI_COR",
        "seed": "TTAACCGGACGTTAGCTAGCTAGGCTACGTTAGGCTAGCCTTAGGCTACGATCGTAGCTAGGCTACGTTTAACCGGACGTTAGCTAGCTAGGCTACGT",
    },
    {
        "taxonomy": "Eukaryota;Cnidaria;Anthozoa;Antipatharia;Antipathidae;Antipathes;Antipathes_dichotoma",
        "prefix": "CNI_ANT",
        "seed": "TTAACCGGACGTTAGCTAGGCTTACGATCGTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAGCTACGTTAACCGGACGTTAGCTAGGCTTACGATC",
    },
    # --- 5. CHORDATA (Demersal, Pelagic & Deep-Sea Fishes) ---
    {
        "taxonomy": "Eukaryota;Chordata;Actinopterygii;Gadiformes;Gadidae;Gadus;Gadus_morhua",
        "prefix": "CHO_GAD",
        "seed": "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACACTACTGGTACATGCTAGCTAGCTAGCACTGGTACATGCTAGCTAGCTAGCTAGCTA",
    },
    {
        "taxonomy": "Eukaryota;Chordata;Actinopterygii;Osmeriformes;Bathylagidae;Bathylagus;Bathylagus_euryops",
        "prefix": "CHO_BAT",
        "seed": "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACTGTACTGGTACATGCTAGCTAGCTAGCACTGGTACATGCTAGCTAGCTAGCTAGCTA",
    },
    {
        "taxonomy": "Eukaryota;Chordata;Actinopterygii;Gadiformes;Macrouridae;Coryphaenoides;Coryphaenoides_armatus",
        "prefix": "CHO_COR",
        "seed": "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACACTAACGTTAGCTAGGCTACGATCGTAACTGGTACATGCTAGCTAGCTAGCTAGCT",
    },
    {
        "taxonomy": "Eukaryota;Chordata;Actinopterygii;Salmoniformes;Salmonidae;Salmo;Salmo_trutta",
        "prefix": "CHO_SAL",
        "seed": "ACTGGTACATGCTAGCTAGCTAGCTAGCTAGCTACGTACGTACTGTACTGGTACATGCTAGCTAGCTACGACTGGTACATGCTAGCTAGCTAGCTAGCT",
    },
    {
        "taxonomy": "Eukaryota;Chordata;Actinopterygii;Cypriniformes;Cyprinidae;Cyprinus;Cyprinus_carpio",
        "prefix": "CHO_CYP",
        "seed": "GCTAGCTAGCTAGCTAGCTACGATCGATCGATCGATCGATCGATGCGCTAGCTAGCTAGCTAGCTACGATGCTAGCTAGCTAGCTAGCTACGATCGATC",
    },
    # --- 6. ECHINODERMATA (Abyssal Sea Cucumbers & Brittle Stars) ---
    {
        "taxonomy": "Eukaryota;Echinodermata;Holothuroidea;Elasipodida;Pelagothuriidae;Enypniastes;Enypniastes_eximia",
        "prefix": "ECH_ENY",
        "seed": "CGTAACGTTAGCTAGCTAGGCTACGTTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAGGCTACGTTAGCCGTAACGTTAGCTAGCTAGGCTACGTTA",
    },
    {
        "taxonomy": "Eukaryota;Echinodermata;Ophiuroidea;Amphiurida;Amphiuridae;Amphiura;Amphiura_filiformis",
        "prefix": "ECH_AMP",
        "seed": "CGTAACGTTAGCTAGCTACGATCGTAGCTAGCTAACGTTAGCTAGGCTACGATCGTAGCTAACCGGTTAGCCGTAACGTTAGCTAGCTACGATCGTAGC",
    },
    {
        "taxonomy": "Eukaryota;Echinodermata;Holothuroidea;Elasipodida;Psychropotidae;Benthodytes;Benthodytes_sanguinolenta",
        "prefix": "ECH_BEN",
        "seed": "CGTAACGTTAGCTAGGCTTACGATCGTAGCTAGGCTAGCTAGGCTACGTTAGCTAGCTAGGCTACGTTAGCCGTAACGTTAGCTAGGCTTACGATCGTA",
    },
    # --- 7. BACTERIA & FUNGI (Pathogens & Biosecurity Targets) ---
    {
        "taxonomy": "Bacteria;Pseudomonadota;Gammaproteobacteria;Vibrionales;Vibrionaceae;Vibrio;Vibrio_parahaemolyticus",
        "prefix": "BAC_VIB",
        "seed": "GATCGAACCGATCGATCGATCGATCGATCGATGCACGTACGTACTGTCGATCGATCGATCGATCGATCGAGATCGAACCGATCGATCGATCGATCGATC",
    },
    {
        "taxonomy": "Bacteria;Pseudomonadota;Gammaproteobacteria;Aeromonadales;Aeromonadaceae;Aeromonas;Aeromonas_hydrophila",
        "prefix": "BAC_AER",
        "seed": "GATCGAACCGATCGATCGATCGATCGATCGATGCACGTACGTACACTCGATCGATCGATCGATCGATCGAGATCGAACCGATCGATCGATCGATCGATC",
    },
    {
        "taxonomy": "Eukaryota;Ascomycota;Sordariomycetes;Hypocreales;Nectriaceae;Fusarium;Fusarium_oxysporum",
        "prefix": "FUN_FUS",
        "seed": "TCGATCGATCGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTATGTCGATCGATCGGCTAGCTAGCTAGCTCGATCGATCGGCTAGCTAGCTAGCTAGC",
    },
]


def mutate_sequence(seed_seq: str, mutation_rate: float = 0.04) -> str:
    """Applies realistic biological nucleotide substitution modeling with a 2:1 Transition-to-Transversion bias."""
    transitions = {"A": "G", "G": "A", "C": "T", "T": "C"}
    transversions = {
        "A": ["C", "T"],
        "G": ["C", "T"],
        "C": ["A", "G"],
        "T": ["A", "G"],
    }

    seq_list = list(seed_seq)
    for i in range(len(seq_list)):
        if random.random() < mutation_rate:
            base = seq_list[i]
            if base in transitions and random.random() < 0.67:
                seq_list[i] = transitions[base]
            elif base in transversions:
                seq_list[i] = random.choice(transversions[base])

    # Simulate variable read lengths (70 to 100 bp)
    target_len = random.randint(70, min(100, len(seq_list)))
    start_pos = random.randint(0, len(seq_list) - target_len)
    return "".join(seq_list[start_pos : start_pos + target_len])


def parse_taxonomy_header(header: str) -> dict:
    ranks = {
        "superkingdom": "Unknown",
        "phylum": "Unknown",
        "class": "Unknown",
        "order": "Unknown",
        "family": "Unknown",
        "genus": "Unknown",
        "species": "Unknown",
    }
    parts = [p.strip() for p in re.split(r"[;,|]", header) if p.strip()]
    clean_parts = [p for p in parts if not p.startswith(">") and p != "root"]
    rank_keys = [
        "superkingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    for i, key in enumerate(rank_keys):
        if i < len(clean_parts):
            ranks[key] = clean_parts[i].replace("_", " ")
    return ranks


def generate_and_export_dataset(
    fasta_path: str = "sample_edna.fasta",
    csv_path: str = "edna_clean_dataset.csv",
    samples_per_class: int = 60,
):
    random.seed(42)
    records = []
    parsed_rows = []

    print(
        f"[*] Generating 1,500 multi-phyla eDNA reads ({samples_per_class} per class x 25 species)..."
    )

    for taxon_meta in TAXONOMY_REGISTRY:
        seed = taxon_meta["seed"]
        tax_str = taxon_meta["taxonomy"]
        tax_info = parse_taxonomy_header(tax_str)

        for i in range(1, samples_per_class + 1):
            seq_id = f"{taxon_meta['prefix']}_{i:03d}"
            mutated_seq = mutate_sequence(seed)

            # FASTA record
            desc = f"{seq_id};root;{tax_str}"
            rec = SeqRecord(Seq(mutated_seq), id=seq_id, description=desc)
            records.append(rec)

            # CSV structured row
            row = dict(tax_info)
            row["seq_id"] = seq_id
            row["sequence"] = mutated_seq
            parsed_rows.append(row)

    SeqIO.write(records, fasta_path, "fasta")
    print(f"[✓] Generated FASTA reference: {fasta_path} ({len(records)} reads)")

    df = pd.DataFrame(parsed_rows)

    # 80/20 Stratified train/test split across species
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["species"], random_state=42
    )

    df["split"] = "train"
    df.loc[test_df.index, "split"] = "test"
    df.to_csv(csv_path, index=False)
    print(f"[✓] Saved cleaned stratified dataset: {csv_path}")

    # Summary table
    print("\n--- Dataset Summary (Phylum Breakdown) ---")
    summary = df.groupby(["phylum", "split"]).size().unstack(fill_value=0)
    print(summary.to_string())


if __name__ == "__main__":
    generate_and_export_dataset()