"""Dark Taxa Review Queue Database & Logging Engine.

Persistent storage layer for anomalous/flagged eDNA sequences with query
and status-update interfaces for the BioSentinel UI review queue.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Any

# ---------------------------------------------------------------------------
# Database path and table schema
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "dark_taxa_queue.db") if "__file__" in dir() else "dark_taxa_queue.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dark_taxa_queue (
    read_id           TEXT PRIMARY KEY,
    raw_sequence      TEXT NOT NULL,
    sequence_length   INTEGER NOT NULL,
    gc_content        REAL NOT NULL,
    bertax_top_rank   TEXT NOT NULL,
    macro_confidence  REAL NOT NULL,
    entropy_score     REAL NOT NULL,
    anomaly_reason    TEXT NOT NULL,
    review_status     TEXT NOT NULL DEFAULT 'Pending',
    notes             TEXT,
    created_at        TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gc_content(sequence: str) -> float:
    """Calculate GC percentage of a nucleotide sequence, formatted to 2 decimals."""
    if not sequence:
        return 0.0
    clean = sequence.strip().upper()
    total = len(clean)
    gc = clean.count("G") + clean.count("C")
    return round((gc / total) * 100, 2)


def _validate_length(sequence: str, min_len: int = 200, max_len: int = 500) -> bool:
    """Ensure sequence length is within the amplicon specification (200-500 bp)."""
    return min_len <= len(sequence.strip()) <= max_len


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database and create the dark_taxa_queue table if it does not exist.

    Returns an open connection with the schema applied.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Insertion & Logging
# ---------------------------------------------------------------------------

def record_dark_taxon(
    conn: sqlite3.Connection,
    read_id: str,
    router_output: dict,
    raw_sequence: str,
    notes: Optional[str] = None,
) -> bool:
    """Record a dark taxon entry from the TaxonomicRouter output.

    Args:
        conn: Open SQLite connection.
        read_id: Unique identifier for the read.
        router_output: Dictionary from `src.taxonomic_router.route()` containing
            decision, predicted_superkingdom, predicted_phylum, macro_confidence,
            entropy_score, anomaly_reason.
        raw_sequence: The original nucleotide sequence string.
        notes: Optional expert notes.

    Returns:
        True if inserted, False if insert failed (duplicate handled by INSERT OR IGNORE).
    """
    # Compute metrics
    seq_len = len(raw_sequence.strip())
    gc = _gc_content(raw_sequence)
    top_rank = router_output.get("predicted_phylum", "Unknown")
    macro_conf = float(router_output.get("macro_confidence", 0.0))
    entropy = float(router_output.get("entropy_score", 0.0))
    anomaly = router_output.get("anomaly_reason", "Unknown")

    # Validate length (200-500 bp amplicon specification)
    length_ok = _validate_length(raw_sequence)
    # Store actual length; DB column is NOT NULL so we store the length regardless
    seq_len_db = seq_len

    # Use ISO timestamp
    created_at = datetime.now(timezone.utc).isoformat()

    # Use INSERT OR IGNORE for deduplication (read_id is PRIMARY KEY)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO dark_taxa_queue
            (read_id, raw_sequence, sequence_length, gc_content,
             bertax_top_rank, macro_confidence, entropy_score,
             anomaly_reason, review_status, notes, created_at)
        VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?, ?)
        """,
        (
            read_id,
            raw_sequence,
            seq_len_db,
            gc,
            top_rank,
            macro_conf,
            entropy,
            anomaly,
            notes,
            created_at,
        ),
    )
    conn.commit()
    # rowcount is 1 if a new row was inserted, 0 if INSERT OR IGNORE skipped (duplicate)
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Query & Review State Management
# ---------------------------------------------------------------------------

def get_pending_queue(
    conn: sqlite3.Connection,
    limit: int = 50,
    offset: int = 0,
    min_entropy: Optional[float] = None,
) -> List[Tuple[Any, ...]]:
    """Fetch unreviewed (Pending) records for the workbench queue.

    Args:
        conn: Open SQLite connection.
        limit: Maximum number of records to return.
        offset: Number of records to skip (for pagination).
        min_entropy: Optional minimum Shannon entropy filter.

    Returns:
        List of tuple rows matching the query.
    """
    cursor = conn.cursor()
    query = """
        SELECT read_id, raw_sequence, sequence_length, gc_content,
               bertax_top_rank, macro_confidence, entropy_score,
               anomaly_reason, review_status, notes, created_at
        FROM dark_taxa_queue
        WHERE review_status = 'Pending'
    """
    params: List[Any] = []

    if min_entropy is not None:
        query += " AND entropy_score >= ?"
        params.append(min_entropy)

    query += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    return cursor.fetchall()


def get_record_by_id(conn: sqlite3.Connection, read_id: str) -> Optional[Tuple[Any, ...]]:
    """Fetch a single record by read_id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT read_id, raw_sequence, sequence_length, gc_content,
               bertax_top_rank, macro_confidence, entropy_score,
               anomaly_reason, review_status, notes, created_at
        FROM dark_taxa_queue
        WHERE read_id = ?
        """,
        (read_id,),
    )
    row = cursor.fetchone()
    return row


# ---------------------------------------------------------------------------
# Status Update Handler
# ---------------------------------------------------------------------------

VALID_STATUSES = {"Pending", "Approved for BLAST", "Noise", "Validated Lineage"}


def update_taxon_status(
    conn: sqlite3.Connection,
    read_id: str,
    new_status: str,
    expert_notes: Optional[str] = None,
) -> bool:
    """Update the review status of a taxon record.

    Args:
        conn: Open SQLite connection.
        read_id: Read identifier.
        new_status: One of 'Approved for BLAST', 'Noise', 'Validated Lineage'.
        expert_notes: Optional expert commentary.

    Returns:
        True if updated successfully, False if status is invalid or not found.
    """
    if new_status not in VALID_STATUSES:
        return False

    cursor = conn.cursor()
    if expert_notes is not None:
        cursor.execute(
            """
            UPDATE dark_taxa_queue
            SET review_status = ?, notes = ?
            WHERE read_id = ?
            """,
            (new_status, expert_notes, read_id),
        )
    else:
        cursor.execute(
            """
            UPDATE dark_taxa_queue
            SET review_status = ?
            WHERE read_id = ?
            """,
            (new_status, read_id),
        )
    conn.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# FASTA Batch Exporter
# ---------------------------------------------------------------------------

def export_queue_to_fasta(
    conn: sqlite3.Connection,
    filepath: str,
    status_filter: str = "Pending",
) -> int:
    """Dump filtered queue records into standard FASTA format.

    Format: >read_id | reason\nSEQUENCE\n

    Args:
        conn: Open SQLite connection.
        filepath: Output FASTA file path.
        status_filter: Only records with this review_status will be exported.

    Returns:
        Number of records written.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT read_id, raw_sequence, anomaly_reason, review_status
        FROM dark_taxa_queue
        WHERE review_status = ?
        """,
        (status_filter,),
    )
    rows = cursor.fetchall()

    written = 0
    with open(filepath, "w", encoding="utf-8") as fh:
        for read_id, seq, reason, status in rows:
            # Truncate long sequences for readability in FASTA header
            reason_tag = reason or "unknown"
            fh.write(f">{read_id} | {reason_tag}\n")
            fh.write(f"{seq}\n")
            written += 1

    return written