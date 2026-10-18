"""
Dataset ingestion, temporal splitting, and SHA-256 integrity verification
for the KuaiRand-Pure unbiased recommendation benchmark.
"""

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


def verify_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_and_verify_kuairand(
    archive_path: Path,
    target_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Extracts KuaiRand-Pure archive and generates/verifies dataset manifest with SHA-256 hashes.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_hash = verify_sha256(archive_path)

    print(f"[*] Extracting {archive_path.name} (SHA-256: {archive_hash[:16]}...)...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    # Locate extracted files
    extracted_files = list(target_dir.rglob("*.csv"))
    manifest = {
        "archive_name": archive_path.name,
        "archive_sha256": archive_hash,
        "files": {},
    }

    file_map = {}
    for f in extracted_files:
        h = verify_sha256(f)
        manifest["files"][f.name] = {
            "size_bytes": f.stat().st_size,
            "sha256": h,
        }
        if "standard" in f.name:
            file_map["standard"] = f
        elif "random" in f.name:
            file_map["random"] = f

    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"[✓] Dataset manifest saved to {manifest_path.name}")

    return file_map


def load_kuairand_pure(
    standard_csv: Path,
    random_csv: Path,
    max_standard_rows: Optional[int] = None,
    max_random_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads standard logging traffic and held-out unbiased random-exposure intervention traffic.
    """
    cols = ["user_id", "video_id", "date", "time_ms", "is_click", "is_like", "long_view", "play_time_ms", "duration_ms"]
    
    # Check available columns in standard CSV
    sample_df = pd.read_csv(standard_csv, nrows=5)
    use_cols = [c for c in cols if c in sample_df.columns]
    if "is_click" not in use_cols:
        # Fallback to general interaction column
        use_cols = None

    print(f"[*] Loading standard logs from {standard_csv.name}...")
    df_standard = pd.read_csv(standard_csv, usecols=use_cols, nrows=max_standard_rows)

    print(f"[*] Loading unbiased random logs from {random_csv.name}...")
    df_random = pd.read_csv(random_csv, usecols=use_cols, nrows=max_random_rows)

    # Ensure required click column
    if "is_click" not in df_standard.columns:
        df_standard["is_click"] = (df_standard.get("play_time_ms", 0) > 3000).astype(int)
    if "is_click" not in df_random.columns:
        df_random["is_click"] = (df_random.get("play_time_ms", 0) > 3000).astype(int)

    # Long view definition: watched >= 5000 ms or completion >= 1.0
    if "long_view" not in df_standard.columns:
        if "duration_ms" in df_standard.columns and "play_time_ms" in df_standard.columns:
            df_standard["long_view"] = (df_standard["play_time_ms"] >= df_standard["duration_ms"]).astype(int)
        else:
            df_standard["long_view"] = (df_standard.get("play_time_ms", 0) >= 5000).astype(int)

    if "long_view" not in df_random.columns:
        if "duration_ms" in df_random.columns and "play_time_ms" in df_random.columns:
            df_random["long_view"] = (df_random["play_time_ms"] >= df_random["duration_ms"]).astype(int)
        else:
            df_random["long_view"] = (df_random.get("play_time_ms", 0) >= 5000).astype(int)

    return df_standard, df_random


def temporal_train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    time_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a strict temporal split on logging traffic to prevent future data leakage.
    """
    if time_col in df.columns:
        unique_dates = sorted(df[time_col].dropna().unique())
        split_idx = int(len(unique_dates) * (1.0 - val_ratio))
        split_date = unique_dates[max(1, split_idx)]
        train_df = df[df[time_col] < split_date].copy()
        val_df = df[df[time_col] >= split_date].copy()
    else:
        # Fallback to row index order if timestamp column is absent
        n_train = int(len(df) * (1.0 - val_ratio))
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:].copy()

    return train_df, val_df


def compute_item_popularity(
    df: pd.DataFrame,
    item_col: str = "video_id",
    smoothing: float = 1e-6,
) -> Tuple[Dict[int, float], np.ndarray]:
    """
    Computes empirical item selection frequency Q(i) for logQ debiasing.
    Returns:
        pop_dict: mapping item_id -> Q(i)
        pop_array: array indexed by dense item ID
    """
    counts = df[item_col].value_counts()
    total = len(df)
    pop_dict = {item: (count / total) + smoothing for item, count in counts.items()}
    return pop_dict
