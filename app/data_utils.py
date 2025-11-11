from __future__ import annotations

import pandas as pd
from typing import List, Dict, Tuple


CSV_COLUMNS = [
    "id",
    "Category",
    "Sub_Category",
    "title/entity_name",
    "questions",
    "answers",
    "additional_info/tags",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Try different encodings to handle various file formats
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"Successfully loaded CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            print(f"Failed to load with {encoding} encoding, trying next...")
            continue
    else:
        raise ValueError(f"Could not load CSV file with any of the tried encodings: {encodings}")
    
    # Ensure expected columns exist
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def build_context_text(row: pd.Series) -> str:
    parts: List[str] = []
    if pd.notna(row.get("Category")):
        parts.append(f"Category: {row['Category']}")
    if pd.notna(row.get("Sub_Category")):
        parts.append(f"Sub-Category: {row['Sub_Category']}")
    if pd.notna(row.get("title/entity_name")):
        parts.append(f"Title: {row['title/entity_name']}")
    if pd.notna(row.get("answers")):
        parts.append(f"Answer: {row['answers']}")
    if pd.notna(row.get("additional_info/tags")):
        parts.append(f"Tags: {row['additional_info/tags']}")
    return " \n ".join(parts)


def expand_training_pairs(df: pd.DataFrame, max_pairs: int | None = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        context = build_context_text(row)
        raw_questions = row.get("questions") or ""
        for q in str(raw_questions).splitlines():
            question = q.strip().strip('"')
            if not question:
                continue
            pairs.append((question, context))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def records_with_context(df: pd.DataFrame) -> List[Dict]:
    records: List[Dict] = []
    for _, row in df.iterrows():
        record = row.to_dict()
        record["context_text"] = build_context_text(row)
        records.append(record)
    return records

