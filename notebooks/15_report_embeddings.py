"""Report → multilingual text embeddings for multimodal training.

Encodes all 4,407 training reports with a multilingual sentence encoder and
writes a CSV (StudyInstanceUID + embedding dims) + a numpy matrix. These feed
the InfoNCE image↔report alignment objective in the multimodal training kernel
(the reports' full medical semantics beyond the 12 extracted binary labels).

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim,
multilingual, CPU-fast). Output artifacts (local, uploaded to Kaggle as the
dataset `rsna-knee-report-embeddings`):
    .codex_work/label_engine_v6/report_embeddings.csv
    .codex_work/label_engine_v6/report_embeddings.npy

Usage:
    .venv/bin/python notebooks/15_report_embeddings.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / ".codex_work" / "label_engine_v6" / "report_embeddings.csv"
OUT_NPY = PROJECT_ROOT / ".codex_work" / "label_engine_v6" / "report_embeddings.npy"

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, device=device)

    train = pd.read_csv(PROJECT_ROOT / "input" / "train.csv")
    texts = train["Report"].fillna("").astype(str).tolist()
    uids = train["StudyInstanceUID"].astype(str).tolist()
    print(f"encoding {len(texts)} reports on {device}")

    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print("embeddings:", embs.shape)

    cols = [f"d{i}" for i in range(embs.shape[1])]
    df = pd.DataFrame(embs, columns=cols)
    df.insert(0, "StudyInstanceUID", uids)
    df.to_csv(OUT_CSV, index=False)
    np.save(OUT_NPY, embs)
    print(f"wrote {OUT_CSV} and {OUT_NPY}")


if __name__ == "__main__":
    main()
