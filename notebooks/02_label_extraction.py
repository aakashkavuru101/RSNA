"""
Label extraction from radiology reports using LLM API.

Run on Kaggle with internet ON (needs OpenCode Go API access).
Produces: silver_labels.csv — the training label matrix.

Primary model: DeepSeek V4 Pro (deepseek-v4-pro)
Fallback:      DeepSeek V4 Flash (deepseek-v4-flash) — faster, cheaper
Last resort:   Kimi K3 (kimi-k3) — use rarely, consumption is faster

Prerequisites:
- Run 01_eda.py first (produces train_parsed.csv)
- Set OPENCODE_API_KEY in Kaggle Secrets (starts with sk-)

Usage (Kaggle notebook):
    exec(open("notebooks/02_label_extraction.py").read())
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
WORK_DIR = Path("/kaggle/working")
OUTPUT_DIR = WORK_DIR

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# ── OpenCode Go configuration ────────────────────────────────────────
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
PRIMARY_MODEL = "deepseek-v4-pro"
FALLBACK_MODEL = "deepseek-v4-flash"
LAST_RESORT_MODEL = "kimi-k3"  # use rarely — consumption is faster

# ── Load data ────────────────────────────────────────────────────────
logger.info("Loading parsed train data...")
train_csv = pd.read_csv(WORK_DIR / "train_parsed.csv")
gold_labels = pd.read_csv(WORK_DIR / "gold_labels.csv")
gold_uids = set(gold_labels["StudyInstanceUID"].tolist())

logger.info(f"Total studies: {len(train_csv)}")
logger.info(f"Gold studies (excluded from silver): {len(gold_uids)}")

# ── LLM extraction ──────────────────────────────────────────────────
logger.info("Setting up LLM extraction via OpenCode Go...")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels.extractor import (
    ExtractionResult, FindingValue, LabelExtractor, LABELS,
    build_extraction_prompt,
)
from src.labels.assembler import assemble_silver_labels, evaluate_against_gold

extractor = LabelExtractor(method="llm")

# Get API key from environment or Kaggle Secrets
api_key = os.environ.get("OPENCODE_API_KEY")
if not api_key:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        api_key = secrets.get_secret("OPENCODE_API_KEY")
    except Exception:
        pass

if not api_key:
    logger.warning("No OPENCODE_API_KEY found. Will use rule-based fallback.")
    client = None
else:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=OPENCODE_BASE_URL,
    )
    logger.info(f"OpenCode Go client initialized (base_url={OPENCODE_BASE_URL})")


def call_llm(prompt: str, model: str, retries: int = 2) -> str:
    """
    Call LLM with retry logic and model fallback chain.

    Tries: primary → flash → kimi (last resort).
    """
    models_to_try = [model]

    # Build fallback chain
    if model == PRIMARY_MODEL:
        models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    elif model == FALLBACK_MODEL:
        models_to_try = [FALLBACK_MODEL, LAST_RESORT_MODEL]

    for attempt_model in models_to_try:
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=attempt_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000,
                )
                if attempt_model != model:
                    logger.info(f"  Fallback: used {attempt_model} instead of {model}")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(
                    f"  {attempt_model} attempt {attempt + 1}/{retries} failed: {e}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff

    raise RuntimeError(f"All models failed for prompt (tried {models_to_try})")


# ── Extract labels ───────────────────────────────────────────────────
logger.info("Extracting labels from reports...")
logger.info(f"Primary: {PRIMARY_MODEL} | Fallback: {FALLBACK_MODEL} | Last resort: {LAST_RESORT_MODEL}")

extraction_results = {}
errors = []
model_usage = {PRIMARY_MODEL: 0, FALLBACK_MODEL: 0, LAST_RESORT_MODEL: 0, "rules": 0}
batch_size = 50

for idx, row in train_csv.iterrows():
    study_uid = row["StudyInstanceUID"]

    # Skip gold studies — they must not be trained on
    if study_uid in gold_uids:
        continue

    report = row.get("Report", "")
    if not report or pd.isna(report):
        extraction_results[study_uid] = [
            ExtractionResult(finding=label, value=FindingValue.NOT_ADDRESSED)
            for label in LABELS
        ]
        continue

    language = row.get("language", "unknown")

    try:
        if client:
            prompt = build_extraction_prompt(report, language)
            response_text = call_llm(prompt, PRIMARY_MODEL)
            results = extractor.extract_from_llm_response(response_text)
            model_usage[PRIMARY_MODEL] += 1  # approximate; fallback tracked in logs
        else:
            results = extractor.extract_rules(report, language)
            model_usage["rules"] += 1

        extraction_results[study_uid] = results

    except Exception as e:
        logger.error(f"Error extracting {study_uid}: {e}")
        errors.append({"study_uid": study_uid, "error": str(e)})
        extraction_results[study_uid] = [
            ExtractionResult(finding=label, value=FindingValue.NOT_ADDRESSED)
            for label in LABELS
        ]

    if (idx + 1) % batch_size == 0:
        logger.info(
            f"  Processed {idx + 1}/{len(train_csv)} reports "
            f"({len(errors)} errors)"
        )
        time.sleep(0.2)  # rate limiting — be respectful to the API

logger.info(f"Extraction complete: {len(extraction_results)} studies, {len(errors)} errors")
logger.info(f"Model usage: {model_usage}")

# ── Assemble silver labels ───────────────────────────────────────────
logger.info("Assembling silver label matrix...")
silver_df = assemble_silver_labels(extraction_results, gold_study_uids=list(gold_uids))

# ── Evaluate against gold ────────────────────────────────────────────
logger.info("Evaluating silver labels against 58 gold studies...")
gold_eval = gold_labels.copy()

merged = gold_eval.merge(
    silver_df,
    on="StudyInstanceUID",
    suffixes=("_gold", "_silver"),
)

from sklearn.metrics import roc_auc_score

eval_results = {}
aucs = []
for label in LABELS:
    gold_col = f"{label}_gold"
    silver_col = f"{label}_silver"

    if gold_col not in merged.columns or silver_col not in merged.columns:
        continue

    y_true = merged[gold_col].dropna()
    y_score = merged.loc[y_true.index, silver_col].dropna()
    y_true = y_true.loc[y_score.index]

    if len(y_true.unique()) < 2:
        continue

    auc = roc_auc_score(y_true, y_score)
    eval_results[label] = auc
    aucs.append(auc)
    logger.info(f"  {label}: {auc:.4f}")

macro_auc = np.mean(aucs) if aucs else 0.0
eval_results["macro_auc"] = macro_auc
logger.info(f"\nMacro AUC vs gold: {macro_auc:.4f}")
logger.info(f"(Target: ≥ 0.87 for good label quality)")

# ── Save outputs ─────────────────────────────────────────────────────
silver_df.to_csv(OUTPUT_DIR / "silver_labels.csv", index=False)
logger.info(f"Silver labels saved: {len(silver_df)} studies")

eval_results["model_usage"] = model_usage
with open(OUTPUT_DIR / "label_extraction_eval.json", "w") as f:
    json.dump(eval_results, f, indent=2)

if errors:
    pd.DataFrame(errors).to_csv(OUTPUT_DIR / "extraction_errors.csv", index=False)

# Detailed extraction results for audit
detailed = []
for study_uid, results in extraction_results.items():
    for r in results:
        detailed.append({
            "StudyInstanceUID": study_uid,
            "finding": r.finding,
            "value": r.value.value,
            "evidence": r.evidence,
        })
pd.DataFrame(detailed).to_csv(OUTPUT_DIR / "extraction_detailed.csv", index=False)

logger.info("\nLabel extraction complete.")
logger.info("Next: run 03_preprocess.py to cache DICOM volumes")
