"""
Local label extraction from radiology reports via OpenCode Go API.

Processes all 4,349 report-only studies using DeepSeek V4 Pro.
Concurrent processing (4 workers). Checkpoints every 10 successful reports.

Gold-study extraction uses a separate checkpoint and never enters the silver
training artifact. API failures are deliberately not checkpointed, so a later
``--resume`` retries them instead of silently treating them as negative labels.

Usage:
    source .venv/bin/activate
    python notebooks/02_label_extraction_local.py [--test N] [--resume]
    python notebooks/02_label_extraction_local.py --gold-only --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels.extractor import build_extraction_prompt, build_soft_extraction_prompt

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("input/extraction.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
PRIMARY_MODEL = "deepseek-v4-pro"
FALLBACK_MODEL = "deepseek-v4-flash"
LAST_RESORT_MODEL = "kimi-k3"
MAX_WORKERS = 4
MAX_COMPLETION_TOKENS = 6000

VARIANT_PATHS = {
    "hard-v1": {
        "checkpoint": Path("input/extraction_checkpoint.json"),
        "gold_checkpoint": Path("input/gold_extraction_checkpoint.json"),
        "silver": Path("input/silver_labels.csv"),
        "detailed": Path("input/extraction_detailed.csv"),
        "gold_detailed": Path("input/gold_extraction_detailed.csv"),
        "eval": Path("input/label_extraction_eval.json"),
        "errors": Path("input/extraction_errors.csv"),
        "gold_errors": Path("input/gold_extraction_errors.csv"),
    },
    "soft-v2": {
        "checkpoint": Path("input/extraction_soft_v2_checkpoint.json"),
        "gold_checkpoint": Path("input/gold_extraction_soft_v2_checkpoint.json"),
        "silver": Path("input/silver_labels_soft_v2.csv"),
        "detailed": Path("input/extraction_soft_v2_detailed.csv"),
        "gold_detailed": Path("input/gold_extraction_soft_v2_detailed.csv"),
        "eval": Path("input/label_extraction_soft_v2_eval.json"),
        "errors": Path("input/extraction_soft_v2_errors.csv"),
        "gold_errors": Path("input/gold_extraction_soft_v2_errors.csv"),
    },
}

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SILENCE_INFORMATIVE = {
    "Baker's": 0.1,
    "Effusion": 0.3,
    "Fracture": 0.2,
}


def parse_llm_response(response_text: str, require_score: bool = False) -> dict:
    """Parse LLM JSON response."""
    import re
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else response_text

    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end]

    data = json.loads(json_str)

    valid_values = {"present", "absent", "uncertain", "not_addressed", "laterality_ambiguous"}
    result = {}
    for label in LABEL_COLS:
        entry = data.get(label, "not_addressed")
        evidence = ""
        if isinstance(entry, dict):
            value = entry.get("value", "not_addressed")
            evidence = entry.get("evidence", "")
            score = entry.get("score")
        else:
            value = entry
            score = None
        if value not in valid_values:
            value = "not_addressed"
        if not isinstance(evidence, str):
            evidence = ""
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"Invalid score type for {label}: {type(score).__name__}")
            score = float(score)
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Score out of range for {label}: {score}")
        elif require_score:
            raise ValueError(f"Missing score for {label}")
        result[label] = {"value": value, "evidence": evidence}
        if score is not None:
            result[label]["score"] = score

    return result


def call_llm(client, prompt: str, retries: int = 2) -> tuple:
    """Call LLM with retry and model fallback. Returns (response_text, model_used)."""
    models = [PRIMARY_MODEL, FALLBACK_MODEL, LAST_RESORT_MODEL]

    for m in models:
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    # Kimi K3 rejects every temperature except 1. DeepSeek is
                    # deterministic at 0 for this extraction task.
                    temperature=1.0 if m == LAST_RESORT_MODEL else 0.0,
                    # Reasoning models count hidden reasoning against this
                    # budget; 2,000 produced HTTP 200 with empty visible output.
                    max_tokens=MAX_COMPLETION_TOKENS,
                )
                choice = response.choices[0]
                text = choice.message.content
                if text and text.strip():
                    return text, m
                reasoning = getattr(choice.message, "reasoning_content", "") or ""
                logger.warning(
                    f"  {m} attempt {attempt+1}/{retries}: empty content "
                    f"(finish_reason={choice.finish_reason}, reasoning_chars={len(reasoning)})"
                )
            except Exception as e:
                logger.warning(f"  {m} attempt {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)

    raise RuntimeError(f"All models failed: {models}")


def process_one(client, study_uid: str, report: str, prompt_builder, require_score: bool) -> tuple:
    """Process a single report. Returns (study_uid, result_dict, model_used, error)."""
    if not report or pd.isna(report):
        empty = {
            label: {"value": "not_addressed", "evidence": "", "score": 0.5}
            if require_score else "not_addressed"
            for label in LABEL_COLS
        }
        return study_uid, empty, "none", None

    prompt = prompt_builder(report)
    try:
        response_text, model_used = call_llm(client, prompt)
        parsed = parse_llm_response(response_text, require_score=require_score)
        return study_uid, parsed, model_used, None
    except Exception as e:
        return study_uid, None, "error", str(e)


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict, path: Path):
    """Atomically save a checkpoint so interruption cannot truncate it."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(results, f)
    os.replace(temp_path, path)


def extracted_value(extraction: dict, label: str) -> str:
    entry = extraction.get(label, "not_addressed")
    if isinstance(entry, dict):
        return entry.get("value", "not_addressed")
    return entry


def is_all_not_addressed(extraction: dict) -> bool:
    return all(extracted_value(extraction, label) == "not_addressed" for label in LABEL_COLS)


def value_to_soft_label(value: str, label: str) -> float:
    if value == "present":
        return 1.0
    elif value == "absent":
        return 0.0
    elif value == "uncertain":
        return 0.5
    elif value == "not_addressed":
        return SILENCE_INFORMATIVE.get(label, 0.5)
    elif value == "laterality_ambiguous":
        return 0.5
    return 0.5


def extracted_score(extraction: dict, label: str) -> float:
    entry = extraction.get(label, "not_addressed")
    if isinstance(entry, dict) and isinstance(entry.get("score"), (int, float)):
        score = float(entry["score"])
        if np.isfinite(score) and 0.0 <= score <= 1.0:
            return score
    return value_to_soft_label(extracted_value(extraction, label), label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, help="Test mode: process only N reports")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Concurrent workers")
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_PATHS),
        default="hard-v1",
        help="Versioned extraction/scoring strategy with isolated artifacts",
    )
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="Extract/evaluate the 58 gold studies using a separate checkpoint",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Rebuild CSV/evaluation artifacts from an existing checkpoint without API calls",
    )
    parser.add_argument(
        "--retry-all-not-addressed",
        action="store_true",
        help="Reprocess checkpoint entries whose 12 findings are all not_addressed",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds (OpenAI SDK retries are disabled)",
    )
    args = parser.parse_args()

    # Load data
    logger.info("Loading train.csv...")
    train_csv = pd.read_csv("input/train.csv")

    gold_mask = train_csv[LABEL_COLS].notna().any(axis=1)
    gold_uids = set(train_csv.loc[gold_mask, "StudyInstanceUID"].tolist())
    logger.info(f"Total: {len(train_csv)}, Gold: {len(gold_uids)}, Silver: {len(train_csv) - len(gold_uids)}")

    paths = VARIANT_PATHS[args.variant]
    checkpoint_file = paths["gold_checkpoint"] if args.gold_only else paths["checkpoint"]
    detailed_output = paths["gold_detailed"] if args.gold_only else paths["detailed"]
    errors_output = paths["gold_errors"] if args.gold_only else paths["errors"]
    silver_output = paths["silver"]
    eval_output = paths["eval"]
    require_score = args.variant == "soft-v2"
    prompt_builder = build_soft_extraction_prompt if require_score else build_extraction_prompt

    client = None
    if not args.assemble_only:
        # Initialize client only for an extraction run. Artifact assembly remains
        # available offline and does not require API credentials.
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENCODE_API_KEY"],
            base_url=OPENCODE_BASE_URL,
            timeout=args.request_timeout,
            max_retries=0,
        )
        logger.info(f"OpenCode Go client initialized (workers={args.workers})")

    # Load checkpoint
    results = load_checkpoint(checkpoint_file) if args.resume or args.assemble_only else {}
    if results:
        logger.info(f"Resuming from checkpoint: {len(results)} studies already processed")

    retry_uids = set()
    if args.retry_all_not_addressed:
        retry_uids = {
            uid for uid, extraction in results.items() if is_all_not_addressed(extraction)
        }
        logger.info(f"Queued {len(retry_uids)} all-not-addressed entries for clean reprocessing")

    # Build work queue
    work_items = []
    if not args.assemble_only:
        candidate_rows = train_csv[gold_mask] if args.gold_only else train_csv[~gold_mask]
        for _, row in candidate_rows.iterrows():
            uid = row["StudyInstanceUID"]
            if uid in results and uid not in retry_uids:
                continue
            work_items.append((uid, row.get("Report", "")))

    if args.test > 0:
        work_items = work_items[:args.test]

    # Only remove retry entries that are actually in this run. This preserves
    # unselected checkpoint rows when --test limits the work queue.
    for uid, _ in work_items:
        if uid in retry_uids:
            del results[uid]

    total_work = len(work_items)
    logger.info(f"Work queue: {total_work} reports to process")

    errors = []
    model_usage = {PRIMARY_MODEL: 0, FALLBACK_MODEL: 0, LAST_RESORT_MODEL: 0, "error": 0}

    if total_work == 0:
        logger.info("Nothing to do.")
    else:
        # Process with thread pool
        t0 = time.time()
        completed = 0
        succeeded = 0
        last_checkpoint_succeeded = 0
        lock = Lock()

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_one, client, uid, report, prompt_builder, require_score
                ): uid
                for uid, report in work_items
            }

            for future in as_completed(futures):
                uid = futures[future]
                try:
                    study_uid, parsed, model_used, error = future.result()
                    with lock:
                        completed += 1
                        if error:
                            errors.append({"study_uid": study_uid, "error": error})
                            model_usage["error"] += 1
                        else:
                            results[study_uid] = parsed
                            succeeded += 1
                            if model_used in model_usage:
                                model_usage[model_used] += 1
                except Exception as e:
                    with lock:
                        errors.append({"study_uid": uid, "error": str(e)})
                        model_usage["error"] += 1
                        completed += 1

                # Checkpoint successful results frequently. Failed UIDs remain
                # absent and will be retried by a later --resume invocation.
                if succeeded > 0 and succeeded % 10 == 0 and succeeded != last_checkpoint_succeeded:
                    with lock:
                        save_checkpoint(results, checkpoint_file)
                    last_checkpoint_succeeded = succeeded
                    elapsed = time.time() - t0
                    rate = completed / max(elapsed, 1)
                    remaining = (total_work - completed) / max(rate, 0.001)
                    logger.info(
                        f"  Progress: {completed}/{total_work} "
                        f"({succeeded} saved, {len(errors)} errors, ~{remaining/60:.0f} min left)"
                    )

        # Final checkpoint
        save_checkpoint(results, checkpoint_file)
        elapsed = time.time() - t0
        logger.info(f"\nExtraction complete: {completed} processed, {len(errors)} errors")
        logger.info(f"Time: {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
        logger.info(f"Model usage: {model_usage}")

    if errors:
        pd.DataFrame(errors).to_csv(errors_output, index=False)
        logger.warning(
            f"{len(errors)} API failures were not checkpointed; rerun with --resume to retry them"
        )
    elif errors_output.exists():
        errors_output.unlink()

    # ── Assemble silver labels ───────────────────────────────────────
    if not args.gold_only:
        logger.info("Assembling silver labels...")

        rows = []
        for study_uid, extraction in results.items():
            if study_uid in gold_uids:
                continue
            row = {"StudyInstanceUID": study_uid}
            for label in LABEL_COLS:
                row[label] = extracted_score(extraction, label)
            rows.append(row)

        silver_df = pd.DataFrame(rows)
        for label in LABEL_COLS:
            if label not in silver_df.columns:
                silver_df[label] = 0.5
        silver_df = silver_df[["StudyInstanceUID"] + LABEL_COLS]
        silver_df.to_csv(silver_output, index=False)
        logger.info(f"Silver labels saved: {len(silver_df)} studies -> {silver_output}")

    # ── Evaluate against gold ────────────────────────────────────────
    logger.info("Evaluating against 58 gold studies...")
    gold_df = train_csv[gold_mask][["StudyInstanceUID"] + LABEL_COLS].copy()

    # Gold extraction is held in a separate checkpoint and is never included in
    # the silver training matrix.
    gold_results = results if args.gold_only else load_checkpoint(paths["gold_checkpoint"])

    if gold_results:
        from sklearn.metrics import roc_auc_score
        evaluated_gold = len(gold_uids.intersection(gold_results))
        eval_results = {
            "gold_studies_evaluated": evaluated_gold,
            "gold_studies_total": len(gold_uids),
        }
        aucs = []

        if evaluated_gold < len(gold_uids):
            logger.warning(
                f"Partial gold evaluation: {evaluated_gold}/{len(gold_uids)} studies; "
                "AUCs may be undefined or unstable"
            )

        for label in LABEL_COLS:
            y_true = []
            y_score = []
            for _, gold_row in gold_df.iterrows():
                uid = gold_row["StudyInstanceUID"]
                if uid not in gold_results:
                    continue
                gold_val = gold_row[label]
                if pd.isna(gold_val):
                    continue
                y_true.append(float(gold_val))
                y_score.append(extracted_score(gold_results[uid], label))

            if len(set(y_true)) < 2:
                continue
            auc = roc_auc_score(y_true, y_score)
            eval_results[label] = round(auc, 4)
            aucs.append(auc)
            logger.info(f"  {label}: {auc:.4f}")

        macro = float(np.mean(aucs)) if aucs else None
        eval_results["macro_auc"] = round(macro, 4) if macro is not None else None
        if macro is None:
            logger.warning("Macro AUC vs gold is undefined until both classes are represented")
        else:
            logger.info(f"\nMacro AUC vs gold: {macro:.4f} (target: >= 0.87)")

        with open(eval_output, "w") as f:
            json.dump(eval_results, f, indent=2)
    else:
        logger.warning("No gold extraction available — run with --gold-only")

    # Save detailed results
    detailed = []
    for study_uid, extraction in results.items():
        for label in LABEL_COLS:
            entry = extraction.get(label, "not_addressed")
            if isinstance(entry, dict):
                value = entry.get("value", "not_addressed")
                evidence = entry.get("evidence", "")
                score = entry.get("score")
            else:
                value = entry
                evidence = ""
                score = None
            detailed.append({
                "StudyInstanceUID": study_uid,
                "finding": label,
                "value": value,
                "score": score,
                "evidence": evidence,
            })
    pd.DataFrame(detailed).to_csv(detailed_output, index=False)
    logger.info(f"Detailed results saved -> {detailed_output}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
