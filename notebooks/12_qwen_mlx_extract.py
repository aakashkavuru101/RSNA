"""
Local open-weight label extraction via MLX (Apple Silicon) — label engine v6.

Runs the soft extraction prompt through a local Qwen model served by MLX.
Two phases:
  1. --gold-only : extract the 58 gold studies, evaluate AUC vs expert labels
  2. --corpus    : extract all 4,349 report-only studies (overnight run)

Artifacts (.codex_work/label_engine_v6/):
    qwen_<slug>_gold_checkpoint.json / qwen_<slug>_corpus_checkpoint.json
    qwen_<slug>_gold_eval.json

Resumable: rerun with --resume; failed reports are retried.
Usage:
    caffeinate -s .venv/bin/python notebooks/12_qwen_mlx_extract.py --gold-only
    caffeinate -s .venv/bin/python notebooks/12_qwen_mlx_extract.py --corpus --resume
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels.extractor import build_soft_extraction_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("qwen_mlx")

OUT_DIR = PROJECT_ROOT / ".codex_work" / "label_engine_v6"
MODEL_ID = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SILENCE_INFORMATIVE = {"Baker's": 0.1, "Effusion": 0.3, "Fracture": 0.2}
MAX_TOKENS = 2048


def slugify(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


def parse_llm_response(response_text: str) -> dict:
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
        score = None
        if isinstance(entry, dict):
            value = entry.get("value", "not_addressed")
            evidence = entry.get("evidence", "") or ""
            score = entry.get("score")
        else:
            value = entry
        if value not in valid_values:
            value = "not_addressed"
        if not isinstance(evidence, str):
            evidence = ""
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"Invalid score type for {label}")
            score = float(score)
            if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Score out of range for {label}: {score}")
        else:
            raise ValueError(f"Missing score for {label}")
        result[label] = {"value": value, "evidence": evidence, "score": score}
    return result


def value_to_soft_label(value: str, label: str) -> float:
    mapping = {
        "present": 1.0,
        "absent": 0.0,
        "uncertain": 0.5,
        "laterality_ambiguous": 0.5,
        "not_addressed": SILENCE_INFORMATIVE.get(label, 0.5),
    }
    return mapping.get(value, 0.5)


def extracted_score(extraction: dict, label: str) -> float:
    entry = extraction.get(label)
    if isinstance(entry, dict):
        score = entry.get("score")
        if isinstance(score, (int, float)) and np.isfinite(score) and 0.0 <= score <= 1.0:
            return float(score)
        return value_to_soft_label(entry.get("value", "not_addressed"), label)
    return value_to_soft_label(str(entry), label)


def save_checkpoint(results: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--corpus", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N reports (smoke)")
    args = parser.parse_args()
    if not (args.gold_only or args.corpus):
        parser.error("choose --gold-only or --corpus")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.model)
    phase = "gold" if args.gold_only else "corpus"
    ckpt_path = OUT_DIR / f"qwen_{slug}_{phase}_checkpoint.json"

    train_csv = pd.read_csv(PROJECT_ROOT / "input" / "train.csv")
    gold_mask = train_csv[LABEL_COLS].notna().any(axis=1)
    if args.gold_only:
        rows = train_csv[gold_mask]
    else:
        rows = train_csv[~gold_mask]
    work = [
        (row["StudyInstanceUID"], row.get("Report", ""))
        for _, row in rows.iterrows()
    ]

    results = {}
    if args.resume and ckpt_path.exists():
        results = json.load(open(ckpt_path))
        logger.info(f"resumed {len(results)} studies from {ckpt_path}")
    work = [(uid, rep) for uid, rep in work if uid not in results]
    if args.limit:
        work = work[: args.limit]
    logger.info(f"phase={phase} model={args.model} queue={len(work)}")

    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    greedy = make_sampler(temp=0.0)

    def chat(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        out = generate(
            model, tokenizer, prompt=text, max_tokens=MAX_TOKENS,
            sampler=greedy, verbose=False,
        )
        return out

    t0 = time.time()
    errors = 0
    for i, (uid, report) in enumerate(work):
        if not report or pd.isna(report):
            results[uid] = {
                label: {"value": "not_addressed", "evidence": "", "score": 0.5}
                for label in LABEL_COLS
            }
        else:
            ok = False
            for attempt in range(2):
                try:
                    text = chat(build_soft_extraction_prompt(report))
                    results[uid] = parse_llm_response(text)
                    ok = True
                    break
                except Exception as e:
                    logger.warning(f"  {uid} attempt {attempt+1} failed: {str(e)[:150]}")
            if not ok:
                errors += 1
                continue  # not checkpointed; retried on resume

        if (i + 1) % 5 == 0 or i + 1 == len(work):
            save_checkpoint(results, ckpt_path)
            rate = (i + 1) / max(time.time() - t0, 1)
            eta_min = (len(work) - i - 1) / max(rate, 0.01) / 60
            logger.info(
                f"  {i+1}/{len(work)} saved={len(results)} errors={errors} "
                f"rate={rate*3600:.0f}/h eta={eta_min:.0f}min"
            )

    save_checkpoint(results, ckpt_path)
    logger.info(f"done: {len(results)} saved, {errors} errors -> {ckpt_path}")

    if args.gold_only:
        from sklearn.metrics import roc_auc_score
        gold_df = train_csv[gold_mask]
        ev = {"n_studies": len(results)}
        aucs = []
        for label in LABEL_COLS:
            y_true, y_score = [], []
            for _, row in gold_df.iterrows():
                uid = row["StudyInstanceUID"]
                if uid not in results or pd.isna(row[label]):
                    continue
                y_true.append(float(row[label]))
                y_score.append(extracted_score(results[uid], label))
            if len(set(y_true)) < 2:
                ev[label] = None
                continue
            auc = roc_auc_score(y_true, y_score)
            ev[label] = round(auc, 4)
            aucs.append(auc)
        ev["macro_auc"] = round(float(np.mean(aucs)), 4) if aucs else None
        with open(OUT_DIR / f"qwen_{slug}_gold_eval.json", "w") as f:
            json.dump(ev, f, indent=2)
        logger.info(f"GOLD EVAL: macro={ev['macro_auc']}")
        for label in LABEL_COLS:
            logger.info(f"  {label}: {ev[label]}")


if __name__ == "__main__":
    main()
