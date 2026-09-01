"""
Label engine v6 — multi-extractor gold-58 evaluation.

Runs the soft extraction prompt across several diverse OpenCode Go API models
on the 58 gold studies, evaluates per-label AUC vs the expert labels, and
reports fusion candidates (mean / per-class oracle). Per the repo contract this
is dev-time label infrastructure: gold extraction results NEVER enter the
silver training matrix; they only measure extractor quality.

Artifact namespace: .codex_work/label_engine_v6/
    gold_<slug>_checkpoint.json  — resumable per-model gold extractions
    gold_<slug>_eval.json        — per-model gold AUC
    multi_extractor_eval.json    — summary incl. fusion probes
    REPORT.md                    — human summary

Usage:
    .venv/bin/python notebooks/11_label_engine_v6_eval.py \
        --models gpt-5.6-luna,kimi-k3,glm-5.3,deepseek-v4-pro,qwen3.8-max,grok-4.5
    (add --resume to continue a partial run)
"""

import argparse
import json
import logging
import os
import re
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

from src.labels.extractor import build_soft_extraction_prompt

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("label_engine_v6")

OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
OUT_DIR = PROJECT_ROOT / ".codex_work" / "label_engine_v6"

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

# Models that reject temperature=0 (per notebooks/02_label_extraction_local.py).
TEMP_ONE_MODELS = {"kimi-k3"}
MAX_COMPLETION_TOKENS = 8000


def slugify(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


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


def call_model(client, model: str, prompt: str, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0 if model in TEMP_ONE_MODELS else 0.0,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                # Newer OpenAI reasoning models require max_completion_tokens
                # and only support the default temperature.
                msg = str(e)
                if any(
                    s in msg
                    for s in ("max_tokens", "max_completion_tokens", "temperature")
                ):
                    kwargs.pop("temperature", None)
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            choice = response.choices[0]
            text = choice.message.content
            if text and text.strip():
                return text
            logger.warning(
                f"  {model} attempt {attempt+1}: empty content "
                f"(finish_reason={choice.finish_reason})"
            )
        except Exception as e:
            last_err = e
            logger.warning(f"  {model} attempt {attempt+1}: {str(e)[:200]}")
        time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"model {model} failed after {retries} attempts: {last_err}")


def run_model_on_gold(client, model: str, gold_rows, results: dict, workers: int) -> list:
    """Extract the gold studies with one model. Mutates `results` (checkpoint).
    Returns list of error strings."""
    work = [(uid, report) for uid, report in gold_rows if uid not in results]
    if not work:
        logger.info(f"[{model}] nothing to do ({len(results)} already done)")
        return []

    errors = []
    lock = Lock()
    t0 = time.time()
    done = 0

    def one(item):
        uid, report = item
        if not report or pd.isna(report):
            return uid, {label: {"value": "not_addressed", "evidence": "", "score": 0.5}
                         for label in LABEL_COLS}, None
        try:
            text = call_model(client, model, build_soft_extraction_prompt(report))
            return uid, parse_llm_response(text), None
        except Exception as e:
            return uid, None, str(e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, item): item[0] for item in work}
        for future in as_completed(futures):
            uid, parsed, err = future.result()
            with lock:
                done += 1
                if err is None:
                    results[uid] = parsed
                else:
                    errors.append(f"{uid}: {err}")
                if done % 10 == 0 or done == len(work):
                    rate = done / max(time.time() - t0, 1)
                    logger.info(
                        f"[{model}] {done}/{len(work)} "
                        f"({len(errors)} errors, ~{(len(work)-done)/max(rate,0.01)/60:.0f} min left)"
                    )
    return errors


def eval_against_gold(gold_df, gold_uids, results: dict) -> dict:
    from sklearn.metrics import roc_auc_score
    out = {}
    aucs = {}
    for label in LABEL_COLS:
        y_true, y_score = [], []
        for _, row in gold_df.iterrows():
            uid = row["StudyInstanceUID"]
            if uid not in results:
                continue
            gold_val = row[label]
            if pd.isna(gold_val):
                continue
            y_true.append(float(gold_val))
            y_score.append(extracted_score(results[uid], label))
        if len(set(y_true)) < 2:
            out[label] = None
            continue
        auc = roc_auc_score(y_true, y_score)
        aucs[label] = auc
        out[label] = round(auc, 4)
    out["macro_auc"] = round(float(np.mean(list(aucs.values()))), 4) if aucs else None
    out["n_studies"] = len(gold_uids.intersection(results))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="gpt-5.6-luna,kimi-k3,glm-5.3,deepseek-v4-pro,qwen3.8-max,grok-4.5",
        help="Comma-separated model ids",
    )
    parser.add_argument(
        "--api",
        choices=["opencode", "openai"],
        default="opencode",
        help="API backend: opencode (Go API) or openai (api.openai.com)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(PROJECT_ROOT / "input" / "train.csv")
    gold_mask = train_csv[LABEL_COLS].notna().any(axis=1)
    gold_df = train_csv[gold_mask]
    gold_rows = [
        (row["StudyInstanceUID"], row.get("Report", ""))
        for _, row in gold_df.iterrows()
    ]
    gold_uids = set(r[0] for r in gold_rows)
    logger.info(f"Gold studies: {len(gold_rows)} | api: {args.api} | models: {models}")

    from openai import OpenAI
    if args.api == "openai":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            timeout=args.request_timeout,
            max_retries=0,
        )
    else:
        client = OpenAI(
            api_key=os.environ["OPENCODE_API_KEY"],
            base_url=OPENCODE_BASE_URL,
            timeout=args.request_timeout,
            max_retries=0,
        )

    per_model_results = {}
    per_model_eval = {}
    for model in models:
        slug = slugify(model)
        ckpt_path = OUT_DIR / f"gold_{slug}_checkpoint.json"
        results = {}
        if args.resume and ckpt_path.exists():
            results = json.load(open(ckpt_path))
            logger.info(f"[{model}] resumed {len(results)} from checkpoint")

        errors = run_model_on_gold(client, model, gold_rows, results, args.workers)

        tmp = ckpt_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, ckpt_path)
        logger.info(f"[{model}] checkpoint saved: {len(results)} studies -> {ckpt_path}")

        ev = eval_against_gold(gold_df, gold_uids, results)
        per_model_results[model] = results
        per_model_eval[model] = ev
        with open(OUT_DIR / f"gold_{slug}_eval.json", "w") as f:
            json.dump(ev, f, indent=2)
        logger.info(f"[{model}] MACRO AUC = {ev['macro_auc']} ({len(errors)} errors)")
        for label in LABEL_COLS:
            logger.info(f"    {label}: {ev[label]}")

    # ── Fusion probes ─────────────────────────────────────────────────
    from sklearn.metrics import roc_auc_score

    def score_matrix(results_by_model):
        uids = sorted(gold_uids.intersection(*[set(r) for r in results_by_model.values()]))
        mat = np.zeros((len(uids), len(LABEL_COLS), len(results_by_model)))
        for k, (model, res) in enumerate(results_by_model.items()):
            for i, uid in enumerate(uids):
                for j, label in enumerate(LABEL_COLS):
                    mat[i, j, k] = extracted_score(res[uid], label)
        y = np.zeros((len(uids), len(LABEL_COLS)))
        uid_row = {row["StudyInstanceUID"]: row for _, row in gold_df.iterrows()}
        for i, uid in enumerate(uids):
            for j, label in enumerate(LABEL_COLS):
                y[i, j] = float(uid_row[uid][label])
        return uids, mat, y

    fusion = {}
    if per_model_results:
        uids, mat, y = score_matrix(per_model_results)
        model_names = list(per_model_results)

        mean_scores = mat.mean(axis=2)
        per_label_mean = {}
        vals = []
        for j, label in enumerate(LABEL_COLS):
            if len(set(y[:, j])) < 2:
                per_label_mean[label] = None
                continue
            auc = roc_auc_score(y[:, j], mean_scores[:, j])
            per_label_mean[label] = round(auc, 4)
            vals.append(auc)
        fusion["mean_all"] = {
            "per_label": per_label_mean,
            "macro_auc": round(float(np.mean(vals)), 4) if vals else None,
        }

        # Greedy forward model selection on macro gold AUC (order-invariant
        # reported as "best_subset" — overfit-prone, direction only).
        best = (None, -1)
        import itertools
        for r in range(1, len(model_names) + 1):
            for combo in itertools.combinations(range(len(model_names)), r):
                m = mat[:, :, list(combo)].mean(axis=2)
                vals = []
                for j in range(len(LABEL_COLS)):
                    if len(set(y[:, j])) < 2:
                        continue
                    vals.append(roc_auc_score(y[:, j], m[:, j]))
                macro = float(np.mean(vals))
                if macro > best[1]:
                    best = ([model_names[i] for i in combo], macro)
        fusion["best_subset"] = {"models": best[0], "macro_auc": round(best[1], 4)}

    summary = {
        "n_gold": len(gold_uids),
        "models": {m: per_model_eval[m] for m in models},
        "fusion": fusion,
    }
    with open(OUT_DIR / "multi_extractor_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY (gold-58 macro AUC)")
    for m in models:
        logger.info(f"  {m:24s} {per_model_eval[m]['macro_auc']}")
    if fusion:
        logger.info(f"  {'MEAN of all models':24s} {fusion['mean_all']['macro_auc']}")
        logger.info(
            f"  {'BEST SUBSET':24s} {fusion['best_subset']['macro_auc']} "
            f"({', '.join(fusion['best_subset']['models'])})"
        )
    logger.info(f"Artifacts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
