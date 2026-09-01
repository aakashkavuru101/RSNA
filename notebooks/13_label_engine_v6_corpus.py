"""
Label engine v6 — full-corpus (4,349 report-only) extraction.

Runs the soft extraction prompt over all report-only studies for a given model
and writes a per-model silver-label CSV + checkpoint. Batching optional to cut
input-token cost on priced APIs.

Artifacts (.codex_work/label_engine_v6/):
    corpus_<slug>_checkpoint.json
    corpus_<slug>.csv          (StudyInstanceUID + 12 soft-score columns)

Usage:
    .venv/bin/python notebooks/13_label_engine_v6_corpus.py \
        --api openai --model gpt-4.1-mini --workers 6 [--batch 8] [--resume]
    .venv/bin/python notebooks/13_label_engine_v6_corpus.py \
        --api opencode --model ox-alpha-free --workers 8 --resume
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

from src.labels.extractor import (
    build_soft_extraction_prompt,
    build_soft_batch_extraction_prompt,
)

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("label_engine_v6_corpus")

OPENCODE_BASE_URL = "https://opencode.ai/zen/go/v1"
OUT_DIR = PROJECT_ROOT / ".codex_work" / "label_engine_v6"

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SILENCE_INFORMATIVE = {"Baker's": 0.1, "Effusion": 0.3, "Fracture": 0.2}
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


def parse_batch_response(response_text: str, ids: list) -> dict:
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    json_str = json_match.group(1) if json_match else response_text
    brace_start = json_str.find("{")
    brace_end = json_str.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        json_str = json_str[brace_start:brace_end]
    data = json.loads(json_str)
    reports = data.get("reports", data)
    out = {}
    for uid in ids:
        entry = reports.get(uid)
        if entry is None:
            raise ValueError(f"batch response missing id {uid}")
        parsed = {}
        for label in LABEL_COLS:
            e = entry.get(label, "not_addressed")
            evidence = ""
            score = None
            if isinstance(e, dict):
                value = e.get("value", "not_addressed")
                evidence = e.get("evidence", "") or ""
                score = e.get("score")
            else:
                value = e
            if value not in {"present", "absent", "uncertain", "not_addressed", "laterality_ambiguous"}:
                value = "not_addressed"
            if not isinstance(evidence, str):
                evidence = ""
            if score is not None:
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(f"Invalid score for {label}")
                score = float(score)
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"Score out of range for {label}")
            else:
                raise ValueError(f"Missing score for {label}")
            parsed[label] = {"value": value, "evidence": evidence, "score": score}
        out[uid] = parsed
    return out


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
                msg = str(e)
                if any(s in msg for s in ("max_tokens", "max_completion_tokens", "temperature")):
                    kwargs.pop("temperature", None)
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise
            text = response.choices[0].message.content
            if text and text.strip():
                return text
            logger.warning(f"  {model} attempt {attempt+1}: empty content")
        except Exception as e:
            last_err = e
            logger.warning(f"  {model} attempt {attempt+1}: {str(e)[:160]}")
        time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"{model} failed: {last_err}")


def value_to_soft_label(value: str, label: str) -> float:
    mapping = {
        "present": 1.0, "absent": 0.0, "uncertain": 0.5,
        "laterality_ambiguous": 0.5, "not_addressed": SILENCE_INFORMATIVE.get(label, 0.5),
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
    parser.add_argument("--api", choices=["opencode", "openai"], default="opencode")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch", type=int, default=0, help="Batch N reports per request")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(args.model)
    ckpt_path = OUT_DIR / f"corpus_{slug}_checkpoint.json"
    csv_path = OUT_DIR / f"corpus_{slug}.csv"

    train_csv = pd.read_csv(PROJECT_ROOT / "input" / "train.csv")
    gold_mask = train_csv[LABEL_COLS].notna().any(axis=1)
    silver_rows = train_csv[~gold_mask]
    work = [(row["StudyInstanceUID"], row.get("Report", "")) for _, row in silver_rows.iterrows()]

    results = {}
    if args.resume and ckpt_path.exists():
        results = json.load(open(ckpt_path))
        logger.info(f"resumed {len(results)} studies")
    work = [(uid, rep) for uid, rep in work if uid not in results]
    if args.limit:
        work = work[: args.limit]
    logger.info(f"api={args.api} model={args.model} queue={len(work)} batch={args.batch}")

    from openai import OpenAI
    if args.api == "openai":
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            timeout=args.request_timeout, max_retries=0,
        )
    else:
        client = OpenAI(
            api_key=os.environ["OPENCODE_API_KEY"],
            base_url=OPENCODE_BASE_URL,
            timeout=args.request_timeout, max_retries=0,
        )

    errors = []
    lock = Lock()
    t0 = time.time()
    done = 0

    def process_item(item):
        if args.batch:
            ids, texts = zip(*item)
            return parse_batch_response(
                call_model(client, args.model, build_soft_batch_extraction_prompt(list(item))),
                list(ids),
            ), None
        uid, report = item
        if not report or pd.isna(report):
            return {uid: {label: {"value": "not_addressed", "evidence": "", "score": 0.5}
                          for label in LABEL_COLS}}, None
        try:
            parsed = parse_llm_response(
                call_model(client, args.model, build_soft_extraction_prompt(report)))
            return {uid: parsed}, None
        except Exception as e:
            return None, str(e)

    if args.batch:
        # group work into batches
        items = [work[i:i + args.batch] for i in range(0, len(work), args.batch)]
    else:
        items = work

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_item, it): it for it in items}
        for future in as_completed(futures):
            parsed, err = future.result()
            with lock:
                done += 1
                if err is None:
                    results.update(parsed)
                else:
                    errors.append(err)
                if done % 5 == 0 or done == len(items):
                    save_checkpoint(results, ckpt_path)
                    rate = done / max(time.time() - t0, 1)
                    eta = (len(items) - done) / max(rate, 0.01)
                    logger.info(
                        f"  {done}/{len(items)} saved={len(results)} errors={len(errors)} "
                        f"eta={eta/60:.0f}min"
                    )

    save_checkpoint(results, ckpt_path)
    logger.info(f"done: {len(results)} saved, {len(errors)} errors")

    rows = []
    for uid, extraction in results.items():
        row = {"StudyInstanceUID": uid}
        for label in LABEL_COLS:
            row[label] = extracted_score(extraction, label)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABEL_COLS)
    df.to_csv(csv_path, index=False)
    logger.info(f"silver CSV -> {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
