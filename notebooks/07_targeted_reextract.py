"""Targeted multi-pass self-consistency re-extraction of high-disagreement reports.

For every selected report, each model votes ``k`` independent times at a
nonzero temperature (default 0.7) so votes decorrelate. Votes are aggregated
per (uid, finding) into a vote distribution, a soft score from the assembler's
value mapping, and an agreement (max vote share). Gold reports use separate
``gold_``-prefixed artifacts and are never mixed into silver training data.

Reuses the client/transport/parsing/checkpoint idioms of
``02_label_extraction_batch.py`` via a file-based import; that module is never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing batch extractor without running its CLI; it performs
# load_dotenv and defines the OpenAI transport, parsing, and atomic JSON helpers.
_spec = importlib.util.spec_from_file_location(
    "label_extraction_batch", Path(__file__).resolve().parent / "02_label_extraction_batch.py"
)
_batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_batch)

from src.labels.assembler import SILENCE_INFORMATIVE, evaluate_against_gold
from src.labels.extractor import LABELS, build_soft_batch_extraction_prompt

BASE_URL = _batch.BASE_URL
VALID_VALUES = _batch.VALID_VALUES
atomic_json = _batch.atomic_json
parse_batch_response = _batch.parse_batch_response

MODELS = ("kimi-k3", "deepseek-v4-pro")
# kimi-k3 rejects any temperature except 1.0 (provider-enforced); deepseek
# models accept the requested temperature. All voting temperatures are nonzero
# so the k samples decorrelate.
MODEL_TEMPERATURE_OVERRIDES = {"kimi-k3": 1.0}
# deepseek-v4-pro is a reasoning model: observed ~7.3k reasoning tokens before
# content, so max_tokens must leave headroom beyond the JSON payload.
MAX_TOKENS = 16000
DISAGREEMENT_CASES = PROJECT_ROOT / ".codex_work" / "supervision_audit" / "high_disagreement_cases.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def value_soft_score(value: str, label: str) -> float:
    """Map one categorical vote to the assembler's soft-label scale."""
    if value == "present":
        return 1.0
    if value == "absent":
        return 0.0
    if value == "not_addressed":
        return SILENCE_INFORMATIVE.get(label, 0.5)
    return 0.5  # uncertain, laterality_ambiguous


def artifact_paths(mode: str, test: bool) -> dict[str, Path]:
    prefix = "gold_" if mode == "gold" else ""
    suffix = "_test" if test else ""
    stem = f"{prefix}targeted_reextract_v1{suffix}"
    paths = {
        "checkpoint": PROJECT_ROOT / "input" / f"{stem}_checkpoint.json",
        "detailed": PROJECT_ROOT / "input" / f"{stem}_detailed.csv",
        "labels": PROJECT_ROOT / "input" / f"{stem}_labels.csv",
        "provenance": PROJECT_ROOT / "input" / f"{stem}_provenance.json",
    }
    if mode == "gold":
        paths["eval"] = PROJECT_ROOT / "input" / f"{stem}_eval.json"
    return paths


def load_work(mode: str) -> list[tuple[str, str]]:
    """Return a deduplicated list of (uid, report) for the requested mode."""
    if mode == "gold":
        train = pd.read_csv(
            PROJECT_ROOT / "input" / "train.csv", dtype={"StudyInstanceUID": str}
        )
        gold = train[train[LABELS].notna().all(axis=1)]
        work = [
            (row.StudyInstanceUID, "" if pd.isna(row.Report) else str(row.Report))
            for row in gold.itertuples(index=False)
        ]
    else:
        cases = pd.read_csv(DISAGREEMENT_CASES, dtype={"uid": str})
        cases = cases.drop_duplicates(subset="uid")
        work = [
            (row.uid, "" if pd.isna(row.report) else str(row.report))
            for row in cases.itertuples(index=False)
        ]
    seen, unique = set(), []
    for uid, report in work:
        if uid not in seen:
            seen.add(uid)
            unique.append((uid, report))
    return unique


def call_vote(client, model: str, report: str, temperature: float, retries: int) -> dict:
    """One self-consistency vote: a single-report batch call, validated strictly.

    Malformed/invalid responses are retried once (per ``retries``), then the
    vote is recorded as missing — never silently coerced.
    """
    prompt = build_soft_batch_extraction_prompt([("0", report)])
    temperature = MODEL_TEMPERATURE_OVERRIDES.get(model, temperature)
    errors = []
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            text = response.choices[0].message.content or ""
            parsed = parse_batch_response(text, ["0"])["0"]
            usage = getattr(response, "usage", None)
            return {
                "model": model,
                "ok": True,
                "findings": parsed,
                "usage": {
                    "model": model,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
            }
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    LOGGER.warning("vote failed model=%s: %s", model, errors[-1][:200])
    return {"model": model, "ok": False, "error": " | ".join(errors)[:500]}


def process_report(
    client, uid: str, report: str, models: tuple, k: int, temperature: float, retries: int
) -> dict:
    """Run all model x k votes for one report concurrently and aggregate per finding."""
    tasks = [(model, vote_index) for model in models for vote_index in range(k)]
    with ThreadPoolExecutor(max_workers=len(tasks)) as vote_pool:
        votes = list(vote_pool.map(
            lambda task: call_vote(client, task[0], report, temperature, retries), tasks
        ))
    valid = [vote for vote in votes if vote["ok"]]
    if not valid:
        raise RuntimeError(f"all {len(votes)} votes failed for {uid}")

    findings = {}
    for label in LABELS:
        values = [vote["findings"][label]["value"] for vote in valid]
        counts = Counter(values)
        plurality_value, plurality_count = counts.most_common(1)[0]
        evidence = ""
        for vote in valid:
            entry = vote["findings"][label]
            if entry["value"] == plurality_value and entry["evidence"]:
                evidence = entry["evidence"]
                break
        findings[label] = {
            "votes": dict(counts),
            "n_valid_votes": len(values),
            "soft_score": float(np.mean([value_soft_score(v, label) for v in values])),
            "agreement": plurality_count / len(values),
            "evidence": evidence,
        }
    return {
        "uid": uid,
        "findings": findings,
        "n_votes_total": len(votes),
        "n_votes_valid": len(valid),
        "usage": [vote.get("usage") for vote in valid],
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(
    results: dict[str, dict], mode: str, paths: dict, started_iso: str,
    args: argparse.Namespace, token_usage: list
) -> None:
    detailed = []
    for uid, result in results.items():
        for label in LABELS:
            finding = result["findings"][label]
            detailed.append({
                "uid": uid,
                "finding": label,
                "votes_json": json.dumps(finding["votes"], sort_keys=True),
                "soft_score": round(finding["soft_score"], 4),
                "agreement": round(finding["agreement"], 4),
                "evidence": finding["evidence"],
            })
    detailed_df = pd.DataFrame(
        detailed, columns=["uid", "finding", "votes_json", "soft_score", "agreement", "evidence"]
    )
    detailed_df.to_csv(paths["detailed"], index=False)

    label_rows = [
        {"StudyInstanceUID": uid,
         **{label: round(result["findings"][label]["soft_score"], 4) for label in LABELS}}
        for uid, result in results.items()
    ]
    labels_df = pd.DataFrame(label_rows, columns=["StudyInstanceUID", *LABELS])
    labels_df.to_csv(paths["labels"], index=False)
    LOGGER.info("wrote %d detailed rows, %d label rows", len(detailed_df), len(labels_df))

    eval_payload = None
    if mode == "gold" and labels_df is not None and len(labels_df):
        train = pd.read_csv(
            PROJECT_ROOT / "input" / "train.csv", dtype={"StudyInstanceUID": str}
        )
        gold_labels = train[train[LABELS].notna().all(axis=1)][["StudyInstanceUID", *LABELS]]
        eval_payload = {
            "gold_studies_evaluated": int(labels_df.StudyInstanceUID.isin(
                set(gold_labels.StudyInstanceUID)).sum()),
            "gold_studies_total": int(len(gold_labels)),
            **evaluate_against_gold(labels_df, gold_labels),
        }
        baselines = {}
        for name in ("label_extraction_eval.json", "label_extraction_soft_v2_eval.json"):
            path = PROJECT_ROOT / "input" / name
            if path.is_file():
                prior = json.loads(path.read_text())
                baselines[name] = {
                    key: prior.get(key)
                    for key in ("macro_auc", "gold_studies_evaluated", *LABELS)
                    if key in prior
                }
        eval_payload["baseline_comparison"] = baselines
        atomic_json(eval_payload, paths["eval"])
        LOGGER.info("gold eval macro_auc=%s", eval_payload.get("macro_auc"))

    provenance = {
        "script": Path(__file__).name,
        "mode": mode,
        "test_mode": bool(args.test),
        "models": list(args.models),
        "k_per_model": args.k,
        "votes_per_report": len(args.models) * args.k,
        "temperature": args.temperature,
        "retries_per_vote": args.retries,
        "started_utc": started_iso,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "reports_processed": len(results),
        "detailed_rows": len(detailed_df),
        "label_rows": len(labels_df),
        "per_call_token_usage": token_usage,
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
            if name in ("detailed", "labels", "eval") and path.is_file()
        },
    }
    atomic_json(provenance, paths["provenance"])
    if eval_payload is not None:
        LOGGER.info("eval: %s", json.dumps(eval_payload, default=float)[:800])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("disagreement", "gold"), default="disagreement")
    parser.add_argument("--test", type=int, default=0,
                        help="process only N reports; artifacts get a _test suffix")
    parser.add_argument("--k", type=int, default=5, help="votes per model per report")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=2,
                        help="reports processed concurrently; each report runs "
                             "models*k votes concurrently, so total in-flight "
                             "calls = workers * models * k")
    parser.add_argument("--retries", type=int, default=1, help="retries per invalid vote")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()
    args.models = MODELS
    if args.k < 1 or args.workers < 1 or args.retries < 0:
        parser.error("k/workers must be positive and retries non-negative")
    if not 0.0 < args.temperature <= 2.0:
        parser.error("temperature must be nonzero so self-consistency votes decorrelate")

    paths = artifact_paths(args.mode, bool(args.test))
    work = load_work(args.mode)
    results = {}
    if args.resume and paths["checkpoint"].is_file():
        results = json.loads(paths["checkpoint"].read_text())
        LOGGER.info("resumed %d reports from %s", len(results), paths["checkpoint"])
    todo = [(uid, report) for uid, report in work if uid not in results]
    if args.test > 0:
        todo = todo[: args.test]
    LOGGER.info(
        "mode=%s queued=%d reports (%d votes each) test=%s",
        args.mode, len(todo), len(args.models) * args.k, bool(args.test),
    )

    started_iso = datetime.now(timezone.utc).isoformat()
    token_usage = []
    if todo:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENCODE_API_KEY"], base_url=BASE_URL,
            timeout=args.request_timeout, max_retries=0,
        )
        lock = Lock()
        failures = []
        started = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(
                    process_report, client, uid, report,
                    args.models, args.k, args.temperature, args.retries,
                ): uid
                for uid, report in todo
            }
            for completed, future in enumerate(as_completed(pending), 1):
                uid = pending[future]
                try:
                    result = future.result()
                except Exception as error:
                    failures.append({"uid": uid, "error": f"{type(error).__name__}: {error}"})
                    LOGGER.error("report %s failed and was NOT checkpointed: %s", uid, error)
                    continue
                with lock:
                    results[uid] = result
                    token_usage.extend(u for u in result.pop("usage") if u)
                    atomic_json(results, paths["checkpoint"])
                LOGGER.info(
                    "reports %d/%d; saved=%d; failed=%d; %.1f reports/min",
                    completed, len(todo), len(results), len(failures),
                    60.0 * completed / max(time.time() - started, 1.0),
                )
        if failures:
            LOGGER.warning("%d reports failed; rerun with --resume to retry", len(failures))

    write_outputs(results, args.mode, paths, started_iso, args, token_usage)
    LOGGER.info("Done; artifacts stem=%s", paths["detailed"].stem)


if __name__ == "__main__":
    main()
