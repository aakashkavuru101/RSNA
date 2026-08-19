"""Reproducible batched report-label extraction through OpenCode Go.

Each output namespace is pinned to one model. Failed or incomplete batches are
never checkpointed, so ``--resume`` safely retries them. Gold rows use a
separate checkpoint and are never written to the silver training artifact.
"""

from __future__ import annotations

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

from src.labels.extractor import LABELS, build_soft_batch_extraction_prompt

load_dotenv(PROJECT_ROOT / ".env")

BASE_URL = "https://opencode.ai/zen/go/v1"
ALLOWED_MODELS = ("kimi-k3", "deepseek-v4-flash", "deepseek-v4-pro")
VALID_VALUES = {
    "present", "absent", "uncertain", "not_addressed", "laterality_ambiguous"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def artifact_paths(model: str, gold_only: bool) -> dict[str, Path]:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    prefix = "gold_" if gold_only else ""
    stem = f"{prefix}extraction_soft_v3_{slug}"
    return {
        "checkpoint": PROJECT_ROOT / "input" / f"{stem}_checkpoint.json",
        "detailed": PROJECT_ROOT / "input" / f"{stem}_detailed.csv",
        "labels": PROJECT_ROOT / "input" / f"silver_labels_soft_v3_{slug}.csv",
        "eval": PROJECT_ROOT / "input" / f"label_extraction_soft_v3_{slug}_eval.json",
        "provenance": PROJECT_ROOT / "input" / f"{stem}_provenance.json",
    }


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    os.replace(temporary, path)


def parse_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("response has no JSON object")
    return json.loads(candidate[start:end])


def validate_finding(entry: object, label: str) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"{label}: finding entry is not an object")
    value = entry.get("value")
    if value not in VALID_VALUES:
        raise ValueError(f"{label}: invalid value {value!r}")
    score = entry.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{label}: score is not numeric")
    score = float(score)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label}: score outside [0, 1]")
    evidence = entry.get("evidence", "")
    if not isinstance(evidence, str):
        raise ValueError(f"{label}: evidence is not text")
    return {"value": value, "score": score, "evidence": evidence}


def parse_batch_response(text: str, expected_ids: list[str]) -> dict[str, dict]:
    payload = parse_json_object(text)
    reports = payload.get("reports")
    if not isinstance(reports, dict):
        raise ValueError("response is missing the reports object")
    if set(map(str, reports)) != set(expected_ids):
        raise ValueError(
            f"response ids differ: expected={expected_ids}, observed={list(reports)}"
        )
    parsed = {}
    for identifier in expected_ids:
        findings = reports.get(identifier)
        if not isinstance(findings, dict) or set(findings) != set(LABELS):
            raise ValueError(f"report {identifier}: finding-key contract drift")
        parsed[identifier] = {
            label: validate_finding(findings[label], label) for label in LABELS
        }
    return parsed


def call_batch(client, model: str, batch: list[tuple[str, str]], retries: int) -> dict:
    local = [(str(index), report) for index, (_, report) in enumerate(batch)]
    prompt = build_soft_batch_extraction_prompt(local)
    expected = [identifier for identifier, _ in local]
    errors = []
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0 if model == "kimi-k3" else 0.0,
                max_tokens=max(6000, 1300 * len(batch)),
            )
            text = response.choices[0].message.content or ""
            parsed = parse_batch_response(text, expected)
            return {batch[int(identifier)][0]: value for identifier, value in parsed.items()}
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(" | ".join(errors))


def chunks(items: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def evaluate_gold(train: pd.DataFrame, results: dict[str, dict]) -> dict:
    from sklearn.metrics import roc_auc_score

    gold = train[train[LABELS].notna().all(axis=1)]
    output = {
        "gold_studies_evaluated": int(sum(uid in results for uid in gold.StudyInstanceUID)),
        "gold_studies_total": int(len(gold)),
        "complete": False,
    }
    if output["gold_studies_evaluated"] != output["gold_studies_total"]:
        output["macro_auc"] = None
        return output
    aucs = []
    for label in LABELS:
        truth = gold[label].to_numpy(float)
        score = np.asarray([results[uid][label]["score"] for uid in gold.StudyInstanceUID])
        if np.unique(truth).size != 2:
            continue
        auc = float(roc_auc_score(truth, score))
        output[label] = auc
        aucs.append(auc)
    output["macro_auc"] = float(np.mean(aucs))
    output["complete"] = True
    return output


def write_outputs(
    train: pd.DataFrame, results: dict[str, dict], paths: dict, gold_only: bool
) -> None:
    detailed = []
    for uid, findings in results.items():
        for label in LABELS:
            detailed.append({"StudyInstanceUID": uid, "finding": label, **findings[label]})
    pd.DataFrame(detailed).to_csv(paths["detailed"], index=False)
    if gold_only:
        evaluation = evaluate_gold(train, results)
        atomic_json(evaluation, paths["eval"])
        LOGGER.info("Gold evaluation: %s", evaluation)
        return
    gold_uids = set(train.loc[train[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    rows = [
        {"StudyInstanceUID": uid, **{label: findings[label]["score"] for label in LABELS}}
        for uid, findings in results.items() if uid not in gold_uids
    ]
    pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS]).to_csv(
        paths["labels"], index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALLOWED_MODELS, required=True)
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.retries < 0:
        parser.error("batch-size/workers must be positive and retries non-negative")

    train = pd.read_csv(
        PROJECT_ROOT / "input" / "train.csv", dtype={"StudyInstanceUID": str}
    )
    gold = train[LABELS].notna().all(axis=1)
    selected = train[gold if args.gold_only else ~gold]
    paths = artifact_paths(args.model, args.gold_only)
    results = {}
    if args.resume and paths["checkpoint"].is_file():
        results = json.loads(paths["checkpoint"].read_text())

    work = [
        (row.StudyInstanceUID, "" if pd.isna(row.Report) else str(row.Report))
        for row in selected.itertuples(index=False)
        if row.StudyInstanceUID not in results
    ]
    if args.test > 0:
        work = work[:args.test]
    batches = chunks(work, args.batch_size)
    LOGGER.info(
        "model=%s gold=%s queued=%d reports in %d batches; resume=%d",
        args.model, args.gold_only, len(work), len(batches), len(results),
    )
    if batches:
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
                executor.submit(call_batch, client, args.model, batch, args.retries): batch
                for batch in batches
            }
            for completed, future in enumerate(as_completed(pending), 1):
                batch = pending[future]
                try:
                    parsed = future.result()
                    with lock:
                        results.update(parsed)
                        atomic_json(results, paths["checkpoint"])
                except Exception as error:
                    failures.append({
                        "uids": [uid for uid, _ in batch],
                        "error": f"{type(error).__name__}: {error}",
                    })
                if completed == 1 or completed % 5 == 0 or completed == len(batches):
                    elapsed = max(time.time() - started, 1.0)
                    LOGGER.info(
                        "batches %d/%d; saved=%d; failed=%d; %.1f reports/min",
                        completed, len(batches), len(results), len(failures),
                        60.0 * completed * args.batch_size / elapsed,
                    )
        provenance = {
            "model": args.model,
            "gold_only": args.gold_only,
            "batch_size": args.batch_size,
            "requested_reports": len(work),
            "saved_reports_total": len(results),
            "failed_batches": failures,
        }
        atomic_json(provenance, paths["provenance"])
        if failures:
            LOGGER.warning("%d batches failed and remain retryable with --resume", len(failures))

    write_outputs(train, results, paths, args.gold_only)
    LOGGER.info("Done; checkpoint=%s", paths["checkpoint"])


if __name__ == "__main__":
    main()
