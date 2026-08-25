"""Self-contained Kaggle kernel: soft label extraction with open-weight Qwen via vLLM.

Re-extracts all 12 report labels for every report-bearing study using a
locally served Qwen-class model (OpenAI-compatible vLLM endpoint on
localhost), replacing the API-capped commercial extractor. Dev-time
infrastructure only: internet ON, never produces submission.csv.

The prompt builder, label list, parsing/validation helpers, and gold
evaluation are copied VERBATIM from src/labels/extractor.py and
notebooks/02_label_extraction_batch.py so the extraction contract does not
drift from the incumbent pipeline. Checkpoint schema:
    {StudyInstanceUID: {label: {"value", "score", "evidence"}}}
written atomically after every successful batch. Failed batches are never
checkpointed, so a rerun with the checkpoint mounted as an input dataset
resumes and retries only the failures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL_LADDER = [
    # Single T4 (16 GB, sm_75). v5 lesson: AWQ has no fast kernel on Turing —
    # vLLM falls back to the unoptimized AWQ path (~0 completions in 3 h).
    # GPTQ-Int4 has working sm_75 kernels; the 4B entry is fp16 (no quant at
    # all). If a P100 (sm_60) is ever assigned despite the metadata, all of
    # these hard-fail — intentional: the ladder exhausts and raises rather
    # than silently degrading.
    "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "Qwen/Qwen3-4B-Instruct-2507",
]
BATCH_SIZE = 4
WORKERS = 4
RETRIES = 2
MAX_MODEL_LEN = 10240  # v4 died on KV cache: 16384 needs 3.0 GiB, T4 had 2.2 free
MAX_NUM_SEQS = 32  # v5: Qwen3-14B OOM'd in sampler warmup with the 256 default
SMOKE = True  # True: 120 silver + 10 gold, then guard + stop. False: full run.

SERVER_PORT = 8000
SERVER_STARTUP_TIMEOUT_S = 25 * 60  # poll /health up to 25 min per model
GPU_MEMORY_UTILIZATION = 0.92
REQUEST_TIMEOUT_S = 900.0

SMOKE_SILVER_STUDIES = 120
SMOKE_GOLD_STUDIES = 10

MIN_PARSE_SUCCESS = 0.98
MIN_GOLD_MACRO_AUC = 0.778  # incumbent in-house extractor's gold macro AUC
MAX_PROJECTED_FULL_RUN_H = 11.0  # 12 h session ceiling minus safety margin

COMPETITION = "rsna-knee-abnormality-detection"
REPORTS_DATASET = "rsna-knee-train-reports"  # aakashkavuru/rsna-knee-train-reports
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
HF_CACHE = Path("/tmp/hf_cache")  # keep multi-GB weights out of the output bundle

VALID_VALUES = {
    "present", "absent", "uncertain", "not_addressed", "laterality_ambiguous"
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


# ----------------------------------------------------------------------------
# VERBATIM copy from src/labels/extractor.py — do not edit in place.
# ----------------------------------------------------------------------------
# The 12 competition labels in submission order
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# Official positivity thresholds from the annotation protocol
POSITIVITY_THRESHOLDS = {
    "ACL": "High-grade partial tear (>50% of fibers) or complete tear",
    "MCL": "High-grade acute tear (grade II-III); grade I sprain is negative",
    "Medial Meniscus": "Signal contacting an articular surface on >=2 images, or definite morphologic deformity/root or displaced tear",
    "Lateral Meniscus": "Signal contacting an articular surface on >=2 images, or definite morphologic deformity/root or displaced tear",
    "Medial OA": ">=1 cm of >50%-thickness cartilage loss in the medial compartment",
    "Lateral OA": ">=1 cm of >50%-thickness cartilage loss in the lateral compartment",
    "PF OA": ">=1 cm of >50%-thickness cartilage loss in the patellofemoral compartment",
    "Effusion": "Moderate or large effusion; trace/small fluid is negative",
    "Synovitis": "Definite synovial thickening/proliferation; isolated effusion or Hoffa signal is not sufficient",
    "Baker's": "Moderate or large Baker/popliteal cyst; small physiologic bursal fluid is negative",
    "Contusion": "Definite geographic traumatic marrow edema without a fracture line or cortical deformity",
    "Fracture": "Definite fracture line, cortical breach, impaction, or subchondral/insufficiency fracture",
}


def build_soft_batch_extraction_prompt(
    reports,
) -> str:
    """Build one rubric-locked request for several independent reports.

    Short request-local identifiers reduce tokens and prevent a model from
    rewriting StudyInstanceUID values. The caller owns the UID mapping.
    """
    if not reports:
        raise ValueError("at least one report is required")
    identifiers = [str(identifier) for identifier, _ in reports]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("batch report identifiers must be unique")

    thresholds_text = "\n".join(
        f"- {label}: {threshold}" for label, threshold in POSITIVITY_THRESHOLDS.items()
    )
    report_blocks = "\n\n".join(
        f"<REPORT id={json.dumps(identifier)}>\n{text}\n</REPORT>"
        for identifier, text in reports
    )
    finding_template = ",\n".join(
        f'      {json.dumps(label)}: '
        '{"value": "...", "score": 0.0, "evidence": "..."}'
        for label in LABELS
    )

    return f"""You are a musculoskeletal radiology expert creating independent soft training targets for knee MRI reports.

Evaluate every report separately. Never transfer a finding between reports. Interpret each report in its original language, including negation, uncertainty, severity, anatomical compartment, and acute versus chronic wording. Medial/lateral are anatomical compartments, not patient left/right.

Official positivity thresholds (borderline is negative):
{thresholds_text}

Scoring anchors:
- 0.00-0.10: explicitly normal or absent
- 0.15-0.35: mild, minimal, grade I, intrasubstance degeneration, or below threshold
- 0.40-0.60: uncertain, not addressed, or incompletely characterized
- 0.65-0.85: likely positive or high-grade without complete size detail
- 0.90-1.00: definite and clearly meets the threshold

Critical rules:
- Meniscus grade 1-2 signal without articular-surface contact is below threshold. Root tear, displaced fragment, definite deformity, or surface contact on at least two images is positive.
- ACL low-grade sprain/partial injury is below threshold; greater than 50 percent disruption or complete rupture is positive.
- MCL grade I/periligamentous edema with intact fibers is below threshold; grade II-III acute disruption is positive.
- OA needs at least 1 cm of greater than 50 percent cartilage loss. Mild thinning, isolated marrow edema, or an equivocal osteophyte is below threshold.
- Trace/small effusion and small Baker cyst are below threshold. Moderate/large is positive.
- Synovitis needs definite synovial thickening/proliferation. Effusion or Hoffa signal alone is insufficient.
- Contusion is geographic traumatic marrow edema without a fracture line or cortical deformity. Degenerative marrow edema is not a contusion.
- Fracture requires a line, cortical breach, impaction, or explicit subchondral/insufficiency fracture. Marrow edema alone is insufficient.
- If a finding is not discussed and no correlated finding supplies real evidence, use value "not_addressed" and a near-neutral score.
- Evidence must be an exact short quote from that report, or an empty string when not addressed.

Reports:
{report_blocks}

Return ONLY strict JSON in this exact outer shape, with every requested id once and all 12 finding keys:
{{
  "reports": {{
    {json.dumps(identifiers[0])}: {{
{finding_template}
    }}
  }}
}}

Allowed values are "present", "absent", "uncertain", "not_addressed", and "laterality_ambiguous". Scores must be finite numbers from 0 to 1. Do not add commentary or markdown fences."""


# ----------------------------------------------------------------------------
# VERBATIM copy from notebooks/02_label_extraction_batch.py — do not edit.
# ----------------------------------------------------------------------------
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


def parse_batch_response(text: str, expected_ids: list) -> dict:
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


def chunks(items: list, size: int) -> list:
    return [items[index:index + size] for index in range(0, len(items), size)]


def evaluate_gold(train: pd.DataFrame, results: dict) -> dict:
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


# ----------------------------------------------------------------------------
# Kernel-specific infrastructure
# ----------------------------------------------------------------------------
def model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def artifact_paths(model: str, gold_only: bool) -> dict:
    """Mirror notebooks/02_label_extraction_batch.py::artifact_paths exactly.

    Gold runs carry the ``gold_`` prefix on checkpoint/detailed/provenance;
    the eval JSON name intentionally has no gold prefix, matching the
    incumbent contract. Gold rows are never written to the silver artifact.
    """
    slug = model_slug(model)
    prefix = "gold_" if gold_only else ""
    stem = f"{prefix}extraction_soft_v3_{slug}"
    return {
        "checkpoint": OUTPUT_DIR / f"{stem}_checkpoint.json",
        "detailed": OUTPUT_DIR / f"{stem}_detailed.csv",
        "labels": OUTPUT_DIR / f"gold_labels_soft_v3_{slug}.csv"
        if gold_only else OUTPUT_DIR / f"silver_labels_soft_v3_{slug}.csv",
        "eval": OUTPUT_DIR / f"label_extraction_soft_v3_{slug}_eval.json",
        "provenance": OUTPUT_DIR / f"{stem}_provenance.json",
        "diagnostics": OUTPUT_DIR / f"{stem}_diagnostics.json",
    }


def find_competition_input() -> Path:
    candidates = [
        # Works when internet is off and competition_sources mount.
        Path("/kaggle/input") / COMPETITION,
        # With internet ON, Kaggle may skip competition_sources; this private
        # dataset carries just train.csv and mounts via dataset_sources.
        Path("/kaggle/input") / REPORTS_DATASET,
        Path("data"),
        Path("input"),
    ]
    for path in candidates:
        if (path / "train.csv").is_file():
            return path
    raise FileNotFoundError("RSNA Knee competition input was not mounted")


def find_input_checkpoint(filename: str) -> Path | None:
    """Locate a checkpoint JSON mounted as a Kaggle input dataset."""
    hits = []
    for root, directories, files in os.walk("/kaggle/input"):
        # Never recursively index the raw DICOM trees just to locate a JSON.
        directories[:] = [d for d in directories if d not in ("train_series", "test_series")]
        if filename in files:
            hits.append(Path(root) / filename)
    if len(hits) > 1:
        raise FileNotFoundError(
            f"ambiguous checkpoint: {len(hits)} copies of {filename} under /kaggle/input"
        )
    return hits[0] if hits else None


def load_checkpoint(path: Path) -> dict:
    if path.is_file():
        results = json.loads(path.read_text())
        log(f"resumed checkpoint {path.name}: {len(results)} studies")
        return results
    found = find_input_checkpoint(path.name)
    if found is not None:
        results = json.loads(found.read_text())
        log(f"resumed input checkpoint {found}: {len(results)} studies")
        return results
    return {}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_parallel_size(model: str) -> int:
    # Single-GPU kernel (NvidiaTeslaT4): always 1; kept as a function so the
    # launch command reads the same if multi-GPU shapes ever return.
    return 1


def install_vllm() -> None:
    # Pin the 0.10.x line: the last releases with reliable Turing (sm_75, T4)
    # attention-backend support; unpinned latest vllm risks dropping sm_75.
    # transformers pinned to 4.x: v5 removed all_special_tokens_extended, which
    # vllm 0.10.2's cached-tokenizer path still reads (smoke v3 crash).
    log("installing vllm==0.10.2 + transformers==4.55.4 (internet on, dev-time only)")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "vllm==0.10.2", "transformers==4.55.4", "openai"],
        check=True,
    )


def server_health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/health", timeout=10
        ) as response:
            return response.status == 200
    except Exception:
        return False


def start_server() -> tuple:
    """Walk the model ladder; return (process, model) for the first that serves."""
    env = dict(os.environ)
    env["HF_HOME"] = str(HF_CACHE)
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    for position, model in enumerate(MODEL_LADDER):
        log_path = OUTPUT_DIR / f"vllm_server_{model_slug(model)}.log"
        command = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--tensor-parallel-size", str(tensor_parallel_size(model)),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--disable-log-requests",
            "--port", str(SERVER_PORT),
        ]
        log(f"ladder[{position}] launching {model} (tp={tensor_parallel_size(model)})")
        log_handle = log_path.open("w")
        process = subprocess.Popen(
            command, stdout=log_handle, stderr=subprocess.STDOUT, env=env
        )
        deadline = time.time() + SERVER_STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if process.poll() is not None:
                log(f"{model} exited during startup (code {process.returncode})")
                break
            if server_health_ok(SERVER_PORT):
                log(f"{model} healthy after {int(time.time() - (deadline - SERVER_STARTUP_TIMEOUT_S))}s")
                return process, model, position
            time.sleep(15)
        else:
            log(f"{model} failed to become healthy within {SERVER_STARTUP_TIMEOUT_S // 60} min")
        process.kill()
        process.wait()
        log_handle.close()
        tail = log_path.read_text(errors="replace")[-2000:] if log_path.is_file() else ""
        log(f"server log tail for {model}:\n{tail}")
        if position == len(MODEL_LADDER) - 1:
            raise RuntimeError("every model on the ladder failed to start; see server logs")
    raise RuntimeError("unreachable")


def call_batch(client, model: str, batch: list, retries: int) -> dict:
    """Same call/parse pattern as the incumbent script; temperature pinned to 0."""
    local = [(str(index), report) for index, (_, report) in enumerate(batch)]
    prompt = build_soft_batch_extraction_prompt(local)
    expected = [identifier for identifier, _ in local]
    errors = []
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
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


def run_extraction(client, model: str, work: list, paths: dict, results: dict) -> dict:
    """Threaded batch loop with atomic checkpointing after every success."""
    batches = chunks(work, BATCH_SIZE)
    log(f"queued={len(work)} reports in {len(batches)} batches; resumed={len(results)}")
    failures = []
    started = time.time()
    lock = Lock()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        pending = {
            executor.submit(call_batch, client, model, batch, RETRIES): batch
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
                log(
                    f"batches {completed}/{len(batches)}; saved={len(results)}; "
                    f"failed={len(failures)}; {60.0 * completed * BATCH_SIZE / elapsed:.1f} reports/min"
                )
    elapsed = time.time() - started
    return {
        "failures": failures,
        "elapsed_s": elapsed,
        "requested_reports": len(work),
        "batches_completed": len(batches) - len(failures),
        "batches_total": len(batches),
        "reports_per_min": 60.0 * len(work) / max(elapsed, 1.0) if work else 0.0,
    }


def write_detailed(results: dict, path: Path) -> None:
    detailed = []
    for uid, findings in results.items():
        for label in LABELS:
            detailed.append({"StudyInstanceUID": uid, "finding": label, **findings[label]})
    pd.DataFrame(detailed).to_csv(path, index=False)


def cell_score(finding: dict, label: str) -> float:
    """Score-first cell value; assembler backfill semantics only if score is absent.

    validate_finding guarantees a numeric score for every parsed finding, so the
    SILENCE_INFORMATIVE / 0.5 backfill below is defensive only and mirrors
    src/labels/assembler.py: not_addressed -> SILENCE_INFORMATIVE or 0.5,
    present -> 1.0, absent -> 0.0, uncertain/laterality_ambiguous -> 0.5.
    """
    score = finding.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and np.isfinite(score):
        return float(score)
    silence = {"Baker's": 0.1, "Effusion": 0.3, "Fracture": 0.2}
    value = finding.get("value")
    if value == "present":
        return 1.0
    if value == "absent":
        return 0.0
    if value == "not_addressed":
        return silence.get(label, 0.5)
    return 0.5


def write_labels(results: dict, path: Path, exclude_uids: set) -> None:
    rows = [
        {"StudyInstanceUID": uid,
         **{label: cell_score(findings[label], label) for label in LABELS}}
        for uid, findings in results.items() if uid not in exclude_uids
    ]
    pd.DataFrame(rows, columns=["StudyInstanceUID", *LABELS]).to_csv(path, index=False)


def main() -> None:
    install_vllm()

    root = find_competition_input()
    # train.csv has embedded newlines in Report (58,556 lines / 4,407 rows);
    # the pandas CSV parser is mandatory.
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    gold_mask = train[LABELS].notna().all(axis=1)
    gold = train[gold_mask]
    silver = train[~gold_mask]
    log(f"train rows={len(train)}; gold={len(gold)}; silver={len(silver)}")

    def report_of(row) -> str:
        return "" if pd.isna(row.Report) else str(row.Report)

    silver_work_all = [
        (row.StudyInstanceUID, report_of(row))
        for row in silver.itertuples(index=False)
        if report_of(row).strip()
    ]
    gold_work_all = [
        (row.StudyInstanceUID, report_of(row))
        for row in gold.itertuples(index=False)
        if report_of(row).strip()
    ]
    log(f"report-bearing: silver={len(silver_work_all)} gold={len(gold_work_all)}")

    process, model, ladder_position = start_server()
    slug = model_slug(model)
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=f"http://localhost:{SERVER_PORT}/v1",
            api_key="EMPTY",
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,
        )

        silver_paths = artifact_paths(model, gold_only=False)
        gold_paths = artifact_paths(model, gold_only=True)
        silver_results = load_checkpoint(silver_paths["checkpoint"])
        gold_results = load_checkpoint(gold_paths["checkpoint"])

        silver_work = [pair for pair in silver_work_all if pair[0] not in silver_results]
        gold_work = [pair for pair in gold_work_all if pair[0] not in gold_results]
        if SMOKE:
            silver_work = silver_work[:SMOKE_SILVER_STUDIES]
            gold_work = gold_work[:SMOKE_GOLD_STUDIES]

        started = time.time()
        silver_stats = run_extraction(client, model, silver_work, silver_paths, silver_results)
        gold_stats = run_extraction(client, model, gold_work, gold_paths, gold_results)
        total_elapsed_s = time.time() - started

        silver_done = sum(1 for uid, _ in silver_work if uid in silver_results)
        gold_done = sum(1 for uid, _ in gold_work if uid in gold_results)
        requested = len(silver_work) + len(gold_work)
        parse_success = (silver_done + gold_done) / max(requested, 1)
        log(f"parse success: {silver_done + gold_done}/{requested} = {parse_success:.4f}")

        diagnostics = {
            "model": model,
            "ladder_position": ladder_position,
            "smoke": SMOKE,
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "retries": RETRIES,
            "parse_success": parse_success,
            "silver": silver_stats,
            "gold": gold_stats,
            "total_elapsed_s": total_elapsed_s,
        }

        if SMOKE:
            # Project the full silver+gold run from the measured smoke rate.
            full_reports = len(silver_work_all) + len(gold_work_all)
            rate = (silver_done + gold_done) / max(total_elapsed_s, 1.0) * 60.0
            projected_h = full_reports / max(rate, 1e-6) / 60.0
            log(
                f"smoke rate {rate:.1f} reports/min -> projected full run "
                f"{projected_h:.2f} h for {full_reports} reports"
            )
            diagnostics["projected_full_run_h"] = projected_h
            atomic_json(diagnostics, silver_paths["diagnostics"])
            if parse_success < MIN_PARSE_SUCCESS:
                raise RuntimeError(
                    f"SMOKE parse success {parse_success:.4f} < {MIN_PARSE_SUCCESS}; "
                    "fix prompt contract or model before a full run"
                )
            if projected_h > MAX_PROJECTED_FULL_RUN_H:
                raise SystemExit(
                    f"projected full run {projected_h:.2f} h exceeds "
                    f"{MAX_PROJECTED_FULL_RUN_H} h session budget; "
                    "diagnostics written, pick a faster ladder entry or batch shape"
                )
            log("SMOKE passed; flip SMOKE=False for the full run")
            return

        # Full run: guards must pass before the silver artifact is written.
        evaluation = evaluate_gold(train, gold_results)
        log(f"gold evaluation: {evaluation}")
        diagnostics["gold_evaluation"] = evaluation
        macro_auc = evaluation.get("macro_auc")
        guard_errors = []
        if parse_success < MIN_PARSE_SUCCESS:
            guard_errors.append(
                f"parse success {parse_success:.4f} < {MIN_PARSE_SUCCESS}"
            )
        if macro_auc is None or macro_auc < MIN_GOLD_MACRO_AUC:
            guard_errors.append(
                f"gold macro AUC {macro_auc} < {MIN_GOLD_MACRO_AUC} (incumbent)"
            )
        if guard_errors:
            atomic_json(diagnostics, silver_paths["diagnostics"])
            atomic_json(diagnostics, silver_paths["provenance"])
            raise RuntimeError(
                "final guards failed; only diagnostics + checkpoints written: "
                + "; ".join(guard_errors)
            )

        gold_uids = set(gold.StudyInstanceUID)
        write_detailed(silver_results, silver_paths["detailed"])
        # Gold rows never enter the silver artifact (defensive filter retained).
        write_labels(silver_results, silver_paths["labels"], gold_uids)
        write_detailed(gold_results, gold_paths["detailed"])
        write_labels(gold_results, gold_paths["labels"], exclude_uids=set())
        atomic_json(evaluation, gold_paths["eval"])

        artifacts = [
            silver_paths["checkpoint"], silver_paths["detailed"], silver_paths["labels"],
            gold_paths["checkpoint"], gold_paths["detailed"], gold_paths["labels"],
            gold_paths["eval"],
        ]
        provenance = {
            **diagnostics,
            "requested_reports_total": len(silver_work_all) + len(gold_work_all),
            "saved_silver_reports_total": len(silver_results),
            "saved_gold_reports_total": len(gold_results),
            "artifact_sha256": {
                path.name: sha256_of(path) for path in artifacts if path.is_file()
            },
        }
        atomic_json(provenance, silver_paths["provenance"])
        atomic_json(provenance, gold_paths["provenance"])
        log(f"done; silver labels -> {silver_paths['labels']}")
    finally:
        log("shutting down vLLM server")
        process.kill()
        process.wait()


if __name__ == "__main__":
    main()
