"""Rule-based labeling-function (LF) teacher -> input/rulebased_labels_v1.csv.

A third, decorrelated supervision source for the labels where the LLM
extraction is weakest on gold (Synovitis, Lateral OA, Effusion, Baker's).
Deterministic lexical labeling functions over the raw report text with
NegEx-style negation and hedge handling. Dependency-free for the labeling
core (stdlib ``re`` only) so the same code can run offline on Kaggle.

Vote semantics per (study, label):
- 1.0 positive vote  — a firm (non-negated, non-hedged) mention fired
- 0.0 negative vote  — only negated mentions fired
- 0.5 abstain        — no mention, hedge-only, or POS/NEG conflict

Aggregation across a report: POS+NEG conflict -> 0.5; POS beats HEDGE;
NEG beats HEDGE (a hedge does not overturn an explicit negative).

Reports are multilingual (en/es/de/nl/fr/hr/tr/el), so cues and finding
patterns cover the major languages seen in the corpus. Labels not covered
by LFs stay 0.5 in the output (exact silver schema, 12 label columns).

Honesty note: patterns were designed from corpus-wide phrasing inspection
which included some gold-row reports, so gold numbers below are mildly
in-sample; treat them as approximate. Gold was never used to fit numeric
thresholds; it is used here for evaluation only (gold_usage: evaluation).

Outputs (the ONLY files written):
- input/rulebased_labels_v1.csv        (4407 x 13, silver schema)
- input/rulebased_labels_v1_eval.json  (coverage/agreement/tiebreak audit)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.labels.assembler import SILENCE_INFORMATIVE
from src.labels.extractor import LABELS

MISSING = 0.5  # abstain sentinel — same contract as the silver labels
MIN_CLASS_COUNT = 5  # skip AUC for a label with <5 positives or <5 negatives

LF_LABELS = ["Effusion", "Synovitis", "Baker's", "Lateral OA"]

DEFAULT_TEACHERS = {
    "silver_v5_fused": PROJECT_ROOT / "input" / "silver_labels_v5.csv",
    "ours_gold_extraction": PROJECT_ROOT / "input" / "gold_extraction_detailed.csv",
    "flight": PROJECT_ROOT / ".codex_work" / "public_datasets"
              / "flight_hybrid_labels" / "report_labels_v4hybrid.csv",
    "stevenleehans": PROJECT_ROOT / ".codex_work" / "public_datasets"
                     / "steven_labels" / "llm_labels_full.csv",
    "pilkwang": PROJECT_ROOT / ".codex_work" / "public_datasets"
                / "pilkwang_labels" / "report_labels_v2.csv",
    "lixin": PROJECT_ROOT / ".codex_work" / "public_datasets"
             / "lixin_sol56_labels" / "report_labels_gpt56sol.csv",
}
# Teachers used for the disagreement-tiebreak analysis (the mirrored public
# LLM teacher sets; our own fused/extraction sources are excluded so the LF
# is compared against genuinely external opinions).
TIEBREAK_TEACHERS = ["flight", "stevenleehans", "pilkwang", "lixin"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# IO helpers (same conventions as 08_label_fusion.py)
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def atomic_write_json(payload: dict, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(payload, indent=2, default=float))
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


# --------------------------------------------------------------------------- #
# NegEx-style cue lexicon (multilingual, corpus-driven)
# --------------------------------------------------------------------------- #

# Pre-negation cues: appear BEFORE the finding mention, same segment.
PRE_NEG = re.compile(
    r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|\babsence\s+of\b|\bfree\s+of\b"
    r"|\bno\s+evidence\s+of\b|\bno\s+sign\w*\s+of\b|\bdenies\b|\bnegative\s+for\b"
    r"|\bsin\b|\bno\s+hay\b|\bausencia\s+de\b|\bsin\s+evidencia\s+de\b"
    r"|\bno\s+se\s+(?:evidencia|aprecia|visualiza|observa)\w*\b|\bsin\s+signos\s+de\b"
    r"|\bkein\w*\b|\bohne\b|\bnicht\b"
    r"|\bgeen\b|\bzonder\b|\bniet\b"
    r"|\bpas\s+de\b|\bsans\b|\babsence\s+de\b|\baucun\w*\b"
    r"|\bbez\b|\bne\s+nalazi\s+se\b|\bnema\b|\bnije\b|\bbez\s+znakova\b"
    r"|\byok\b|\byoktur\b|mevcut\s+de\w+)",
    re.IGNORECASE)

# Post-negation cues: appear AFTER the mention, same segment.
POST_NEG = re.compile(
    r"(?:\bis\s+(?:absent|excluded|ruled\s+out)|\bare\s+(?:absent|excluded)"
    r"|\bnot\s+(?:seen|present|identified|demonstrated|visualized)\b"
    r"|\babsent\b|\bexcluded\b|\bruled\s+out\b"
    r"|:\s*(?:none|aucun\w*|no)\b"
    r"|\bpas\s+de\b|\bpas\s+d[''']\b|\baucun\w*\b"
    r"|\bausente\b|\bse\s+descarta\b"
    r"|\bnicht\s+nachweisbar\b|\bausgeschlossen\b"
    r"|\bniet\s+aangetoond\b|\bnije\s+vidljiv\w*\b|\bno\s+se\s+visualiza\b)",
    re.IGNORECASE)

# Hedge cues: possibility / differential / recommendation language.
HEDGE = re.compile(
    r"(?:\bpossib\w+\b|\bquestionable\b|\bcannot\s+(?:be\s+)?exclud\w+"
    r"|\br/o\b|\brule\s+out\b|\bto\s+rule\s+out\b|\bmay\s+represent\b"
    r"|\bmight\s+represent\b|\bcould\s+represent\b|\bsuspect\w*\b|\bd/d?x\b"
    r"|\bdifferential\b|\bif\s+clinically\s+(?:indicated|necessary)\b"
    r"|\bclinical\s+correlation\b|\bprobablemente\b|\bposible\w*\b"
    r"|\beventuell\w*\b|\bmoeglich\w*\b|\bmogelijk\w*\b|\bvermoeden\b)",
    re.IGNORECASE)

# Segment split: sentence/line/list boundaries. Colons are kept inside the
# segment so "Baker cyst: None" style post-negation stays attached.
SEGMENT_SPLIT = re.compile(r"[.\n;•»>]+|(?:^|\s)\d{1,2}\s(?=\S)")

PRE_WINDOW = 60   # chars to look back from a mention for a pre-negation cue
POST_WINDOW = 40  # chars to look ahead for a post-negation cue
HEDGE_WINDOW = 55


def _segments(text: str):
    """Yield (start, segment_text) pairs preserving offsets into `text`."""
    pos = 0
    for m in SEGMENT_SPLIT.finditer(text):
        seg = text[pos:m.start()]
        if seg.strip():
            yield pos, seg
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        yield pos, tail


def _classify(seg: str, start: int, end: int, negative_phrase: bool) -> str:
    """Verdict for one mention [start, end) inside segment `seg`.

    Priority: explicit-negative phrase > pre-negation > post-negation >
    hedge > positive.
    """
    if negative_phrase:
        return "NEG"
    before = seg[max(0, start - PRE_WINDOW):start]
    if PRE_NEG.search(before):
        return "NEG"
    after = seg[end:end + POST_WINDOW]
    if POST_NEG.search(after):
        return "NEG"
    window = seg[max(0, start - HEDGE_WINDOW):end + HEDGE_WINDOW]
    if HEDGE.search(window):
        return "HEDGE"
    return "POS"


# --------------------------------------------------------------------------- #
# Labeling functions
# --------------------------------------------------------------------------- #
#
# Each label maps to a list of (compiled_pattern, negative_phrase_flag).
# Patterns are matched per segment; every match becomes a mention whose
# verdict comes from _classify(). negative_phrase=True patterns are
# intrinsically negative mentions ("effusion within physiologic limits").

_OA_CUE_EN = (r"(?:osteoarthrit\w*|arthros\w*|chondros\w*|chondropath\w*"
              r"|chondromalaci\w*|joint\s+space\s+narrow\w*|osteophyt\w*"
              r"|cartilage\s+(?:loss|fissur\w*|defect\w*|thinning|ulcer\w*|denud\w*)"
              r"|degenerative\s+(?:change\w*|disease))")

LF_PATTERNS: dict[str, list[tuple[re.Pattern, bool]]] = {
    "Effusion": [
        (re.compile(r"\beffusion\w*\b", re.I), False),
        (re.compile(r"\bha?emarthrosis\b|\bhemartrosis\b", re.I), False),
        (re.compile(r"\bjoint\s+fluid\b", re.I), False),
        (re.compile(r"\bderrame\b", re.I), False),
        (re.compile(r"\bgelenk\w*erguss\w*\b|\berguss\b", re.I), False),
        (re.compile(r"\bhydrops\b|\bgewrichtsvocht\b", re.I), False),
        (re.compile(r"\b[ée]panchement\w*\b", re.I), False),
        (re.compile(r"\bizljev\w*\b", re.I), False),
        # Explicit "physiologic amount" phrasing is a negative statement.
        (re.compile(r"\bphysiologic\w*\b[^.\n]{0,40}?\b(?:effusion|joint\s+fluid)\b"
                    r"|\b(?:effusion|joint\s+fluid)\b[^.\n]{0,40}?\bphysiologic\w*\b"
                    r"|\bwithin\s+normal\s+(?:physiologic\w*\s+)?limit\w*\b", re.I), True),
    ],
    "Synovitis": [
        (re.compile(r"\bsynovitis\b|\bsinovitis\b|\bsynovialitis\b"
                    r"|\breizsynovialitis\b", re.I), False),
        (re.compile(r"\bsynovial\s+(?:hypertroph\w*|thickening|proliferation"
                    r"|enhanc\w*|hyperemia|membrane\s+thickening)\b", re.I), False),
        (re.compile(r"\bthick\w*\s+synovial\s+(?:tissue|membrane|lining)\b"
                    r"|\bhypertroph\w*\s+(?:of\s+the\s+)?synovium\b", re.I), False),
        (re.compile(r"\bsynoviale\s+hypertrofie\b", re.I), False),
        (re.compile(r"\bengrosamiento\s+sinovial\b|\bhipertrofia\s+sinovial\b"
                    r"|\bsinovial\s+difus\w*\b", re.I), False),
        (re.compile(r"\bproliferaci\w+\s+sinovij\w*\b", re.I), False),
    ],
    "Baker's": [
        (re.compile(r"\bbaker'?s?\b|\bbakerov\w*\b", re.I), False),
        (re.compile(r"\bpopliteal\s+cyst\w*\b|\bpoplitealn\w*\s+cist\w*\b"
                    r"|\bpoplitealne\s+ciste\b", re.I), False),
        (re.compile(r"\bquiste\w*\s+popl[ií]teo\w*\b", re.I), False),
        (re.compile(r"\bpopliteale\s+cyste\b|\bcyste\s+popliteale\b", re.I), False),
    ],
    "Lateral OA": [
        # EN: lateral compartment first, OA cue within 40 chars.
        (re.compile(r"\blateral\w*\s+(?:(?:femoro|tibio)femoral\s+)?"
                    r"(?:joint\s+)?compartment\w*[^.\n]{0,40}?" + _OA_CUE_EN, re.I), False),
        # EN: OA cue first, "lateral compartment" within 25 chars.
        (re.compile(_OA_CUE_EN + r"[^.\n]{0,25}?\blateral\s+(?:compartment"
                    r"|(?:femoro|tibio)femoral)", re.I), False),
        # EN shorthand: "OA of (the/all) lateral ... compartment(s)".
        (re.compile(r"\bOA\s+of\s+(?:the\s+|all\s+)?(?:\w+\s+){0,2}lateral\b", re.I), False),
        # Tricompartmental OA/chondrosis involves the lateral compartment
        # ("tricompartmental marginal osteophytes" alone does NOT count).
        (re.compile(r"\btricompartmental\s+(?:mild\s+|moderate\s+|severe\s+)?"
                    r"(?:osteoarthrit\w*|chondros\w*|chondral\s+thinning"
                    r"|degenerative\s+change\w*)", re.I), False),
        # ES
        (re.compile(r"(?:condropat[ií]a|artrosis|gonartrosis|[uú]lcera\w*\s+condral\w*"
                    r"|adelgazamiento\s+condral)[^.\n]{0,50}?"
                    r"(?:compartimento\s+)?femorotibial\s+lateral", re.I), False),
        (re.compile(r"\bfemorotibial\s+lateral\b[^.\n]{0,50}?"
                    r"(?:condropat|artros|[uú]lcera|osteofit|adelgazamiento)", re.I), False),
        (re.compile(r"\b(?:gon)?artrosis\b[^.\n]{0,25}?\blateral\b", re.I), False),
        # DE / NL
        (re.compile(r"\blaterale?\s+gonarthrose\b|\bgonarthrose\b[^.\n]{0,30}?"
                    r"\blateral\w*\b|\blaterale\s+gonartrose\b"
                    r"|\bgonartrose\b[^.\n]{0,30}?\blateraal\b", re.I), False),
        (re.compile(r"\bfemorotibial\s+lateral\b[^.\n]{0,40}?"
                    r"(?:chondropathie|arthrose|osteophyt)", re.I), False),
        # FR
        (re.compile(r"\barthrose\b[^.\n]{0,30}?\b(?:lat[ée]rale|externe)\b", re.I), False),
        # Explicit negatives: preserved/normal lateral compartment cartilage.
        (re.compile(r"\blateral\w*[^.\n]{0,30}?\bcartilage\b[^.\n]{0,25}?"
                    r"\b(?:intact|normal|preserved|unremarkable)\b", re.I), True),
        (re.compile(r"\b(?:preserved|intact|normal)\b[^.\n]{0,30}?"
                    r"\blateral\w*\s+(?:femorotibial\s+)?(?:articular\s+)?cartilage\b",
                    re.I), True),
        (re.compile(r"\bfemorotibial\s+lateral\b[^.\n]{0,30}?\bsin\s+alteraciones\b"
                    r"|\bcart[ií]lago\w*\b[^.\n]{0,40}?\bfemorotibial\s+lateral\b"
                    r"[^.\n]{0,30}?\bsin\s+alteraciones\b", re.I), True),
    ],
}

# Mention guards: substrings that must not fire a pattern even if matched
# (handled by pattern specificity above; kept as a documented safety net for
# synovial-cyst-like false positives on the bare "sinovial difusa" pattern).
_GUARDS: dict[str, re.Pattern] = {
    "Synovitis": re.compile(r"\b(?:quiste|quistes| cyst\w*|hemangioma|lipoma"
                            r"|sarcoma|metaplasia|osteochondromatosis)\s+sinovi\w+\b"
                            r"|\bsinovi\w+\s+(?:cyst\w*|quiste\w*|hemangioma"
                            r"|lipoma|sarcoma)\b", re.I),
}


def label_report(text: str) -> dict[str, float]:
    """Run all LFs over one report; return {label: 0.0/0.5/1.0} for LF_LABELS."""
    votes: dict[str, set[str]] = {label: set() for label in LF_LABELS}
    for _, seg in _segments(text):
        for label in LF_LABELS:
            guard = _GUARDS.get(label)
            # Intrinsic negative-phrase spans suppress any other mention that
            # overlaps them ("effusion within physiologic limits" must not
            # also fire the bare "effusion" positive pattern).
            neg_spans = [(m.start(), m.end())
                         for pattern, neg_phrase in LF_PATTERNS[label]
                         if neg_phrase for m in pattern.finditer(seg)]
            for pattern, neg_phrase in LF_PATTERNS[label]:
                for m in pattern.finditer(seg):
                    if not neg_phrase and any(
                            m.start() < e and m.end() > s for s, e in neg_spans):
                        continue
                    if guard and guard.search(seg[max(0, m.start() - 20):
                                                  m.end() + 20]):
                        continue
                    votes[label].add(_classify(seg, m.start(), m.end(), neg_phrase))
    out = {}
    for label in LF_LABELS:
        v = votes[label]
        if "POS" in v and "NEG" in v:
            out[label] = MISSING  # conflict -> abstain
        elif "POS" in v:
            out[label] = 1.0
        elif "NEG" in v:
            out[label] = 0.0
        else:
            out[label] = MISSING  # hedge-only or no mention
    return out


def apply_lfs(train: pd.DataFrame) -> pd.DataFrame:
    """Apply LFs to every study; return a wide frame in the silver schema."""
    rows = []
    for uid, report in zip(train["StudyInstanceUID"], train["Report"]):
        row = {"StudyInstanceUID": uid}
        row.update(label_report(str(report) if pd.notna(report) else ""))
        rows.append(row)
    wide = pd.DataFrame(rows)
    for label in LABELS:
        if label not in wide.columns:
            wide[label] = MISSING
    return wide[["StudyInstanceUID"] + LABELS]


# --------------------------------------------------------------------------- #
# Teacher loading (wide silver schema or long-form detailed CSV)
# --------------------------------------------------------------------------- #

VALUE_MAP = {"present": 1.0, "absent": 0.0, "uncertain": MISSING,
             "laterality_ambiguous": MISSING}


def _pivot_detailed(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"uid": "StudyInstanceUID"})
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df = df[df["finding"].isin(LABELS)]
    score_col = next((c for c in ("soft_score", "score")
                      if c in df.columns and df[c].notna().any()), None)
    df["_score"] = (pd.to_numeric(df[score_col], errors="coerce")
                    if score_col is not None else np.nan)
    fallback = df["_score"].isna()
    if fallback.any():
        df.loc[fallback, "_score"] = [
            VALUE_MAP.get(v, SILENCE_INFORMATIVE.get(f, MISSING)
                         if v == "not_addressed" else np.nan)
            for v, f in zip(df.loc[fallback, "value"], df.loc[fallback, "finding"])
        ]
    df = df.drop_duplicates(subset=["StudyInstanceUID", "finding"], keep="first")
    wide = df.pivot(index="StudyInstanceUID", columns="finding", values="_score")
    return wide.reset_index()


def load_teacher(path: Path) -> pd.DataFrame:
    """Load one teacher as a wide frame: StudyInstanceUID + LABELS (0.5 = abstain)."""
    header = pd.read_csv(path, nrows=0)
    if "finding" in header.columns:
        df = _pivot_detailed(pd.read_csv(path, dtype={"StudyInstanceUID": str,
                                                      "uid": str}))
    else:
        keep = ["StudyInstanceUID"] + [c for c in LABELS if c in header.columns]
        df = pd.read_csv(path, dtype={"StudyInstanceUID": str}, usecols=keep)
    df = df.drop_duplicates(subset=["StudyInstanceUID"], keep="first")
    for label in LABELS:
        if label not in df.columns:
            df[label] = np.nan
        df[label] = pd.to_numeric(df[label], errors="coerce")
    df[LABELS] = df[LABELS].fillna(MISSING)
    return df[["StudyInstanceUID"] + LABELS]


def binarize(values: np.ndarray) -> np.ndarray:
    """Soft value -> hard vote: >0.5 -> 1, <0.5 -> 0, ==0.5/NaN -> NaN."""
    out = np.full(values.shape, np.nan)
    out[values > 0.5] = 1.0
    out[values < 0.5] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Gold evaluation
# --------------------------------------------------------------------------- #

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if (y_true == 1).sum() < MIN_CLASS_COUNT or (y_true == 0).sum() < MIN_CLASS_COUNT:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def per_label_metrics(votes: np.ndarray, gold_vals: np.ndarray) -> dict:
    """Coverage / accuracy / AUC of one source on gold cells it addresses."""
    addressed = np.isfinite(votes) & (votes != MISSING)
    n = int(addressed.sum())
    out = {"n_gold": int(len(gold_vals)), "n_addressed": n,
           "coverage": round(n / len(gold_vals), 4) if len(gold_vals) else 0.0}
    if n == 0:
        out.update(accuracy=None, auc=None)
        return out
    y = gold_vals[addressed]
    v = votes[addressed]
    out["accuracy"] = round(float((binarize(v) == y).mean()), 4)
    out["auc"] = (None if np.isnan(a := safe_auc(y, v)) else round(a, 4))
    return out


def evaluate_on_gold(lf: pd.DataFrame, teachers: dict[str, pd.DataFrame],
                     gold: pd.DataFrame) -> dict:
    """Per-label metrics for the LF and every teacher on the 58 gold studies.

    For each label the LF is also compared against each teacher restricted to
    the gold cells the LF addresses (same-cell comparison; teacher cells that
    abstain are dropped and n reported)."""
    gold_idx = gold.set_index("StudyInstanceUID")[LABELS].astype(float)
    lf_idx = lf.set_index("StudyInstanceUID")[LABELS]
    common = [u for u in gold_idx.index if u in lf_idx.index]
    gold_vals = gold_idx.loc[common]

    result: dict[str, dict] = {}
    for label in LF_LABELS:
        y = gold_vals[label].to_numpy(dtype=float)
        lf_votes = lf_idx.loc[common, label].to_numpy(dtype=float)
        entry = {"rulebased_v1": per_label_metrics(lf_votes, y),
                 "teachers": {}, "same_cell_vs_teachers": {}}
        lf_addr = (lf_votes != MISSING)
        for name, tdf in teachers.items():
            t_idx = tdf.set_index("StudyInstanceUID")[LABELS]
            t_common = [u for u in common if u in t_idx.index]
            t_all = t_idx.loc[t_common, label].to_numpy(dtype=float)
            y_all = gold_idx.loc[t_common, label].to_numpy(dtype=float)
            entry["teachers"][name] = per_label_metrics(t_all, y_all)
            # Same-cell: LF-addressed gold cells where the teacher also votes.
            t_on_lf = t_idx[label].reindex(common).to_numpy(dtype=float)
            both = lf_addr & np.isfinite(t_on_lf) & (t_on_lf != MISSING)
            n_both = int(both.sum())
            acc = (round(float((binarize(t_on_lf[both]) == y[both]).mean()), 4)
                   if n_both else None)
            entry["same_cell_vs_teachers"][name] = {
                "n": n_both, "teacher_accuracy_on_lf_cells": acc,
                "lf_accuracy_on_lf_cells": (
                    round(float((binarize(lf_votes[lf_addr]) == y[lf_addr]).mean()), 4)
                    if int(lf_addr.sum()) else None)}
        result[label] = entry
    return result


SOFT_DISAGREE_SPREAD = 0.3  # relaxed "soft disagreement": vote spread >= this


def _teacher_vote_stack(mats: dict[str, pd.DataFrame], label: str,
                        uids) -> np.ndarray:
    """(n_uids, n_teachers) matrix of soft votes aligned on `uids`."""
    return np.column_stack([
        mats[n][label].reindex(uids).to_numpy(dtype=float) for n in mats])


def _tiebreak_block(lf_votes: np.ndarray, y: np.ndarray,
                    cells: np.ndarray, votes: np.ndarray) -> dict:
    """LF tiebreak stats on a boolean cell selection, majority as context."""
    n = int(cells.sum())
    entry = {"n_cells": n}
    if not n:
        return entry
    lf_d, y_d, v_d = lf_votes[cells], y[cells], votes[cells]
    voted = lf_d != MISSING
    entry["lf_abstains"] = int((~voted).sum())
    entry["lf_votes"] = int(voted.sum())
    entry["lf_correct"] = (int((binarize(lf_d[voted]) == y_d[voted]).sum())
                           if voted.any() else 0)
    entry["lf_tiebreak_accuracy"] = (round(entry["lf_correct"] / entry["lf_votes"], 4)
                                     if entry["lf_votes"] else None)
    hard = binarize(np.where(np.isfinite(v_d), v_d, np.nan))
    n_pos = np.nansum(hard == 1.0, axis=1)
    n_neg = np.nansum(hard == 0.0, axis=1)
    maj_vote = np.where(n_pos > n_neg, 1.0, np.where(n_neg > n_pos, 0.0, np.nan))
    decided = np.isfinite(maj_vote)
    entry["public_majority_accuracy"] = (
        round(float((maj_vote[decided] == y_d[decided]).mean()), 4)
        if decided.any() else None)
    entry["public_majority_ties"] = int((~decided).sum())
    return entry


def tiebreak_analysis(lf: pd.DataFrame, teachers: dict[str, pd.DataFrame],
                      gold: pd.DataFrame) -> dict:
    """LF behavior on cells where the public LLM teachers disagree.

    Three lenses per label:
    - strict (gold): >=1 public teacher hard-votes 1 and >=1 hard-votes 0.
      On the 58 gold studies this is rare — the teachers' disagreements are
      mostly in soft ranking, not hard calls — so counts may be ~0.
    - soft (gold): vote spread (max-min over non-abstaining public teachers)
      >= SOFT_DISAGREE_SPREAD; LF accuracy vs gold where the LF votes.
    - corpus (no gold available): hard disagreement over all 4407 studies;
      only the LF's vote distribution is reported (shows the LF engages on
      contested cells rather than abstaining)."""
    gold_idx = gold.set_index("StudyInstanceUID")[LABELS].astype(float)
    lf_idx = lf.set_index("StudyInstanceUID")[LABELS]
    t_mats = {n: t.set_index("StudyInstanceUID")[LABELS]
              for n, t in teachers.items() if n in TIEBREAK_TEACHERS}
    gold_common = [u for u in gold_idx.index if u in lf_idx.index]
    corpus_uids = list(lf_idx.index)

    out: dict[str, dict] = {}
    for label in LF_LABELS:
        y = gold_idx.loc[gold_common, label].to_numpy(dtype=float)
        lf_gold = lf_idx.loc[gold_common, label].to_numpy(dtype=float)
        v_gold = _teacher_vote_stack(t_mats, label, gold_common)
        hard_gold = binarize(np.where(np.isfinite(v_gold), v_gold, np.nan))
        strict = (np.nansum(hard_gold == 1.0, axis=1) > 0) & \
                 (np.nansum(hard_gold == 0.0, axis=1) > 0)
        nonabstain = np.where(v_gold == MISSING, np.nan, v_gold)
        spread = np.nanmax(nonabstain, axis=1) - np.nanmin(nonabstain, axis=1)
        soft = np.where(np.isfinite(spread), spread, 0.0) >= SOFT_DISAGREE_SPREAD

        lf_corpus = lf_idx[label].to_numpy(dtype=float)
        v_corpus = _teacher_vote_stack(t_mats, label, corpus_uids)
        hard_corpus = binarize(np.where(np.isfinite(v_corpus), v_corpus, np.nan))
        strict_corpus = (np.nansum(hard_corpus == 1.0, axis=1) > 0) & \
                        (np.nansum(hard_corpus == 0.0, axis=1) > 0)
        lf_on_corpus = lf_corpus[strict_corpus]

        out[label] = {
            "strict_gold": _tiebreak_block(lf_gold, y, strict, v_gold),
            f"soft_gold_spread>={SOFT_DISAGREE_SPREAD}":
                _tiebreak_block(lf_gold, y, soft, v_gold),
            "strict_corpus": {
                "n_cells": int(strict_corpus.sum()),
                "lf_pos": int((lf_on_corpus == 1.0).sum()),
                "lf_neg": int((lf_on_corpus == 0.0).sum()),
                "lf_abstain": int((lf_on_corpus == MISSING).sum()),
            },
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected name=path, got {spec!r}")
    name, _, raw = spec.partition("=")
    return name.strip(), Path(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path,
                        default=PROJECT_ROOT / "input" / "train.csv")
    parser.add_argument("--gold-labels", type=Path,
                        default=PROJECT_ROOT / "input" / "gold_labels.csv")
    parser.add_argument("--teacher", action="append", type=parse_named_path,
                        default=[], metavar="name=path",
                        help="comparison teacher (wide or detailed CSV); "
                             "repeatable. Defaults to the built-in teacher set.")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "input" / "rulebased_labels_v1.csv")
    parser.add_argument("--eval", type=Path,
                        default=PROJECT_ROOT / "input" / "rulebased_labels_v1_eval.json")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing outputs (never silent)")
    args = parser.parse_args()

    for path in (args.output, args.eval):
        if path.exists() and not args.force:
            parser.error(f"{path} already exists — pass --force to overwrite")

    # 1. Parse train.csv robustly (embedded newlines in Report).
    train = pd.read_csv(args.train, dtype=str)
    if not {"StudyInstanceUID", "Report"} <= set(train.columns):
        parser.error(f"unexpected train.csv columns: {list(train.columns)}")
    LOGGER.info("train: %d studies with reports", len(train))

    # 2. Run the labeling functions.
    lf = apply_lfs(train)
    cov = {label: round(float((lf[label] != MISSING).mean()), 4)
           for label in LF_LABELS}
    LOGGER.info("LF coverage over %d studies: %s", len(lf), cov)
    values = lf[LABELS].to_numpy(dtype=float)
    if not np.isin(values, [0.0, 0.5, 1.0]).all():
        raise RuntimeError("LF emitted a value outside {0, 0.5, 1} — refusing to write")
    atomic_write_csv(lf, args.output)
    LOGGER.info("wrote %s (%d studies)", args.output, len(lf))

    # 3. Load comparison teachers + gold.
    teacher_paths = dict(DEFAULT_TEACHERS)
    teacher_paths.update(dict(args.teacher))
    teachers: dict[str, pd.DataFrame] = {}
    for name, path in teacher_paths.items():
        if not path.is_file():
            LOGGER.warning("teacher %s missing at %s — skipped", name, path)
            continue
        teachers[name] = load_teacher(path)
        LOGGER.info("teacher %s: %d studies from %s", name, len(teachers[name]), path)

    gold = pd.read_csv(args.gold_labels, dtype={"StudyInstanceUID": str})
    gold[LABELS] = gold[LABELS].astype(float)
    LOGGER.info("gold: %d studies", len(gold))

    # 4. Evaluate.
    gold_eval = evaluate_on_gold(lf, teachers, gold)
    tiebreak = tiebreak_analysis(lf, teachers, gold)
    for label in LF_LABELS:
        m = gold_eval[label]["rulebased_v1"]
        tb = tiebreak[label]
        LOGGER.info("%-10s LF coverage=%.3f acc=%s auc=%s | strict=%s soft=%s corpus=%s",
                    label, m["coverage"], m["accuracy"], m["auc"],
                    tb["strict_gold"].get("n_cells"),
                    tb[[k for k in tb if k.startswith("soft_gold")][0]].get("n_cells"),
                    tb["strict_corpus"]["n_cells"])

    # 5. Audit JSON.
    audit = {
        "script": Path(__file__).name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gold_usage": "evaluation",
        "honesty_note": (
            "LF patterns were designed from corpus-wide phrasing inspection "
            "that included some gold-row reports, so gold agreement numbers "
            "are mildly in-sample. Gold labels are themselves noisy relative "
            "to report phrasing (e.g. 'moderate joint effusion' labeled 0, "
            "a measured Baker cyst labeled 0), which caps achievable "
            "agreement. No numeric threshold was fit on gold."),
        "lf_labels": LF_LABELS,
        "vote_semantics": {"positive": 1.0, "negative": 0.0,
                           "abstain/hedge/conflict": MISSING},
        "n_train_studies": int(len(lf)),
        "corpus_coverage": cov,
        "corpus_vote_distribution": {
            label: {"pos": int((lf[label] == 1.0).sum()),
                    "neg": int((lf[label] == 0.0).sum()),
                    "abstain": int((lf[label] == MISSING).sum())}
            for label in LF_LABELS},
        "gold_evaluation": gold_eval,
        "tiebreak_analysis": {
            "teachers_used": [n for n in TIEBREAK_TEACHERS if n in teachers],
            "definition": {
                "strict_gold": ("gold cells where >=1 public teacher hard-votes 1 "
                                "and >=1 hard-votes 0 (vote>0.5 / <0.5; 0.5=abstain)"),
                "soft_gold": ("gold cells with public-teacher vote spread >= "
                              f"{SOFT_DISAGREE_SPREAD} among non-abstaining votes"),
                "strict_corpus": ("same hard-disagreement rule over all 4407 "
                                  "studies; no gold there, so only the LF vote "
                                  "distribution is reported"),
                "lf_tiebreak_accuracy": ("fraction of LF non-abstain votes on the "
                                         "selected cells that match gold"),
            },
            "per_label": tiebreak,
        },
        "inputs": {
            "train": {"path": str(args.train), "sha256": sha256_file(args.train)},
            "gold_labels": {"path": str(args.gold_labels),
                            "sha256": sha256_file(args.gold_labels)},
            "teachers": {n: {"path": str(p), "sha256": sha256_file(p)}
                         for n, p in teacher_paths.items() if p.is_file()},
        },
        "output": {"path": str(args.output), "sha256": sha256_file(args.output),
                   "n_rows": int(len(lf))},
    }
    atomic_write_json(audit, args.eval)
    LOGGER.info("wrote %s", args.eval)
    LOGGER.info("Done")


if __name__ == "__main__":
    main()
