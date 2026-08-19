"""
Silver label extraction from multilingual radiology reports.

Primary: LLM-based extraction (GPT-4o-mini or open-weight Qwen3).
Fallback: Rule-based CheXpert-style labeler.

Key insight from research: LLM-extracted labels score 0.878 macro AUC
vs 0.814 for regex on the 58 gold studies. Label quality is the single
largest controllable lever.

IMPORTANT: 25.4% of report-label cells are "not addressed" — these must
be mapped to soft label 0.5 (masked in loss), never coerced to 0.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FindingValue(Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"
    NOT_ADDRESSED = "not_addressed"
    LATERALITY_AMBIGUOUS = "laterality_ambiguous"


@dataclass
class ExtractionResult:
    """Result of extracting one finding from a report."""
    finding: str
    value: FindingValue
    evidence: str = ""
    confidence: float = 1.0


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


def build_extraction_prompt(report_text: str, language: str = "unknown") -> str:
    """
    Build the LLM prompt for extracting labels from a radiology report.

    Includes official positivity thresholds and strict JSON output format.
    """
    thresholds_text = "\n".join(
        f"- {label}: {threshold}" for label, threshold in POSITIVITY_THRESHOLDS.items()
    )

    prompt = f"""You are a musculoskeletal radiology expert. Extract findings from this knee MRI report.

Report language: {language}
Report text:
---
{report_text}
---

For each of the 12 findings below, classify as one of:
- "present": The finding is clearly described as present, meeting the positivity threshold
- "absent": The finding is explicitly described as absent or normal
- "uncertain": The finding is mentioned but with uncertainty (e.g., "possible", "cannot exclude")
- "not_addressed": The finding is not mentioned in the report at all
- "laterality_ambiguous": The finding is mentioned but the side (medial/lateral) is unclear

Official positivity thresholds (borderline = negative):
{thresholds_text}

Output strict JSON with exactly these 12 keys:
{{
  "ACL": {{"value": "...", "evidence": "..."}},
  "MCL": {{"value": "...", "evidence": "..."}},
  "Medial Meniscus": {{"value": "...", "evidence": "..."}},
  "Lateral Meniscus": {{"value": "...", "evidence": "..."}},
  "Medial OA": {{"value": "...", "evidence": "..."}},
  "Lateral OA": {{"value": "...", "evidence": "..."}},
  "PF OA": {{"value": "...", "evidence": "..."}},
  "Effusion": {{"value": "...", "evidence": "..."}},
  "Synovitis": {{"value": "...", "evidence": "..."}},
  "Baker's": {{"value": "...", "evidence": "..."}},
  "Contusion": {{"value": "...", "evidence": "..."}},
  "Fracture": {{"value": "...", "evidence": "..."}}
}}

Rules:
- "evidence" should be the exact quote from the report supporting your classification, or empty string if not_addressed.
- If the report does not mention a finding, use "not_addressed" — do NOT guess.
- If a finding is described but does not meet the positivity threshold, use "absent".
- For medial/lateral pairs, only use "laterality_ambiguous" if the finding is present but side is truly unclear.
"""
    return prompt


def build_soft_extraction_prompt(report_text: str, language: str = "unknown") -> str:
    """Build a severity-preserving prompt for image-label probability ranking.

    The categorical value remains useful for auditing, while ``score`` avoids
    collapsing normal, borderline, and near-threshold report descriptions into
    the same value. This matters because the competition metric is ROC AUC.
    """
    thresholds_text = "\n".join(
        f"- {label}: {threshold}" for label, threshold in POSITIVITY_THRESHOLDS.items()
    )

    return f"""You are a musculoskeletal radiology expert creating training targets for a knee MRI competition.

Your goal is to estimate, from the REPORT ONLY, how likely an independent expert image review would mark each finding positive under the competition rubric. Reports can under-call or over-call the image findings. Preserve severity as a continuous score so normal, mild, borderline, and definite disease are not tied.

First identify the report language and interpret negation, uncertainty, severity,
anatomical compartment, and chronic versus acute wording in that language. Do not
translate medial/lateral into the patient's left/right side: they are anatomical
compartments within the imaged knee.

Report language: {language}
Report text:
---
{report_text}
---

For every finding return:
- "value": one of "present", "absent", "uncertain", "not_addressed", "laterality_ambiguous"
- "score": a number from 0.0 to 1.0 ranking the probability that expert IMAGE review meets the positivity threshold
- "evidence": the exact supporting quote, or an empty string when not addressed

Official positivity thresholds (borderline is negative):
{thresholds_text}

Continuous-score anchors:
- 0.00-0.10: explicitly normal/absent
- 0.15-0.35: mild, minimal, grade I-II, incipient, focal below threshold, or weak indirect evidence
- 0.40-0.60: uncertain, incompletely characterized, not addressed with meaningful correlated findings, or close to threshold
- 0.65-0.85: likely positive or high-grade without complete size detail
- 0.90-1.00: definite and clearly meets threshold

Label-specific ranking rules:
- OA: distinguish intact cartilage (near 0), mild thinning/grade I-II/incipient OA (low but nonzero), high-grade or full-thickness loss without enough size detail (intermediate-high), and >=1 cm high-grade loss (near 1). Consider joint-space narrowing and osteophytes as supporting evidence, but do not call them definitive alone.
- Effusion: no fluid (near 0), trace/small/mild fluid (low), moderate fluid (intermediate-high), large/marked fluid (near 1).
- Synovitis: explicit synovial thickening/proliferation/inflammation drives the score. When synovitis is not addressed, moderate/large effusion or advanced OA may raise the score modestly; absence of effusion lowers it. Do not label every effusion as definite synovitis.
- Baker's: a small popliteal/Baker cyst is below threshold; moderate, large, dissecting, or ruptured cyst is high. Do not confuse other posterior cysts with a Baker cyst.
- Meniscus: degeneration or intrasubstance signal not reaching a surface is low; a surface-reaching tear on adequate evidence, root tear, displaced fragment, ghost/truncation, or definite morphologic tear is high.
- ACL: sprain/low-grade partial injury is below the >50% disruption threshold; high-grade partial or complete rupture is high.
- MCL: periligamentous edema or grade-I sprain with intact fibers is low; grade-II partial disruption or grade-III discontinuity is high.
- Contusion: marrow edema explicitly attributed to degeneration, OA, fracture, or another non-traumatic cause is not automatically a bone contusion.
- Fracture: marrow edema alone is not a fracture; an explicit fracture line, impaction, insufficiency, or subchondral fracture is high.
- Not addressed: use a near-neutral score unless a label-specific correlated finding above provides real evidence. Do not turn report silence into a confident negative except when the report explicitly gives a normal survey of that structure.

Output ONLY strict JSON with exactly these 12 keys and no commentary:
{{
  "ACL": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "MCL": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Medial Meniscus": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Lateral Meniscus": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Medial OA": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Lateral OA": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "PF OA": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Effusion": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Synovitis": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Baker's": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Contusion": {{"value": "...", "score": 0.0, "evidence": "..."}},
  "Fracture": {{"value": "...", "score": 0.0, "evidence": "..."}}
}}"""


def build_soft_batch_extraction_prompt(
    reports: Sequence[Tuple[str, str]],
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


class LabelExtractor:
    """
    Extracts silver labels from radiology reports.

    Supports LLM-based extraction (primary) and rule-based fallback.
    """

    def __init__(self, method: str = "llm"):
        """
        Args:
            method: "llm" or "rules"
        """
        self.method = method

    def extract_from_llm_response(
        self,
        response_text: str,
    ) -> List[ExtractionResult]:
        """
        Parse LLM JSON response into ExtractionResults.
        """
        # Try to extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {response_text[:200]}")
            return [
                ExtractionResult(finding=label, value=FindingValue.NOT_ADDRESSED)
                for label in LABELS
            ]

        results = []
        for label in LABELS:
            entry = data.get(label, {})
            value_str = entry.get("value", "not_addressed")
            evidence = entry.get("evidence", "")

            try:
                value = FindingValue(value_str)
            except ValueError:
                value = FindingValue.NOT_ADDRESSED

            results.append(ExtractionResult(
                finding=label,
                value=value,
                evidence=evidence,
            ))

        return results

    def extract_rules(self, report_text: str, language: str = "en") -> List[ExtractionResult]:
        """
        Rule-based extraction fallback.

        Uses keyword matching with negation detection.
        This is a simplified version — the full multilingual labeler
        requires per-language phrase files.
        """
        results = []
        text_lower = report_text.lower()

        # Simple keyword patterns for each finding
        # This is a starting point — the full labeler needs per-language files
        keyword_map = {
            "ACL": [r"\bacl\b", r"anterior cruciate"],
            "MCL": [r"\bmcl\b", r"medial collateral"],
            "Medial Meniscus": [r"medial meniscus", r"\bmm\b.*tear"],
            "Lateral Meniscus": [r"lateral meniscus", r"\blm\b.*tear"],
            "Medial OA": [r"medial.*(?:osteoarthritis|oa|cartilage loss|chondral)"],
            "Lateral OA": [r"lateral.*(?:osteoarthritis|oa|cartilage loss|chondral)"],
            "PF OA": [r"patellofemoral.*(?:osteoarthritis|oa|cartilage)"],
            "Effusion": [r"effusion", r"joint fluid", r"fluid.*joint"],
            "Synovitis": [r"synovitis", r"synovial.*(?:thickening|inflammation)"],
            "Baker's": [r"baker", r"popliteal cyst"],
            "Contusion": [r"contusion", r"bone bruise", r"bone marrow edema", r"\bbml\b"],
            "Fracture": [r"fracture", r"broken", r"cortical.*disruption"],
        }

        negation_patterns = [
            r"no\s+(?:evidence\s+of\s+)?",
            r"(?:not|n't)\s+",
            r"without\s+",
            r"absence\s+of\s+",
            r"negative\s+for\s+",
            r"rule[sd]?\s+out\s+",
        ]

        for label, patterns in keyword_map.items():
            found = False
            negated = False

            for pattern in patterns:
                for match in re.finditer(pattern, text_lower):
                    found = True
                    # Check for negation before the match
                    prefix = text_lower[:match.start()]
                    for neg in negation_patterns:
                        if re.search(neg + r"$", prefix):
                            negated = True
                            break
                    if not negated:
                        # Check for negation in the same sentence
                        sentence_start = max(0, prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
                        sentence = text_lower[sentence_start:match.start()]
                        for neg in negation_patterns:
                            if re.search(neg, sentence):
                                negated = True
                                break

            if not found:
                results.append(ExtractionResult(
                    finding=label, value=FindingValue.NOT_ADDRESSED
                ))
            elif negated:
                results.append(ExtractionResult(
                    finding=label, value=FindingValue.ABSENT
                ))
            else:
                results.append(ExtractionResult(
                    finding=label, value=FindingValue.PRESENT
                ))

        return results

    def extract(
        self,
        report_text: str,
        language: str = "unknown",
        llm_client=None,
    ) -> List[ExtractionResult]:
        """
        Extract labels from a report.

        Args:
            report_text: The radiology report text.
            language: Detected language code.
            llm_client: Optional LLM client (e.g., openai.OpenAI).

        Returns:
            List of 12 ExtractionResults, one per label.
        """
        if self.method == "llm" and llm_client is not None:
            prompt = build_extraction_prompt(report_text, language)
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
            )
            return self.extract_from_llm_response(
                response.choices[0].message.content
            )
        else:
            return self.extract_rules(report_text, language)
