"""Produce a schema-valid all-0.5 dry-run submission on Kaggle."""

from pathlib import Path

import pandas as pd


LABEL_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
EXPECTED_COLUMNS = ["StudyInstanceUID", *LABEL_COLUMNS]


def find_competition_file(filename: str, input_root: Path = Path("/kaggle/input")) -> Path:
    candidates = [
        input_root / "competitions/rsna-knee-abnormality-detection" / filename,
        input_root / "rsna-knee-abnormality-detection" / filename,
        input_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(input_root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename!r} under {input_root}; found {matches}"
        )
    return matches[0]


def main(
    input_root: Path = Path("/kaggle/input"),
    output_path: Path = Path("/kaggle/working/submission.csv"),
) -> None:
    sample_path = find_competition_file("sample_submission.csv", input_root)
    submission = pd.read_csv(sample_path)

    if list(submission.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected submission columns: {list(submission.columns)!r}; "
            f"expected {EXPECTED_COLUMNS!r}"
        )
    if submission.empty:
        raise ValueError("The competition sample submission contains no studies")
    if submission["StudyInstanceUID"].isna().any() or not submission["StudyInstanceUID"].is_unique:
        raise ValueError("StudyInstanceUID values must be non-null and unique")

    submission.loc[:, LABEL_COLUMNS] = 0.5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    written = pd.read_csv(output_path)
    if list(written.columns) != EXPECTED_COLUMNS or len(written) != len(submission):
        raise RuntimeError("Written submission failed schema or row-count validation")
    if not written[LABEL_COLUMNS].eq(0.5).all().all():
        raise RuntimeError("Written dry-run predictions are not uniformly 0.5")

    print(f"Wrote validated submission: {output_path} ({len(written)} studies)")


if __name__ == "__main__":
    main()
