from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "path5_supervision_fusion_train" / "rsna_knee_path5_supervision_fusion_train.ipynb"
OUTPUT = Path(__file__).resolve().parent / "rsna_knee_path6_kimi_fs160_train.ipynb"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, observed {count}")
    return source.replace(old, new, 1)


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    notebook["cells"][0]["source"] = (
        "# Path 6 — Kimi FS/160mm clean E11 training\n\n"
        "Supervision policy stays identical to Path 5. This experiment applies the "
        "Kimi diagnostic routing guide to the image arm: fat-suppressed sagittal, "
        "coronal, and axial series are restored as primary slots, a coronal non-FS "
        "morphology slot is retained, crop expands to 160 mm, and the slice band widens "
        "to reduce edge clipping for MCL, effusion, Baker cyst tails, contusion, and "
        "fracture-line cases. Gold labels remain monitor-only.\n"
    )

    source = "".join(notebook["cells"][32]["source"])
    source = replace_once(
        source,
        '"mode": "path5-e11-supervision-fusion-clean-training-only",',
        '"mode": "path6-kimi-fs160-clean-training-only",',
        "audit mode",
    )
    source = replace_once(
        source,
        '"Path 3 — clean diverse E11 training"',
        '"Path 6 — Kimi FS/160mm clean training"',
        "title marker",
    ) if '"Path 3 — clean diverse E11 training"' in source else source
    source = replace_once(
        source,
        "SLICE_BAND = (0.12, 0.88)",
        "SLICE_BAND = (0.05, 0.95)",
        "wider slice band",
    )
    old_slots = '''E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", None, False),
    ("COR_NOFS", "Coronal", None, False),
    ("AX_NOFS", "Axial", None, False),
    ("SAG_FS", "Sagittal", None, True),
]
E11_CROP_MM = 130.0'''
    new_slots = '''E11_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("COR_NOFS", "Coronal", None, False),
]
E11_CROP_MM = 160.0'''
    source = replace_once(source, old_slots, new_slots, "Kimi FS160 slots")
    source = replace_once(
        source,
        '"non-suppressed slots plus 1 suppressed anchor at a 130 mm physical crop"',
        '"fat-suppressed sagittal/coronal/axial slots plus coronal non-FS morphology at a 160 mm physical crop"',
        "recipe description",
    )
    source = replace_once(
        source,
        "Path5 E11 masked scanner-grouped weak OOF macro AUC",
        "Path6 Kimi FS160 masked scanner-grouped weak OOF macro AUC",
        "weak OOF log",
    )
    source = replace_once(
        source,
        "Path5 E11 gold monitor macro AUC",
        "Path6 Kimi FS160 gold monitor macro AUC",
        "gold monitor log",
    )
    source = replace_once(
        source,
        '"version": "path5-e11-supervision-fusion-clean-1",',
        '"version": "path6-kimi-fs160-clean-1",',
        "bundle version",
    )
    source = replace_once(
        source,
        'audit["status"] = "PATH5_E11_SUPERVISION_FUSION_TRAINED_PARENT_PRESERVED"',
        'audit["status"] = "PATH6_KIMI_FS160_TRAINED_PARENT_PRESERVED"',
        "status",
    )
    old_rationale = (
        '"differs_from_parent_arm": (\n'
        '                "parent reads 3 fat-suppressed slots at full frame; this reads 3 "\n'
        '                "non-suppressed slots plus 1 suppressed anchor at a 130 mm physical crop"\n'
        '            ),'
    )
    if old_rationale in source:
        source = replace_once(
            source,
            old_rationale,
            '"differs_from_parent_arm": (\n'
            '                "parent reads 3 fat-suppressed slots at full frame; this reads the "\n'
            '                "same FS diagnostic core plus coronal non-FS morphology at 160 mm, "\n'
            '                "with a wider 5-95% slice band for crop-risk targets"\n'
            '            ),',
            "audit recipe rationale",
        )
    notebook["cells"][32]["source"] = [line + "\n" for line in source.splitlines()]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
