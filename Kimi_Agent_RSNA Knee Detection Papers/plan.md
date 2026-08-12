# Plan — RSNA Knee Abnormality Detection (Kaggle) Seed-Knowledge Research

## Context
User registered for the Kaggle competition "RSNA Knee Abnormality Detection" and has NO medical or domain background. Working platform: Kimi Code (agentic coding). Goal: a comprehensive research dossier + curated seed research articles so a non-specialist can bootstrap the project.

## Stage 1 — Competition Reconnaissance (deep-research-swarm, Route B)
- Verify the competition page (kaggle.com/competitions/rsna-knee-abnormality-detection): task definition, modality (X-ray vs MRI vs both), labels/abnormality classes, dataset size, evaluation metric, timeline, prizes, notebook/compute rules, code competition constraints.
- If the exact competition page is unreachable/new, anchor to the RSNA dataset series and nearest analogs (RSNA bone age, MURA, MRNet).
- Subagent A (explore): scrape/search competition page + Kaggle discussion threads + starter notebooks.

## Stage 2 — Domain & Literature Research (parallel subagents)
- Subagent B (explore): Clinical background for laypersons — knee anatomy basics, common abnormalities detectable on imaging (OA, fractures, effusion, ligament/meniscus tears), how radiologists read knee images, grading systems (Kellgren-Lawrence etc.).
- Subagent C (explore + scholar plugin): Seed research articles — MRNet (Stanford knee MRI), MURA, Knee Osteoarthritis severity grading (Kaggle KL datasets), deep learning for musculoskeletal imaging, prior RSNA competition winning solutions & GitHub repos. Deliver: annotated paper list with links, why each matters.
- Subagent D (explore): Technical seed knowledge — preprocessing for radiographs/MRI, common architectures (ResNet/EfficientNet/ViT), augmentation, class imbalance, metric optimization strategies from past RSNA competitions.

## Stage 3 — Synthesis & Deliverable
- Orchestrator synthesizes a structured research brief (Markdown): competition overview, domain primer, dataset & EDA guide, modeling roadmap for Kimi Code workflow, annotated bibliography of seed papers/articles, practical next steps.
- Convert to .docx via docx skill (writing default output).

## Validation
- Every factual claim about the competition tied to a cited source; papers must have verifiable links (DOI/arXiv/pubmed).
