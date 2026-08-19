# Section 04 — Anatomical Routing, Label Policy, and Crop Risk (RSNA Knee Abnormality Detection)

Scope: 12 binary labels; metadata = Anatomical_Plane, Fluid_Sensitive, Fat_Suppression only (no named sequences). "FS PD/T2" below = any fluid-sensitive fat-suppressed series; "PD (non-FS ok)" = non-fluid-sensitive PD/T1-type series usable for morphology. Typical FOV 140–160 mm; ~20–45 slices/series.

## Part 1 — Anatomical Routing Table

| Finding | Primary plane | Primary sequence | Anatomical region | Adjacent slices needed | Crop type | Main confounders |
|---|---|---|---|---|---|---|
| ACL | sagittal (confirm coronal + axial) | FS PD/T2 | intercondylar notch; Blumensaat line; femoral origin on LFC, tibial spine insertion | 1 if unequivocal discontinuity/nonvisualization, else 2 planes (no two-slice rule; one-slice signal alone insufficient) | 130 mm OK — keep notch, LFC sulcus, posterolateral tibia (contusion pattern) | mucoid degeneration (celery-stalk, intact fibers); partial-volume at femoral origin; normal striation / magic-angle; ACL ganglion |
| MCL | coronal (confirm axial) | FS PD/T2 | medial femoral epicondyle (~31 mm above joint line) → superficial tibial insertion (~62 mm below joint line) | inspect full course (~3–6 coronal); focal avulsion may occupy only 1–2 | 130 mm RISKY — verify full medial course (≥75 mm below joint line) and medial soft tissues | MCL bursitis; reactive periligamentous edema (meniscus/OA/effusion); pes anserine bursitis; chronic thickening / Pellegrini-Stieda |
| Medial Meniscus | sagittal (confirm coronal/axial) | PD (non-FS ok) + FS PD/T2 | medial body (bow-tie), posterior horn, posterior root (~8 mm anterior to PCL insertion) | ≥2 touching slices (two-slice rule; 1 sag + 1 cor acceptable); unequivocal morphology/displaced fragment overrides | 130 mm OK | grade 1–2 intrasubstance signal (NEGATIVE); flounce / chondrocalcinosis / ossicle; displaced fragment in notch/recesses; postoperative signal (≥25% resection/repair) |
| Lateral Meniscus | sagittal (confirm coronal/axial) | PD (non-FS ok) + FS PD/T2 | lateral body/horns, popliteus hiatus, posterior root (~11 mm posterior to ACL) | ≥2 touching slices (two-slice rule); morphology overrides | 130 mm OK | popliteus tendon pseudotear; meniscofemoral ligaments (Humphrey/Wrisberg); magic angle in posterior horn/root; discoid meniscus (bow-tie ≥3 consecutive 5 mm slices) |
| Medial OA | coronal (sagittal for posterior margins) | FS PD (non-FS PD/T1 best for osteophytes) | medial femoral condyle + medial tibial plateau weight-bearing cartilage; marginal/spine osteophytes | host rule: ≥1 cm of >50% cartilage loss → defect spans ~3–4 slices at 3–4 mm; survey whole compartment | 130 mm OK | focal traumatic chondral defect; WORMS-1 signal-only change (NEGATIVE); equivocal/<3 mm osteophyte (NEGATIVE); BML without cartilage loss |
| Lateral OA | coronal (sagittal for posterior margins) | FS PD (non-FS PD/T1 best for osteophytes) | lateral femoral condyle + lateral tibial plateau weight-bearing cartilage | as Medial OA (≥1 cm of >50% cartilage loss) | 130 mm OK | same as Medial OA; contusion-pattern BML overlying cartilage |
| PF OA | axial (confirm sagittal) | FS PD | patellar facets + trochlear groove; PF marginal osteophytes | as above; assess full retropatellar stack | 130 mm OK on axial | small patellar-pole osteophytes <3 mm (NEGATIVE); partial-volume at facet apex/ridge; plica-related cartilage wear; overlying effusion |
| Effusion | axial for grading (sagittal for measurement) | FS PD/T2 | suprapatellar pouch ABOVE patella (midline sagittal AP diameter; axial retropatellar space) | 1 representative slice suffices; max AP on midline sagittal; host: moderate/large (MOAKS 2–3 ≈ ≥5 mm AP) | 130 mm RISKY on sagittal — superior pouch extends above patella | trace physiologic fluid (NEGATIVE); synovitis inseparable on non-contrast; debris/hemarthrosis |
| Synovitis | axial + sagittal | FS PD/T2 (no contrast in dataset) | Hoffa fat pad; suprapatellar pouch walls; medial/lateral gutters | no slice-count rule; look for frond-like/nodular INTERMEDIATE-signal thickening on ≥2 planes if possible | 130 mm OK | effusion — inseparable on non-contrast (Hoffa signal specificity 10–38%); plica (<3 mm normal); PVNS/TGCT (low T1+T2, blooming); lipoma arborescens (fat signal) |
| Baker's Cyst | axial (sagittal for extent/rupture) | FS PD/T2 | posteromedial popliteal fossa; neck between semimembranosus tendon and medial gastrocnemius head AT joint line | neck diagnostic on 1–2 axial slices ("speech-bubble"); scroll sagittal for rupture tracking | neck inside FOV — 130 mm OK on axial/coronal; RISKY for large/ruptured cyst tails on sagittal/inferior axial | popliteal artery aneurysm (flow artifact/continuity — don't miss); ganglion (no neck); small physiologic bursal fluid (NEGATIVE); hematoma/abscess |
| Contusion | all three planes | FS PD/T2 (+ T1 if present, to EXCLUDE a line) | subchondral marrow; pivot-shift pattern: LFC sulcus + posterolateral tibial plateau; dislocation pattern: medial patellar facet + LFC | edema geographic/ill-defined across many slices; NO line in ANY plane | 130 mm RISKY — posterolateral tibial plateau contusions sit near crop edge | occult fracture (line ≥2 slices + 2 planes); SIFK (subchondral low line + edema, older MFC); OA BML; persistent bruise (median healing 42 weeks) |
| Fracture | plane best showing the line; confirm in 2nd orthogonal plane | FS PD/T2 (T1 if present — most specific for the line) | cortical step-off, subchondral impaction, low-signal line within edema; plateau rims (coronal), anterior/posterior plateau (sagittal), patella (axial) | line visible in ≥2 contiguous slices + 2 planes; single-slice single-plane line = artifact/volume averaging | 130 mm OK (same edge caveat as Contusion) | volume averaging on one slice; contusion without line (label Contusion); SIFK vs osteonecrosis; absent T1 lowers specificity |

## Part 2 — Label Policy Summary + Crop Risk

### Official positive thresholds (host, discussion/733343; double-read MSK radiologists, 3rd-reader adjudication)

- **ACL**: positive = high-grade partial (>50% fibers) or full tear. Negative: intact fibers, isolated mucoid degeneration, one-slice signal only, isolated secondary signs (anterior translation, notch sign, contusion pattern).
- **MCL**: positive = high-grade acute tear (≈ MRI grade II–III: partial fiber disruption or complete discontinuity). Negative: grade I sprain/periligamentous edema with intact fibers; chronic thickened ligament without edema; bursitis.
- **Medial / Lateral Meniscus**: positive = signal touching an articular surface on ≥2 images (two-slice-touch) OR morphologic deformity (truncation, ghost root, displaced fragment, extrusion with root tear). Negative: grade 1–2 intrasubstance signal without surface contact; ligament/tendon/magic-angle mimics; postoperative intermediate signal alone.
- **Medial / Lateral / PF OA**: positive = ≥1 cm of high-grade (>50% thickness) cartilage loss per compartment. Negative: WORMS-1 signal-only change, equivocal or <3 mm isolated osteophytes, isolated BML/Hoffa signal.
- **Effusion**: positive = moderate or large (MOAKS 2–3; ≈ suprapatellar AP ≥5 mm). Negative: trace/physiologic fluid.
- **Synovitis**: positive per host criteria (moderate/large grading family); on non-contrast data treat frond-like/nodular intermediate-signal thickening as positive; uniform fluid signal alone = effusion only. Hoffa signal alone is sensitive but non-specific — label cautiously.
- **Baker's Cyst**: positive = moderate or large posteromedial cyst with diagnostic neck. Negative: small physiologic bursal fluid; neck-less collections.
- **Contusion**: positive = geographic marrow edema-like signal without line or contour deformity. Negative: anything meeting Fracture criteria.
- **Fracture**: positive = discrete low-signal line (best on T1 if present), cortical breach, or subchondral impaction, confirmed on ≥2 slices AND 2 planes. Negative: line on a single slice/plane only (→ Contusion).
- **Global**: borderline findings are graded NEGATIVE; no "abnormal" umbrella label; no PCL/LCL labels. Only 58/4,407 train studies have per-condition labels — the rest must be mined from multilingual reports.

### 130 mm crop risk summary

A 130 mm central crop is smaller than the typical 140–160 mm FOV and trims ~5–15 mm per side; centered on the joint line it reaches ~65 mm each way.

- **HIGH RISK**
  - **MCL** — tibial insertion centroid ~62 mm below the joint line (range to ~71.5 mm): a joint-line-centered 130 mm crop barely includes it and clips the distal footprint/edema in large or off-center knees; in-plane medial soft tissues sit at the skin margin. Require: full medial epicondyle, medial joint line, ≥75 mm below medial plateau.
  - **Effusion** — suprapatellar pouch extends above the patella: tight sagittal crops clip the superior pouch and under-grade effusion.
  - **Baker's cyst** — neck at the joint line is safe, but tails of large/ruptured cysts run off the inferior axial stack and posterior sagittal FOV edge.
  - **Contusion** — posterolateral tibial plateau bruises (pivot-shift pattern) lie near the lateral/posterior crop edge.
- **LOW RISK**
  - **ACL** (notch is central), **menisci** (body/horns/roots well inside), **tibiofemoral OA cartilage**, **Baker's cyst neck**, **PF OA on axial** (patella/trochlea within FOV), **fracture lines** (osseous, central).
- **Practical recommendation**: verify the crop fits per-series against DICOM PixelSpacing before cropping — the forum's "silent no-op" warning (an over-large requested crop can silently do nothing; discussion/734105). Consider 160 mm or FOV-adaptive cropping; never assume a central 130 mm crop is universally safe, especially for MCL and sagittal fluid collections.

## Part 3 — References & Evidence Notes

1. Bien N, et al. Deep-learning-assisted diagnosis for knee MRI (MRNet). PLoS Med 2018;15(11):e1002699 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/
2. De Smet AA, Tuite MJ. Use of the "two-slice-touch" rule for the MRI diagnosis of meniscal tears. AJR 2006 — https://pubmed.ncbi.nlm.nih.gov/16675771/
3. De Smet AA. How I Diagnose Meniscal Tears on Knee MRI. AJR 2012;199:481–499 (cited as "Helms, AJR" in Dim 01 source) — https://ajronline.org/doi/10.2214/AJR.12.8663
4. Bolog N, Andreisek G. Reporting knee meniscal tears: pitfalls. 2016 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/
5. Ng AWH, et al. Imaging of the anterior cruciate ligament — https://pmc.ncbi.nlm.nih.gov/articles/PMC3302044/
6. Dove L, et al. 2024 ACL injury MRI review — https://pmc.ncbi.nlm.nih.gov/articles/PMC11463185/
7. Celikyay F, et al. Mucoid degeneration of the ACL vs tear — https://pmc.ncbi.nlm.nih.gov/articles/PMC7333554/
8. Liu F, et al. Morphology of the superficial MCL (origin/insertion distances) — https://pmc.ncbi.nlm.nih.gov/articles/PMC2954927/
9. Vosoughi F, et al. MCL injury MRI grading — https://pmc.ncbi.nlm.nih.gov/articles/PMC8221433/
10. Baker JC, et al. 2018. Postoperative meniscus MRI performance — https://pubmed.ncbi.nlm.nih.gov/29949412/
11. Hunter DJ, et al. 2011. MOAKS scoring system — https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/
12. MOST MRI-based knee OA definitions — https://pmc.ncbi.nlm.nih.gov/articles/PMC10922537/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC10361157/
13. Zhang Y, et al. Osteophyte grading/atlas — https://pmc.ncbi.nlm.nih.gov/articles/PMC6235223/
14. Roemer FW, et al. MRI assessment of synovitis and joint fluid / Hoffa-synovitis reliability (non-contrast specificity 10–38%) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6504589/
15. OMERACT synovitis definition (post-contrast reference standard) — within https://pmc.ncbi.nlm.nih.gov/articles/PMC6504589/
16. MOAKS effusion-synovitis scoring — https://pmc.ncbi.nlm.nih.gov/articles/PMC6310094/
17. ACLOAS effusion grading (suprapatellar AP diameter) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7677197/
18. Perdikakis E, Skiadas V. MRI of cysts and cyst-like lesions around the knee — https://pmc.ncbi.nlm.nih.gov/articles/PMC3675245/
19. Jarraya M, et al. Occult fracture pictorial review — https://pmc.ncbi.nlm.nih.gov/articles/PMC3613077/
20. Bone marrow signal alteration of the knee (AJR) — https://www.ajronline.org/doi/10.2214/AJR.10.4961
21. Ochi M, et al. 2022. SIFK review — https://pmc.ncbi.nlm.nih.gov/articles/PMC9068663/ ; Malghem J, et al. 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10545656/
22. Boks SS, et al. 2007. MRI follow-up of posttraumatic bone bruises (median healing 42 weeks). AJR 189:556–562 — https://pubmed.ncbi.nlm.nih.gov/17715100/
23. Bone bruise review — https://pmc.ncbi.nlm.nih.gov/articles/PMC4340462/ ; post-traumatic marrow terminology — https://pmc.ncbi.nlm.nih.gov/articles/PMC7571512/ ; T1 for fracture lines — https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/
24. Kaggle: competition overview — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview ; data page — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
25. Kaggle discussions: 733343 (host annotation criteria) — https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343 ; 733652 (external data) — .../discussion/733652 ; 733965 (host clarifications, LLM rules) — .../discussion/733965 ; 734105 (130 mm physical crop, "silent no-op") — .../discussion/734105

*Caveat: all community-sourced items (crop sizes, slice sampling, plane weighting, backbone choices, forum label interpretations) are early-competition practice, not validated winning solutions.*
