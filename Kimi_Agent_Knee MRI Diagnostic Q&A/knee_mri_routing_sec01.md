# Section 1: Competition Context

The RSNA Knee Abnormality Detection challenge is **live**: launched July 30, 2026, entry/merger deadline October 15, 2026, final submission October 22, 2026, winners announced at RSNA 2026 in November. This is a code competition (no internet, ≤9 h runtime) scored by **macro-averaged ROC AUC across 12 binary targets**; external data and pretrained models are allowed (competition overview).

**Dataset.** A brand-new RSNA AI Challenge collection — explicitly **not MRNet-derived** — of >5,000 knee MRI exams from 16–19 sites across five continents, paired with original radiology reports in ~9–12 languages. Intensities, orientations, and resolutions vary across series and studies; slices per series typically 20–45 (median 30), with a long tail. DICOMs use mixed transfer syntaxes and an 86-tag allowlist. **Critical label facts: only 58 of 4,407 train studies carry per-condition labels**; the other 4,349 have report text only, so the core challenge is mining labels from multilingual free text. The ~1,300 test studies have **no reports at all**. Series metadata gives only plane (Sagittal/Coronal/Axial) plus Fluid_Sensitive and Fat_Suppression flags — **no named sequences** (no "sagittal T1" or "coronal PD FS"). Community EDA counts ~9,864 sagittal / 8,609 coronal / 5,898 axial series (official data page; dim03).

**Official 12-label taxonomy** (matches this document's finding list exactly): ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. No umbrella "abnormal" label; no PCL/LCL targets.

**Host-posted annotation criteria (discussion/733343)**, double-read by MSK radiologists with third-reader adjudication:
- ACL = high-grade partial (>50% fibers) or full tear
- MCL = high-grade acute tear
- Meniscus = surface-touching signal on ≥2 images OR morphologic deformity
- OA = ≥1 cm of high-grade (>50% thickness) cartilage loss per compartment
- Effusion and Baker's cyst = moderate or large
- **Borderline findings are graded negative** — bias your decision thresholds toward specificity.

**Community practices (early-competition, NOT validated winning solutions — none exist yet).** Physical-scale central crops of 130–160 mm resized to a fixed pixel grid (e.g., 130 mm → 336 px ≈ 0.387 mm/px), because naive resizing of larger FOVs can erase 1–3 mm pathology; the forum warns to **verify the crop actually fits the array or the operation silently becomes a no-op** (discussion 734105; LB 0.768–0.894 baselines). Slice sorting must use **ImagePositionPatient · row/column vectors from ImageOrientationPatient, never filename**. No Laterality column exists — derive laterality from image-center geometry (flip coronal/axial horizontally, reverse sagittal stack order); this matters because 5 of 12 targets are medial/lateral pairs. Common architectures: per-plane encoders with cross-plane fusion into 12 sigmoid heads; DINOv2 small/base is the dominant backbone; MRNet is raised as external data. A metadata-only probe reaches ~0.65 macro AUC random-fold but 0.598 scanner-grouped — no metadata shortcut.

**Transferable evidence — MRNet / Bien et al. 2018 (PLoS Medicine).** 1,370 Stanford exams; protocol coronal T1, coronal T2 FS, sagittal PD, sagittal T2 FS, axial PD FS; the model used one series per plane (sagittal T2, coronal T1, axial PD). Slices per series 17–61, mean 31.5 ± 8.0. Per-slice AlexNet → max-pool over slices → logistic-regression fusion. Plane-importance findings: **axial PD most beneficial for abnormality and meniscal tear; coronal T1 for ACL tear** (AUCs 0.937 abnormal / 0.965 ACL / 0.847 meniscal). Takeaway: no single plane dominates; fuse planes, and expect ~20–45 usable slices per series.

# Section 2: Ten Highest-Priority Questions

## 1. MCL — Is coronal fat-suppressed imaging essential, and does a central crop risk cutting the ligament?

**Yes, coronal FS PD/T2 is effectively essential, and yes, a 130 mm central crop can clip the distal ligament.**

- Coronal FS PD/T2 shows the MCL along its long axis — periligamentous edema, superficial/deep layer involvement, discontinuity — and is the highest-value sequence; sagittal is poor for a structure this thin in-plane. Axial FS PD/T2 is complementary for injury level and the posteromedial corner. If fat suppression fails, grade I and small deep-layer lesions lose conspicuity (PMC3548666).
- Anatomy vs crop: the superficial MCL originates ~31.1 ± 4.6 mm above the femoral joint line and inserts ~62.4 ± 5.5 mm below the tibial joint line (centroid range ~54.7–71.5 mm; mean length ~100.7 mm; Liu et al.). A 130 mm crop centered on the joint line reaches only ~65 mm each way — it barely covers the tibial insertion centroid and can clip the distal attachment footprint, periligamentous edema, and larger or off-center knees. An in-plane 130 mm crop can also trim the subcutaneous medial soft tissues where the MCL lies.
- **ML rule: preserve the full medial femoral epicondyle, the medial joint line, and ≥75 mm below the medial tibial plateau; verify the crop mask against DICOM PixelSpacing rather than assuming a central 130 mm crop is safe.** The MCL target is a high-grade acute tear (grade II–III); grade I sprains stay negative (host criteria, discussion/733343).

## 2. Synovitis — Can it be reliably distinguished from uncomplicated effusion on non-contrast MRI?

**No — not reliably. Treat Synovitis as a low-confidence, high-uncertainty label.**

- On non-contrast MRI, synovial thickening and simple effusion are both bright on PD FS/T2 FS; the OA literature's composite term "effusion-synovitis" exists precisely because they merge (PMC6504589). Contrast-enhanced MRI is the reference standard: inflamed synovium enhances, static fluid does not.
- Quantitative evidence: Hoffa fat-pad signal change on non-contrast MRI is sensitive (86–97%) but **very poorly specific (10–38%)** against CE-MRI (Roemer et al.); Loeuille et al. found only CE-MRI correlated with microscopically proven synovitis — no correlation for non-contrast MRI.
- Practical heuristic when no contrast exists: frond-like, villous, nodular, or irregular **intermediate** signal (lower than fluid, ~cartilage signal) lining the joint, or mass-like suprapatellar wall thickening, suggests synovitis; smooth thin-walled uniform fluid does not. Do not over-trigger: the host counts synovitis as its own target, so calibrate with the 58 labeled studies and expect noisy positives.

## 3. Lateral meniscus — Which sagittal/coronal regions identify body, horn, and root tears, and how many contiguous abnormal slices are expected?

**Map regions to planes, and require surface-touching signal on ≥2 images (or definite morphology) — matching the host rule.**

- Region-by-plane mapping: **body** — peripheral sagittal slices (continuous "bow-tie," ~2 contiguous bow-ties at 4–5 mm thickness) and mid-coronal (triangular body, truncation, extrusion); **anterior horn** — central-anterior sagittal, anterior coronal, axial; **posterior horn** — central-posterior sagittal, posterior coronal, axial; **roots** — posterior/central coronal (truncation sign), central sagittal (ghost meniscus), thin axial (radial cleft). Axial is best for mapping radial, root, and parrot-beak tears (dim01).
- **Two-slice-touch rule (De Smet & Tuite 2006):** surface-contacting signal in the same region on two images (classically two consecutive sagittal slices; one sagittal + one coronal acceptable) gives ~90% PPV versus ~53% on a single sagittal image. The host codifies this: signal touching a surface on ≥2 images OR morphologic deformity.
- Exceptions: root tears may be conspicuous on only 1–2 key slices per plane — morphology (ghost, truncation, radial cleft, extrusion; lateral extrusion threshold ~1 mm) overrides slice count. Lateral-specific mimics: popliteus tendon, meniscofemoral ligaments (Humphrey/Wrisberg), transverse intermeniscal ligament, magic angle, discoid meniscus (≥3 consecutive 5 mm bow-ties; diffuse signal alone ≠ tear).

## 4. Lateral OA — Which cartilage surfaces and osteophytes define a positive case, and which planes show them best?

**Positive = ≥1 cm of >50%-thickness cartilage loss in the lateral tibiofemoral compartment (host rule); coronal PD FS is the workhorse plane; borderline findings stay negative.**

- Surfaces: lateral femoral condyle and lateral tibial plateau weight-bearing cartilage; marginal osteophytes of the lateral condyle/plateau and tibial spines. Coronal PD FS covers the tibiofemoral weight-bearing cartilage and marginal osteophytes; sagittal adds posterior condyle/plateau margins and central weight-bearing cartilage; axial is for patellofemoral disease, not this target (dim02; OAI osteophyte study).
- Thresholding: the host criterion (≥1 cm of high-grade >50% cartilage loss per compartment) is stricter than the MOST MRI definition (focal partial-thickness cartilage defect WORMS ≥2 plus a definite osteophyte). Under the host rule, a definite osteophyte (WORMS ≥2, ~≥3 mm spur) supports but does not alone satisfy positivity. Note osteophytes are **less conspicuous on FS sequences** (marrow fat suppressed) — read them as cortical/marginal spurs.
- Stay negative: WORMS cartilage grade 1 (signal-only, normal thickness), equivocal osteophytes (WORMS 1), isolated <3 mm marginal spurs, small physiologic fluid. Joint-space narrowing tracks cartilage loss (KL 2 with JSN: 44% WORMS ≥4 cartilage vs 4% without) — use it as a supporting cue, not a primary one.

## 5. Baker's cyst — Does the entire posteromedial region need to remain in the crop?

**The diagnostic neck always lies within a 140–160 mm FOV; only the tails of large/ruptured cysts risk clipping — mainly at the sagittal posterior edge or axial inferior end.**

- The cyst's neck sits between the semimembranosus tendon and medial gastrocnemius head, immediately posteromedial to the medial femoral condyle at the joint line — inside even a tight central crop of axial/coronal FOV (ACR caps knee FOV at 16 cm; typical 140–160 mm).
- Risk zones: on **sagittal** series the cyst sits near the posterior FOV edge, so a 130 mm crop can clip it; the tail of a large or ruptured cyst dissecting into the calf can run off the inferior end of the axial stack. The diagnostic neck remains in-plane regardless.
- **ML rule:** diagnose on axial PD FS via the "speech-bubble" neck configuration (required to exclude mimics — especially popliteal artery aneurysm, the critical don't-miss mimic); use sagittal for craniocaudal extent and rupture. The host counts only moderate-or-large cysts, so small physiologic bursal fluid stays negative — the cases that matter are exactly the ones large enough to be visible even if the tail is clipped.

## 6. ACL — Is continuity across several adjacent sagittal slices necessary, rather than one strongly abnormal slice?

**No formal two-slice rule exists. One unequivocal discontinuity or nonvisualization can be positive; signal abnormality on a single slice is never enough.**

- The normal ACL spans only ~2–3 contiguous sagittal (or coronal) slices at 3–4 mm thickness, so "several abnormal slices" is not a requirement (Ng et al.).
- Positive-level evidence: fiber discontinuity or fluid-filled gap, diffuse enlargement with cloud-like FS hyperintensity plus lost morphology, wavy/abnormal contour, abnormal slope vs Blumensaat's line, nonvisualization/empty notch, or avulsion. Complete discontinuity confirmed in **at least two image planes** is the strongest criterion (Dove et al.).
- Single-slice **signal** alone is insufficient — partial-volume averaging at the femoral attachment, normal distal striation, mucoid degeneration ("celery-stalk," preserved fibers, no discontinuity), and short-TE effects all mimic it. When sagittal is equivocal, confirm on coronal/axial or add morphology plus secondary signs (anterior tibial translation ~≥5 mm, pivot-shift contusion pattern, deep lateral femoral notch — supportive only, never stand-alone).
- Host rule: positive = high-grade partial (>50% fibers) or full tear; bundle-level partial grading is unreliable on routine MRI, so low-grade partials and equivocal cases go negative.

## 7. Contusion vs fracture — Which sequences and signs separate marrow edema from an occult fracture?

**The T1 non-FS linear low-signal line is the key discriminator — but it may not exist in this dataset; fall back to cortical breach, subchondral impaction, or a line visible in ≥2 slices and 2 planes; geographic non-linear edema is contusion.**

- T1 non-FS is the key fracture-line sequence ("T1W images should be obtained to detect fracture lines"; PD is less reliable for the line). Contusion: geographic, non-linear, ill-defined low/intermediate T1 signal that does not fully replace marrow fat (Mink & Deutsch). Fracture: linear/curvilinear low-signal line on both T1 and T2, ± cortical breach, step-off, or subchondral impaction/depression.
- Dataset caveat: only plane + fluid-sensitivity/fat-suppression flags are provided and T1 is not guaranteed; with PD FS planes only, infer fracture from a low-signal line within PD-FS edema confirmed in **≥2 contiguous slices AND a second orthogonal plane**, cortical step-off, or subchondral impaction — otherwise label Contusion (dim02). A single-slice, single-plane line is likely volume averaging or artifact.
- Edema-pattern priors: the classic ACL-injury pattern (lateral femoral condyle sulcus + posterolateral tibial plateau) is contusion unless a line or depression exists. Watch for subchondral insufficiency fracture (SIFK): thin subchondral T2-hypointense line parallel to the plate + extensive edema, typically weight-bearing MFC in older adults, often with medial posterior root tears — differentiate from osteonecrosis (serpiginous curvilinear boundary engaging the plate, double-line sign). Subchondral low-signal >4 mm thick or >14 mm long carries poor prognosis (Ochi 2022; Malghem 2023).

## 8. Effusion vs synovitis — Can both be independently present, and what evidence differentiates them?

**Yes, they frequently coexist — and on non-contrast MRI they merge into one bright complex; morphology of the fluid's walls is the only practical differentiator.**

- Separable on CE-MRI (enhancing synovium vs non-enhancing fluid); merged on the non-contrast PD FS/T2 FS available here — hence "effusion-synovitis."
- Uncomplicated effusion: large fluid volume with **smooth, thin, uniform walls**. Coexisting synovitis: **irregular, frond-like, or nodular intermediate-signal** (fluid-intermediate, ~cartilage signal) thickening of the recess walls, or Hoffa fat-pad hyperintensity (sensitive 86–97%, specific only 10–38%).
- Effusion grading for the host's moderate-or-large threshold: sagittal suprapatellar AP diameter ≥5 mm (ACLOAS grade 2–3: 5–10 mm / ≥10 mm) or MOAKS axial grade 2–3 (suprapatellar bursa convexity → capsular distension). Crop note: the suprapatellar pouch extends above the patella — a tight 130 mm joint-line-centered crop on sagittal can clip the pouch and underestimate effusion.
- Mimics to exclude before calling synovitis: plica (thin <3 mm medial-gutter band; pathologic ≥3 mm contacting MFC), PVNS/TGCT (low-to-intermediate on T1 and T2, GRE blooming), lipoma arborescens (fat-signal fronds that suppress on FS), post-surgical synovial change.

## 9. Meniscal degeneration vs tear — Should intrameniscal signal that does not reach an articular surface count as negative?

**Yes — unequivocally negative. Grade 1/2 intrasubstance signal is degeneration, not tear, and this matches the host rule exactly.**

- Grade 1 = globular intrameniscal signal, no surface contact; grade 2 = linear intrameniscal signal, no definite surface contact; grade 3 = signal reaches a surface. Grades 1–2 represent intrasubstance/myxoid degeneration (and normal vascularity in children) and are negative for "tear" (PMC7878874).
- Host criteria: positive requires signal **touching a surface on ≥2 images OR morphologic deformity**; borderline findings are graded negative. So if you cannot tell whether signal reaches a surface, the high-specificity call — and the host-consistent call — is **no definite tear**.
- Practical: morphology (truncation, surface defect, displaced fragment, root ghost/truncation) can override the signal rule; in a clearly postoperative meniscus, surface-contacting intermediate signal loses specificity — use fluid-equivalent T2 clefts or displaced fragments instead.

## 10. OA grading — What minimum cartilage loss, JSN, osteophyte, or subchondral change constitutes positive OA?

**Competition rule: ≥1 cm of >50%-thickness cartilage loss per compartment. A definite osteophyte supports; equivocal and grade-1 findings stay negative.**

- Reference frame (MOST MRI definition): definite cartilage damage = WORMS ≥2 (at least a partial-thickness focal defect), definite osteophyte = WORMS ≥2 (~≥3 mm spur); "MOST simple" OA = both. The host's ≥1 cm of >50% loss is a higher bar — focal grade-2 fissuring alone stays negative (host criteria, discussion/733343).
- **Osteophytes:** definite marginal or central osteophyte counts as supporting evidence; WORMS 1 "equivocal" spurs and isolated <3 mm spurs (common at patellar poles/tibial spines, in older or athletic knees) are negative. Osteophytes are less conspicuous on FS sequences — read them on the least-suppressed series available.
- **Joint-space narrowing:** not independently scored on MRI; it tracks cartilage loss (KL 2 with JSN → 44% WORMS ≥4 cartilage vs 4% without JSN), so use it as a corroborating cue.
- **Subchondral change** (BMLs, cysts, sclerosis, attrition): supportive, never required and never sufficient alone.
- Compartment routing: medial/lateral OA on coronal PD FS; PF OA on axial PD FS (patellar facets, trochlea); sagittal for trochlea, patellar poles, posterior margins. **Stay negative:** WORMS cartilage grade 1 signal-only change, equivocal osteophytes, small physiologic fluid, isolated Hoffa signal.
