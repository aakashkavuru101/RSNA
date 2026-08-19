# Research Dim 1 — Ligaments & Meniscus MRI Criteria (for ML labeling)

**Bottom line:** sagittal fluid-sensitive images = primary ACL series; coronal FS PD/T2 = primary MCL series; sagittal PD plus coronal/axial confirmation for meniscal tears. For high-specificity meniscus labels, intrasubstance signal NOT reaching an articular surface is negative; a confident native tear needs surface contact on two images, or unequivocal abnormal morphology/displacement.

## 1. ACL tear

### Best plane and sequence
- **Primary plane: sagittal** (or sagittal-oblique parallel to the ACL). Coronal and axial should always be reviewed because the ACL is oblique and prone to partial-volume artifact. (Ng et al., Imaging of the ACL, https://pmc.ncbi.nlm.nih.gov/articles/PMC3302044/)
- **Preferred sequence: fluid-sensitive intermediate/PD or T2, fat-suppressed.** Acute edema, hemorrhage, fiber gap, marrow contusions most conspicuous on FS PD/FS T2. Non-FS PD/T1 useful for fiber morphology and chronic change. Standard diagnostic ACL imaging: 3–4 mm slices, 14–16 cm FOV. (musculoskeletalkey.com ACL MRI review)
- Coronal useful for: empty intercondylar notch/empty lateral wall, femoral avulsion, distinguishing ACL from PCL and partial-volume effects.
- Axial useful for: femoral origin, bundle-level partial tears, confirming a suspected gap seen on only one sagittal slice.

### Primary signs (positive label rests on direct signs)
- Fiber discontinuity or fluid-filled gap
- Diffuse ligament enlargement with cloud-like FS PD/T2 hyperintensity (edema/hemorrhage)
- Wavy, lax, bowed, or abnormal contour
- Abnormal slope: normal ACL is parallel to or steeper than Blumensaat's line; torn distal stump lies too horizontal, proximal stump too vertical
- Nonvisualization or empty notch (chronic complete tears)
- Avulsion at femoral or tibial attachment
- Partial tear: increased signal/laxity with some fibers remaining continuous; high-grade partial >50% fibers; bundle grading on routine MRI is difficult. (Dove et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC11463185/; Ng et al.)

### Secondary signs (supportive only, never stand-alone)
- Anterior tibial translation: ~≥5 mm on lateral-compartment sagittal ≈ 86% sensitivity / 99% specificity; >7 mm essentially diagnostic (position/method dependent). (Ng et al.)
- Pivot-shift bone-contusion pattern: mid-lateral femoral condyle near condylopatellar sulcus + posterolateral tibial plateau (may include medial posterior tibia/MFC from contrecoup).
- Deep lateral femoral notch (depth ~1.5–2 mm+) — highly specific, insensitive.
- Uncovered posterior horn of lateral meniscus; PCL buckling; patellar tendon buckling.
- Segond fracture (lateral proximal tibial cortical avulsion) — uncommon but strongly associated with ACL rupture.

### Is one abnormal slice enough?
- **No formal ACL two-slice rule.** Normal ACL seen on ~2–3 contiguous sagittal or coronal slices at 3–4 mm thickness; dedicated oblique-axial imaging can show it on 11–15 contiguous images. (Ng et al.)
- Practical: a single unequivocal complete discontinuity/nonvisualization can be enough. Signal alone on one slice is NOT enough (partial volume, normal striation, mucoid change, short-TE effects). For a high-quality label, confirm in ≥2 orthogonal planes or use morphology + secondary signs. Complete discontinuity in at least two image planes is an important criterion. (Dove et al.)

### Mimics / false positives
- **Mucoid degeneration**: fusiform thickened ACL, diffuse high signal, preserved parallel fibers, "celery-stalk" appearance; often with ganglion/intraosseous cysts; no discontinuity, no pivot-shift pattern. In one series: discontinuity in 97% of complete tears vs 0% of mucoid degeneration; celery-stalk in 66% of mucoid cases vs 0% of tears. (Celikyay et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC7333554/)
- **ACL ganglion**: lobulated sharply marginated fluid-equivalent lesion; may splay but not disrupt fibers.
- **Partial tear**: overlaps with complete tear, mucoid change, normal striation; MRI sensitivity/specificity much lower than for complete tears.
- **Partial-volume averaging** at femoral attachment / between bundles / notch fluid — scroll adjacent slices, confirm orthogonally.
- **Magic-angle/short-TE signal**: collagen near ~55° to B0 appears brighter on short-TE PD/T1; for ACL, partial volume and distal striation are the dominant pitfalls.

### Chronicity / surgery
- Chronic tears: nonvisualization/resorption, empty notch, attenuation, abnormal residual slope; distal stump may scar to PCL and falsely appear continuous; acute edema may be absent; chronic anterior tibial translation can persist. (Ng et al.)
- ACL graft: early grafts may be low signal; intermediate signal common during revascularization at ~4–8 months; ligamentization produces native-like low signal by ~12 months (up to 24). Graft failure: discontinuity/fluid gap, abnormal orientation, recurrent anterior translation, tunnel malposition/widening, impingement, graft ganglion. Do NOT call early homogeneous graft hyperintensity alone a retear. (https://pmc.ncbi.nlm.nih.gov/articles/PMC6395843/)

### Suggested label policy
- Positive: unequivocal discontinuity/nonvisualization, abnormal stump morphology, fluid-sensitive signal + lost morphology (include high-grade partial tears per competition definition).
- Equivocal: one-slice signal only, mucoid degeneration with intact fibers, possible partial volume.
- Negative: intact continuous fibers, normal orientation, isolated secondary signs, isolated mucoid degeneration.

## 2. MCL tear

### Best plane and sequence
- **Coronal FS PD/T2 is the highest-value sequence — effectively essential**: shows the ligament along its long axis, periligamentous edema, superficial/deep layer involvement, discontinuity. Sagittal is poor for MCL (thin in the through-plane direction).
- **Axial FS PD/T2 complementary**: cross-sectional fiber continuity, injury level (femoral vs mid-substance vs tibial), deep MCL, posterior oblique ligament, posteromedial corner. (https://pmc.ncbi.nlm.nih.gov/articles/PMC3548666/)
- If FS fails, non-FS PD/T2 shows thickening/discontinuity but grade I and small deep-layer lesions are less conspicuous.

### Anatomy and the 130 mm crop question (superficial MCL)
- Origin: medial femoral epicondyle, centroid ~31.1 ± 4.6 mm proximal to femoral joint line.
- Insertion: medial tibia, centroid ~62.4 ± 5.5 mm below tibial joint line (range ~54.7–71.5 mm).
- Mean centroid-to-centroid length 100.7 ± 9.5 mm; AP width ~10.9 mm proximally, 17.7 mm centrally, 10.7 mm distally. (Liu et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC2954927/)
- The common "7–10 cm below joint line" is too distal for most knees; tibial centroid usually ~6 cm.
- **A 130 mm crop centered on the joint line** extends ~65 mm each way: barely includes the tibial insertion centroid (~62 mm) but may clip the distal attachment footprint, periligamentous edema, larger patients, off-center acquisitions.
- A 130 mm square in-plane crop may exclude the ligament if it trims medial soft tissues or is centered on the notch.
- **ML recommendation**: verify crop masks against DICOM pixel spacing; preserve full medial femoral epicondyle, medial joint line, and ≥75 mm below the medial tibial plateau. Do not assume a central 130 mm crop is always safe.

### Grading (MRI grade may not equal clinical laxity)
- Grade I sprain: periligamentous edema/hemorrhage superficial to ligament; ligament intact.
- Grade II partial tear: intraligamentous high signal, thickening, partial fiber disruption or deep-fiber disruption; some fibers continuous.
- Grade III complete tear: full-thickness discontinuity, avulsion, wavy/retracted fibers, fluid extravasation. (Vosoughi et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC8221433/; Radiopaedia MCL grading)
- Recommendation: grade I = sprain/low-grade injury, not "tear"; binary "MCL tear" ≈ grade II–III. Competition host criteria: MCL = high-grade acute tear.

### Slices needed
- No validated MCL two-slice rule. Superficial MCL ~11–18 mm wide AP → occupies ~3–6 consecutive 3–4 mm coronal slices (geometric estimate). A focal femoral avulsion may be most abnormal on only 1–2 coronal slices. Axial crosses the ligament repeatedly over its ~100 mm course; use to confirm injury level. Inspect the entire femoral-epicondyle-to-distal-tibia course.

### Mimics / associated lesions
- MCL bursitis: fluid between superficial and deep layers near joint line, without fiber disruption.
- Pes anserine bursitis/tendinopathy: more anteromedial/distal.
- Reactive superficial edema from medial meniscal pathology, OA, effusion, or MPFL injury without true fiber damage.
- Meniscocapsular separation / deep MCL injury (meniscofemoral + meniscotibial components) with intact superficial MCL: fluid/separation between medial meniscus and capsule, meniscal displacement. (AJR "Five Overlooked Injuries on Knee MRI")
- Stener-like MCL lesion: distal complete tear displaced superficial to pes anserine tendons.

### Chronicity
- Healed/chronic: thickened low-signal but continuous ligament without edema.
- Pellegrini-Stieda: calcification/ossification near proximal femoral MCL attachment = prior injury.
- Chronic lax MCL may look near-normal on static MRI.

## 3. Meniscal tears (medial & lateral)

### Planes by anatomic region
| Region | Highest-yield plane/location | Key signs |
|---|---|---|
| Body | Sagittal peripheral compartment; coronal mid-body | Sagittal "bow-tie"; coronal triangular body, truncation, extrusion, displaced fragment |
| Anterior horn | Central-anterior sagittal; anterior coronal; axial | Surface signal, cleft, small/truncated horn |
| Posterior horn | Central-posterior sagittal; posterior coronal; axial | Vertical/horizontal signal, peripheral/ramp lesions, root abnormality |
| Roots | Posterior/central coronal, central sagittal, thin axial | Ghost meniscus, truncation, radial cleft, absent attachment, extrusion |

- Sagittal: peripheral body slices = continuous bow-tie; central slices = separate triangular anterior/posterior horns. With 4–5 mm slices the normal 9–12 mm body gives ~2 contiguous bow-tie images (more with 3 mm slices). (radiologykey.com)
- Coronal: mid = body/extrusion; anterior = anterior horn/root; posterior = posterior horn + posterior roots.
- Axial: best for mapping radial, root, parrot-beak, displaced flap tears.

### Preferred sequence / slice thickness
- Best conventional meniscal evaluation: high-resolution short-TE PD (often non-FS sagittal+coronal); FS PD/T2 improves conspicuity of fluid, marrow edema, meniscocapsular injury, postoperative clefts. Representative protocol: sagittal non-FS PD, coronal non-FS PD, FS axial PD/T2, ~2.5–3 mm slices. (Bolog & Andreisek, https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/)
- Competition setup: sagittal T1/PD = morphology + surface signal; coronal PD FS = body, extrusion, roots, radial tears, fragments; axial PD FS = roots and radial/parrot-beak mapping. Do not discard sagittal T1: morphology and surfacing signal remain useful.

### Native tear criteria and two-slice rule
Two primary criteria: (1) abnormal intrameniscal signal reaching a superior or inferior articular surface; (2) abnormal morphology (truncation, surface defect, abnormally small segment, displaced fragment, bucket-handle). (Helms, AJR "How I Diagnose Meniscal Tears")
- **Two-slice-touch rule**: surface-contacting signal in the same meniscal region on two images (classically two consecutive sagittal; practically one sagittal + one coronal also acceptable). Tear probability: one image ~18–55%; ≥2 images ~90–96% (PPV ~53% one sagittal → ~90% two consecutive sagittal). (De Smet & Tuite, https://pubmed.ncbi.nlm.nih.gov/16675771/; Bolog & Andreisek)
- Exceptions: abnormal morphology/displaced fragment can override; root tear may be conspicuous on only 1–2 key slices per plane; tiny radial tear may be clear on one slice but lower confidence without orthogonal confirmation.

### Degeneration vs tear
- Grade 1: globular intrameniscal signal, no surface contact. Grade 2: linear intrameniscal signal, no definite surface contact. Grade 3: signal reaches a surface.
- **Grades 1–2 = negative for "tear"** in an ML label (intrasubstance/myxoid degeneration; normal vascular signal in children). If uncertain whether signal reaches a surface, high-specificity practice = no definite tear. (https://pmc.ncbi.nlm.nih.gov/articles/PMC7878874/)

### Root tears (functionally severe; disrupt hoop fibers; can behave like subtotal meniscectomy)
- Ghost meniscus on sagittal (expected posterior horn/root fibrocartilage absent at attachment).
- Truncation sign on coronal (abrupt vertical fluid-signal gap at root attachment).
- Radial cleft/defect on axial (thin slices).
- Meniscal extrusion: medial body extrusion >~3 mm beyond tibial margin abnormal, strongly associated with posterior medial root tears; lateral threshold ~1 mm. (Bolog & Andreisek)
- Root locations: anterior medial root ~7 mm anterior to ACL; posterior medial root ~8 mm anterior to superior PCL insertion; anterior lateral root ~4.1 mm lateral to posterolateral ACL bundle; posterior lateral root ~10.8 mm posterior to ACL / 12.7 mm anterior to PCL.

### Common mimics
- Transverse intermeniscal ligament (anterior horns) — trace across Hoffa's fat pad.
- Popliteus tendon (posterolateral) — mimics posterior lateral horn tear or loose body.
- Meniscofemoral ligaments (Humphrey/Wrisberg) — from posterior horn of lateral meniscus to MFC; mimic tear/displaced fragment.
- Magic-angle signal in curved posterior horn fibers on short-TE sequences.
- Pediatric vascularity — prominent signal in young patients ≠ tear.
- Meniscal flounce, chondrocalcinosis, meniscal ossicles alter contour/signal.

### Discoid lateral meniscus
- Persistent bow-tie on ≥3 consecutive 5 mm sagittal slices; coronal body width >14–15 mm; meniscal/tibial width ratio >20%. (https://pmc.ncbi.nlm.nih.gov/articles/PMC5134787/)
- Discoid morphology alone ≠ tear; diffuse internal signal has poor specificity in discoid menisci; linear surface-reaching signal/deformation/displacement is concerning. Wrisberg type may lack normal posterior tibial attachment.

### Postoperative meniscus
- <25% meniscectomy: native criteria remain reasonably accurate.
- ≥25% meniscectomy or repair: surface-contacting intermediate signal on PD/T1 no longer specific (granulation tissue, exposed grade-2 signal, normal healing). Reliable recurrent-tear signs: fluid-equivalent T2 signal entering meniscus, gadolinium on direct MR arthrography, displaced fragment, new-location tear, fluid-filled cleft. Repaired menisci may retain high signal for years despite healing. (Baker et al., https://pubmed.ncbi.nlm.nih.gov/29949412/)
- MR arthrography outperforms conventional MRI after repair/≥25% resection (~85–93% vs ~57–80%).
- Recommendation: separate postoperative class if metadata available; do not use native grade-3 signal alone in a clearly postoperative meniscus.

## 4. Series length / slice extent
- Stanford MRNet: 17–61 images per series, mean 31.48 ± 7.97 → practically ~20–40, commonly 24–40, variable. (Bien et al., https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- Original MRNet model selected sagittal T2 FS, coronal T1, axial PD FS (full protocol had coronal T1/T2 FS, sagittal PD/T2 FS, axial PD FS).

| Pathology | Typical slice extent (3–4 mm) | Rule |
|---|---|---|
| ACL | Normal ligament ~2–3 sagittal/coronal slices; complete tear may affect all ACL slices or be obvious on one | No two-slice rule; confirm in another plane |
| MCL | ~3–6 coronal slices (band ~11–18 mm AP); focal avulsions fewer | No fixed rule; inspect full course |
| Small meniscal signal tear | Preferably two images same region (one sag + one cor OK) | Two-slice-touch rule |
| Radial/root tear | Often 1–2 key slices per plane + orthogonal signs/extrusion | Morphology overrides |
| Longitudinal/bucket-handle | Many slices; fragment may lie far from parent | Morphology > slice count |
| Discoid meniscus | ≥3 consecutive 5 mm sagittal bow-ties | Validated criterion |

### Practical ML implications
1. Use inter-slice context (3D / 2.5D / attention over series).
2. Per-plane inference then fusion: sagittal → ACL; coronal → MCL; meniscus → sagittal + coronal/axial confirmation.
3. Preserve physical metadata (slice thickness, spacing, orientation, FOV).
4. Avoid tight crops: ACL — full notch + lateral femoral condyle/posterolateral tibia; MCL — medial soft tissue + ≥75 mm below medial joint line; meniscus — peripheral body, posterior horns, tibial spine/roots, notch (displaced fragments).
5. High-specificity labels: ACL = direct discontinuity/nonvisualization/abnormal stump; MCL = grade II–III; meniscus = grade-3 surface signal on two images or root/truncation/ghost morphology or displaced fragment; grade 1–2 negative.
6. Treat postoperative knees separately.
