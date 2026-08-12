# 2. Clinical Primer for Non-Doctors

This chapter gives you the minimum clinical literacy needed to work on knee magnetic resonance imaging (MRI) data: what the anatomy is, how the images are made, what the twelve competition labels actually mean, and how much the humans who produced those labels disagree with each other. Every medical term is defined in plain language at first use, with an analogy; afterwards it is used as-is. No prior anatomy or radiology knowledge is assumed.

## 2.1 Knee Anatomy in Fifteen Minutes

**The joint.** The knee is the largest synovial joint in the body — a synovial joint being one whose bone ends live inside a fluid-filled capsule, like engine parts sealed inside an oil bath. Structurally it is a modified hinge between three bones: the **femur** (thigh bone), the **tibia** (shin bone), and the **patella** (kneecap).[^1^] Think of a door hinge that also permits a small amount of twist. The bottom of the femur splits into two rounded knobs called **condyles** (medial = inner side, lateral = outer side) that rest on the nearly flat top of the tibia, the **tibial plateau** — two balls sitting on a table. Because balls-on-a-table is inherently unstable, the knee relies on soft-tissue "ropes," "washers," and "padding." There are really two joints in one: the **tibiofemoral joint** (femur on tibia, with a medial and a lateral compartment) and the **patellofemoral joint** (kneecap gliding in a femoral groove called the **trochlea**).

**Ligaments — the ropes.** Four ligaments stabilize the knee. The **anterior cruciate ligament (ACL)** and **posterior cruciate ligament (PCL)** cross each other in the center of the joint (cruciate = crossing); the ACL stops the tibia sliding forward and controls rotation, while the thicker PCL stops it sliding backward.[^2^] The ACL is the most commonly torn knee ligament — the classic pivoting sports injury. On the sides, the **medial collateral ligament (MCL)** is a broad flat band resisting forces that buckle the knee inward, and the **lateral collateral ligament (LCL)** is a cord resisting outward bowing. Normal ligaments are taut, low-water structures, and — this matters later — they appear dark on every MRI sequence.[^3^]

**Menisci — the washers.** Sitting on the tibial plateau are two C-shaped wedges of **fibrocartilage** (a tough, rubbery cartilage): the medial and lateral **menisci**. Each has an **anterior horn** (front tip), a **body** (middle), and a **posterior horn** (rear tip). Like rubber washers between pipe fittings, they deepen the socket, spread load, and absorb shock: they transmit roughly 50% of the load through the medial compartment and 70% through the lateral, and removing one raises contact stress by 100–300%.[^4^] Only the outer rim has a blood supply (the "red zone," which can heal); the inner "white zone" cannot.

**Cartilage — the non-stick coating.** The bone ends and the back of the patella are covered by a few millimeters of **articular (hyaline) cartilage**, a smooth, water-rich tissue that lets bone glide on bone almost friction-free — the joint's Teflon coating.[^5^] It has no blood supply and no nerves, so it heals poorly, and its loss is the core lesion of osteoarthritis.

**Supporting cast.** Tendons are cables from muscle to bone: the quadriceps tendon runs into the top of the patella and the patellar tendon runs from its bottom to the tibia (together, the **extensor mechanism** — the pulley system that straightens the knee). **Bursae** are small fluid-filled sacs that work like bubble wrap, reducing friction where tendons glide over bone; one of them, behind the knee, is where a Baker cyst forms. The **synovium** is the membrane lining the joint capsule; it secretes lubricating **synovial fluid**. Normally only a trace of fluid exists, so excess fluid is always a sign that something is wrong. Finally, **Hoffa's fat pad** is a cushion of fat behind the patellar tendon, and **bone marrow** — the fatty tissue inside bones — turns out to be one of the most informative tissues on MRI, because injury and overuse change its water content.

## 2.2 How Knee MRI Works: Planes, Sequences, and Signal Logic

**The physics in one paragraph.** MRI exploits hydrogen protons, which are abundant in water and fat. A strong magnet (1.5 or 3 tesla in clinical knee imaging) aligns the protons like compass needles; a radiofrequency pulse tips them out of alignment; as they relax back, they emit radio signals that receiver coils detect; magnetic field gradients encode where each signal came from; a computer reconstructs the image.[^6^] Two relaxation constants generate contrast: **T1** (how fast protons realign with the main field) and **T2** (how fast they dephase relative to each other). By choosing pulse timing parameters — the repetition time (TR) and echo time (TE) — the scanner emphasizes T1 contrast, T2 contrast, or a high-detail intermediate called **proton density (PD)**. No ionizing radiation is involved.

**Planes.** A knee MRI exam is not one image but several **series**, each a stack of roughly 3–4 mm slices acquired in one of three orthogonal planes:[^7^][^8^]

| Plane | What the slice shows | Structures best evaluated |
|---|---|---|
| Sagittal | Side-view slices, left to right | ACL, PCL, meniscal horns, extensor mechanism, marrow |
| Coronal | Front-view slices, front to back | MCL, LCL, meniscal bodies, roots and extrusion, compartment cartilage |
| Axial | Top-down cross-sections | Patellofemoral cartilage, trochlea, popliteal fossa (Baker cyst), bursae |

The plane–pathology mapping to memorize: **sagittal → cruciate ligaments and meniscal horns; coronal → collateral ligaments and meniscal bodies; axial → patellofemoral cartilage and Baker cysts.**[^7^][^8^]

**Sequences and signal logic.** The one mental model you need: **water is the star of pathology detection.** Almost everything that goes wrong in a knee — tears, bruises, inflammation, cysts — involves fluid or edema (swelling: excess water in tissue). Sequences either render anatomy crisply (T1, PD) or make fluid glow (T2 and its fat-suppressed variants). Meanwhile ligaments, tendons, menisci, and cortical bone are normally **dark on every sequence**, so any bright signal inside them is suspicious. Fat is bright on T1 and stays annoyingly bright on fast T2; **fat suppression (FS)** deliberately turns fat dark so that fluid "pops" against a dark background, and **STIR** (short tau inversion recovery) is an alternative suppression method that is very uniform and robust near metal.[^9^][^10^][^11^]

| Sequence | Fat | Fluid/edema | Ligament, meniscus, tendon | Primary role |
|---|---|---|---|---|
| T1-weighted | Bright | Dark | Dark | Anatomy, marrow fat, fracture lines |
| T2-weighted | Bright | **Bright** | Dark | Fluid, cysts, effusion, inflammation |
| PD | Bright | Intermediate | Dark, high detail | Highest anatomic detail: menisci, ligaments, cartilage |
| PD-FS / T2-FS | **Dark** | **Bright** | Dark | The workhorse: edema, tears, cartilage, effusion |
| STIR | Very dark | Very bright | Dark | Uniform suppression; sensitive to edema; robust near metal |

A standard clinical knee protocol combines the three orthogonal planes with fluid-sensitive fat-suppressed sequences plus one T1 series, typically four to seven series per exam; sagittal and coronal PD-FS are the meniscal workhorses.[^4^] For you as a modeler, this means each exam is a small set of 3D volumes that differ in plane and contrast, are spatially ordered within a series, but are **not** co-registered across series — and protocol details vary across institutions, a major domain-shift axis.

Why do radiologists insist on multiple planes and sequences? Three reasons that translate directly into model design. First, **confirmation**: partial-volume averaging (a slice straddling two structures) can fake a lesion on a single image, so a real abnormality should appear in two planes or on two consecutive images. The classic quantification: meniscal signal reaching the articular surface on at least two images has a positive predictive value (PPV) for a true tear of 94% (medial) and 96% (lateral); on only one image it drops to 43% and 18%.[^4^] Second, **geometry**: no single plane contains an oblique structure like the ACL. Third, **contrast complementarity**: a lesion invisible on T1 may glow on T2-FS. Models that pool evidence across planes and sequences are mimicking this cross-confirmation step.

## 2.3 The Twelve Target Abnormalities

The competition asks for twelve binary exam-level labels, scored as the unweighted macro-average of the twelve per-label areas under the receiver operating characteristic curve (AUC).[^12^][^13^] The labels are: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial osteoarthritis (OA), Lateral OA, patellofemoral (PF) OA, Effusion, Synovitis, Baker's cyst, Contusion, and Fracture.[^12^] Gold-standard labels were produced by two subspecialty musculoskeletal radiologists with a third adjudicator, using published **positivity thresholds** — and, critically, **borderline cases were graded negative**, a specificity-favoring convention you should mirror when choosing decision thresholds.[^12^]

| Label | What it is (plain language) | Best plane & sequence | Competition positivity threshold |
|---|---|---|---|
| ACL | Tear of the anterior cruciate ligament; the commonest knee ligament injury[^14^] | Sagittal PD-FS/T2-FS; confirm coronal/axial | >50% fiber disruption[^12^] |
| MCL | Sprain or tear of the inner-side stabilizer band[^15^] | Coronal PD-FS/T2-FS | Not separately published; borderline graded negative[^12^] |
| Medial Meniscus | Cleft in the inner C-shaped cartilage washer[^16^] | Sagittal PD-FS (horns) + coronal (body) | Signal reaching an articular surface on ≥2 images[^12^] |
| Lateral Meniscus | Same, outer washer (more mobile, tears less often) | Sagittal PD-FS + coronal | Signal reaching an articular surface on ≥2 images[^12^] |
| Medial OA | Wear-and-tear of the inner femur–tibia compartment: cartilage loss, bone spurs, marrow lesions[^17^] | Coronal + sagittal PD-FS | ≥1 cm of high-grade cartilage loss in the compartment[^12^] |
| Lateral OA | Same, outer compartment | Coronal + sagittal PD-FS | ≥1 cm high-grade cartilage loss[^12^] |
| PF OA | Wear behind the kneecap / trochlear groove | Axial PD-FS | ≥1 cm high-grade cartilage loss[^12^] |
| Effusion | Excess joint fluid; the joint's nonspecific distress signal[^18^] | Sagittal + axial T2-FS/PD-FS | Not separately published; borderline graded negative[^12^] |
| Synovitis | Inflamed, thickened joint lining, often with fluid[^19^] | Sagittal/axial PD-FS (Hoffa's fat pad surrogate) | Not separately published; borderline graded negative[^12^] |
| Baker's | Fluid outpouching into a bursa behind the knee; secondary to intra-articular disease in 87–98% of cases[^20^][^21^] | Axial T2-FS ("speech-bubble" neck) | Not separately published; borderline graded negative[^12^] |
| Contusion | Bone bruise: trabecular microfracture + marrow edema after impaction[^22^] | Any plane, fluid-sensitive (T2-FS/PD-FS/STIR); T1 to characterize | Not separately published; borderline graded negative[^12^] |
| Fracture | Broken bone, often radiographically occult (tibial plateau, patella) | Sagittal/coronal T1 (dark line) + fluid-sensitive edema | Not separately published; borderline graded negative[^12^] |

Three label-interpretation cautions. First, the **meniscal** threshold is the two-image rule from §2.2 — intrameniscal signal that never reaches a surface is common degeneration, not a tear, and the old numeric "grade 1/2/3 signal" reporting scheme is now considered clinically obsolete.[^16^][^23^] Second, **OA** is a whole-joint disease; the competition operationalizes it as high-grade cartilage loss of at least 1 cm in a compartment, which corresponds roughly to the most severe grade (full-thickness loss) of the standard modified Outerbridge cartilage scale.[^24^] The radiographic Kellgren–Lawrence grade you may see in reports cannot be assigned from MRI alone.[^25^] Third, **contusion patterns are diagnostic clues**: a pivot-shift injury leaves bruises on the lateral femoral condyle and posterolateral tibial plateau and strongly implies an ACL tear — so labels are correlated, not independent.[^22^]

## 2.4 How Radiologists Read a Knee MRI — and How Much They Disagree

**The clinical framing.** Most knee MRIs are ordered for "internal derangement of the knee" — an umbrella term for suspected mechanical problems inside the joint (torn meniscus, torn ligaments, loose fragments). MRI is the non-invasive test of choice for this indication and has largely replaced diagnostic arthroscopy (camera surgery into the joint).[^26^]

**The search pattern.** Radiologists do not free-browse; they sweep a fixed checklist so nothing is missed:[^27^] (1) technical quality (motion artifact, failed fat suppression can fake or hide edema); (2) fluid (effusion, Baker cyst); (3) bone marrow (edema patterns map the injury mechanism); (4) ligaments (cruciates on sagittal, collaterals on coronal); (5) menisci on sagittal plus coronal, applying the two-image rule; (6) cartilage on all six articular surfaces; (7) tendons and extensor mechanism; (8) synovium and fat pads; (9) peripheral "don't-forget" areas. The report then follows a fixed skeleton — clinical indication, technique, systematic Findings, and a numbered Impression — and a survey found 47% of referring clinicians skip straight to the Impression.[^28^] Structured-report templates itemize exactly the compartments you see in the label taxonomy (fluid, each meniscus, each ligament, each cartilage compartment).[^29^] The modeling takeaway: the radiologist's checklist is effectively a multi-task classifier over anatomically indexed regions, and each abnormality class has a characteristic (plane, sequence, region) where its signal-to-noise is highest.

**Inter-reader agreement — why your labels are noisy.** Agreement between readers is measured with **Cohen's kappa**, a chance-corrected statistic where 1.0 is perfect agreement and 0 is chance. The conventional interpretation bands are: 0.0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.0 almost perfect.[^30^] Representative values for knee MRI:

| Finding | Kappa | Context | Band |
|---|---|---|---|
| ACL tear | 0.94–0.98 | Two readers, standard and 3D protocols[^31^] | Almost perfect |
| Acute ACL injury | 0.89–0.93 | Two radiologists[^32^] | Almost perfect |
| ACL (MRI vs arthroscopy) | 0.75 | Proxy for ground-truth noise[^33^] | Substantial |
| Medial meniscus | 0.91 reader / 0.60 arthroscopy | [^31^][^33^] | Substantial–almost perfect |
| Lateral meniscus | 0.89 reader / 0.35 arthroscopy | [^31^][^33^] | Weak–almost perfect |
| Cartilage | 0.84 best / 0.03–0.32 arthroscopy | [^31^][^33^] | None–almost perfect |
| Tibial cartilage area (MOAKS) | 0.36 | OA whole-joint scoring[^34^] | Fair |
| Hoffa-synovitis (MOAKS, intra-rater) | 0.42 | Same expert re-reading[^34^] | Moderate |
| Tibial osteophytes (MOAKS) | 0.49 | OA scoring[^34^] | Moderate |

The pattern is consistent: **agreement is best for ACL tears (κ ≈ 0.75–0.98) and worst for cartilage grading (κ ≈ 0.03–0.84 depending on reference standard), with synovitis grading in between (κ ≈ 0.42 intra-rater).**[^31^][^33^][^34^] Even the same expert re-reading Hoffa-synovitis only reaches moderate agreement. Expect label noise to be lowest for ACL and gross meniscal tears and highest for cartilage, synovitis, and lateral meniscus — an argument for soft or ordinal targets on graded findings, and for treating the twelve label AUCs as having very different achievable ceilings.

**A calibration anchor.** The methodological ancestor of this competition is Stanford's MRNet (Bien et al., 2018): a deep learning model trained on 1,370 knee MRIs achieved AUC 0.965 for ACL tears, 0.847 for meniscal tears, and 0.937 for general abnormality detection, with external-validation ACL accuracy (82.4%) comparable to radiologists.[^35^] Those numbers, ranked in the same order as the kappa table above, are not a coincidence: model ceilings track human label reliability.

## 2.5 Glossary and Self-Study Resources

**Glossary.** Terms appear roughly in the order you will meet them in reports and labels.

| Term | Plain-language meaning |
|---|---|
| Condyle | Rounded knob at the end of the femur that rolls on the tibia |
| Tibial plateau | Flat top of the shin bone that carries the menisci |
| Intercondylar notch | Groove between the femoral condyles housing the cruciate ligaments |
| Trochlea | Femoral groove the kneecap glides in |
| Meniscus (medial/lateral) | C-shaped fibrocartilage shock absorber; parts: anterior horn, body, posterior horn, roots |
| Fibrocartilage | Tough, rubbery cartilage (menisci), vs the smooth hyaline cartilage coating bone ends |
| Cruciate ligaments (ACL/PCL) | Crossing central ropes limiting forward/backward sliding of the tibia |
| Collateral ligaments (MCL/LCL) | Side ropes resisting inward/outward buckling |
| Extensor mechanism | Quadriceps tendon → patella → patellar tendon; straightens the knee |
| Hoffa's fat pad | Fat cushion behind the patellar tendon; a standard site for grading synovitis |
| Bursa | Small fluid sac reducing friction, like bubble wrap |
| Synovium / synovial fluid | Joint lining and the lubricant it secretes |
| Internal derangement | Umbrella term for structural problems inside the joint (tears, loose bodies) |
| Sprain / partial / complete tear | Ligament injury severity grades 1–3 |
| Avulsion | Ligament or tendon pulling off its bony attachment, sometimes with a bone chip |
| Tear orientation | Horizontal (cleavage), longitudinal vertical, radial, oblique, complex — the cleft's geometry |
| Bucket-handle tear | Displaced longitudinal meniscal tear flipped into the notch; can lock the knee |
| Meniscal extrusion | Meniscus bulging ≥3 mm past the tibial edge; marker of root tear and OA |
| Parameniscal cyst | Fluid cyst at the meniscal rim; implies an underlying tear |
| Chondromalacia | Cartilage softening/degeneration, graded I–IV (modified Outerbridge) |
| Osteochondral lesion | Damage involving cartilage plus the bone beneath it |
| Loose body | Free fragment of cartilage or bone floating in the joint ("joint mouse") |
| Bone marrow lesion (BML) / contusion | Ill-defined marrow signal change (bright on fluid-sensitive, dark on T1) from bruising or overload |
| Effusion | Excess joint fluid; hemarthrosis = blood in the joint |
| Synovitis | Inflamed, thickened joint lining; definitively diagnosed on contrast-enhanced MRI |
| Baker (popliteal) cyst | Joint fluid squeezed into a bursa behind the knee; rupture mimics a leg blood clot |
| Osteophyte | Bone spur at joint margins; hallmark of OA |
| Subchondral | "Beneath the cartilage" — where BMLs, cysts, and sclerosis of OA occur |
| Kellgren–Lawrence (KL) grade | Radiographic OA severity scale 0–4; not assignable from MRI alone |
| WORMS / BLOKS / MOAKS | Research scoring systems that grade the whole knee on MRI, feature by feature |
| Sagittal / coronal / axial | The three orthogonal imaging planes (side / front / top-down slices) |
| T1W / T2W / PD | Contrast weightings: anatomy / fluid-bright / high-detail intermediate |
| Fat suppression (FS), STIR | Techniques that darken fat so fluid and edema light up |
| Hyperintense / hypointense | Brighter / darker than the reference tissue on a given sequence |

**Self-study resources, in recommended order.**

1. **Chien et al., "Magnetic resonance imaging of the knee," Polish Journal of Radiology 2020** — the single best primer: protocol tables, per-structure MRI sections, and the two-image PPV rule.[^4^]
2. **Radiopaedia** (free, peer-reviewed articles plus real cases): start with "Knee joint,"[^1^] "Anterior cruciate ligament tear,"[^14^] and "Baker cyst."[^20^]
3. **Chana-Rodríguez et al., "Reporting knee meniscal tears," Insights into Imaging 2016** — tear taxonomy, pitfalls, and why old signal-grading is obsolete.[^16^]
4. **Kohn et al., "Classifications in Brief: Kellgren–Lawrence," 2016** — the OA radiographic scale, its history, and its reliability limits.[^36^]
5. **Hunter et al., "MOAKS," Osteoarthritis and Cartilage 2011** — the current MRI whole-joint OA scoring standard; its reliability tables tell you which findings humans score consistently.[^34^]
6. **Bien et al., MRNet (2018)** — read for the task formulation and baseline performance of deep learning on exactly this data type.[^35^]

## Sources

[^1^]: Radiopaedia, "Knee joint" — https://radiopaedia.org/articles/knee-joint-1 (accessed 2026-08-10)
[^2^]: EPOS/ESR educational exhibit, "Acute trauma of the knee ligaments: Following the contusion pattern" (ECR 2016, C-1617) — https://epos.myesr.org/poster/esr/ecr2016/C-1617/background (2016-03-02)
[^3^]: ESSR Refresher Course 2008, "Knee collateral ligaments" (A. Karantanas) — https://www.essr.org/content-essr/uploads/2016/10/Refresher-Course2008.pdf (2008)
[^4^]: Chien A, et al. "Magnetic resonance imaging of the knee." Pol J Radiol 2020 (PMC7571514) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/ (2020)
[^5^]: Radiopaedia, "Synovial joints" (rID-42705) — https://radiopaedia.org/articles/synovial-joints (accessed 2026-08-10)
[^6^]: NIST, "How Does an MRI Machine Work?" — https://www.nist.gov/how-do-you-measure-it/how-does-mri-machine-work (2025-05-14)
[^7^]: RadiologyKey, "Lower Limb II: Knee" — https://radiologykey.com/lower-limb-ii-knee/ (2016-07-24)
[^8^]: MusculoskeletalKey, "The Knee" (MRI protocol chapter) — https://musculoskeletalkey.com/the-knee/ (2016-05-28)
[^9^]: MXR Imaging, "T1 vs. T2 MRI: Key Differences" — https://mxrimaging.com/blogs/t1-vs-t2-mri-imaging/ (2026-03-30)
[^10^]: AJR, "Comparison of Fat-Suppressed T2-Weighted FSE and Modified STIR" — https://ajronline.org/doi/10.2214/ajr.185.2.01850371 (2012)
[^11^]: RadioGraphics, "Fat-Suppression Techniques for 3-T MR Imaging of the Musculoskeletal System" (PMC4359893) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4359893/ (2015)
[^12^]: Cross-Verification — RSNA Knee Abnormality Detection Research (annotation protocol, thresholds, label list, metric) — /mnt/agents/output/research/rsna_knee_cross_verification.md (2026-08-10)
[^13^]: RuntimeWire, "RSNA opens $77,000 challenge for AI that reads knee MRI and reports" — https://runtimewire.com/article/rsna-knee-mri-ai-challenge-2026 (2026-08-07)
[^14^]: Radiopaedia, "Anterior cruciate ligament tear" (rID-12490) — https://radiopaedia.org/articles/anterior-cruciate-ligament-tear (accessed 2026-08-09)
[^15^]: Radsource, "Medial Supporting Structures of the Knee with Emphasis on the MCL" — https://radsource.us/medial-supporting-structures-knee-emphasis-medial-collateral-ligament/ (2024-04-24)
[^16^]: Chana-Rodríguez F, et al. "Reporting knee meniscal tears: technical aspects, typical pitfalls and how to avoid them." Insights Imaging 2016 (PMC4877346) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/ (2016)
[^17^]: Radsource, "Osteoarthritis (OA) of the Knee" — https://radsource.us/osteoarthritis-oa-of-the-knee/ (2023-05-08)
[^18^]: "Joint effusion of the knee: potentialities and limitations…" Insights Imaging 2015 (PMC4630268) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4630268/ (2015)
[^19^]: Burke CJ, et al. "MRI of Synovitis and Joint Fluid" (PMC6504589) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6504589/ (2019)
[^20^]: Radiopaedia, "Baker cyst" (rID-21117) — https://radiopaedia.org/articles/baker-cyst-2 (accessed 2026-08-06)
[^21^]: "Intramuscular Dissecting Baker's Cysts: A Case Series" (PMC10846661) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10846661/ (2024)
[^22^]: EPOS/ESSR 2019 poster P-0162, "Bone Contusion Patterns of the Knee at MRI" — https://epos.myesr.org/poster/esr/essr2019/P-0162/imaging%20findings%20or%20procedure (2019-05-24)
[^23^]: Radiopaedia, "MRI grading system for abnormal meniscal signal intensity" (rID-36617) — https://radiopaedia.org/articles/mri-grading-system-for-abnormal-meniscal-signal-intensity (accessed 2026-08-06)
[^24^]: pacs.de, "Outerbridge grading system" (mirrors Radiopaedia modified Outerbridge grading) — https://pacs.de/term/outerbridge-grading-system (2022-11-08)
[^25^]: "Grading of Knee Osteoarthritis Based on Kellgren-Lawrence Classification…" Cureus 2024 (PMC11624959) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11624959/ (2024)
[^26^]: "MRI of Internal Derangements and Other Knee Pathologies in Adult Nigerians" (PMC11214712) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11214712/ (2024)
[^27^]: Chinmay Gupte, "How do you read a knee MRI" — https://www.chinmaygupte.com/how-do-you-read-a-knee-mri (accessed 2026-08-10)
[^28^]: JACR, "Analysis of Different Levels of Structured Reporting in Knee MRI" — https://www.sciencedirect.com/science/article/abs/pii/S1076633220300131 (2020-10-01)
[^29^]: MusculoskeletalKey, "The Knee" — "BOX 1: The Structured Report: Knee" — https://musculoskeletalkey.com/the-knee-9/ (2016-12-21)
[^30^]: NCBI Bookshelf, "Interpretation of kappa (from Landis and Koch 1977)" — https://www.ncbi.nlm.nih.gov/books/NBK92287/table/executivesummary.t2/ (n.d.)
[^31^]: "Can a single isotropic 3D FSE sequence replace three-plane standard PD FS knee MRI at 1.5 T?" Br J Radiol 2015 (PMC4651376) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4651376/ (2015)
[^32^]: Springer, "Timing of MRI affects the accuracy and interobserver agreement of anterolateral ligament tears detection in ACL deficient knees" — https://link.springer.com/article/10.1186/s43019-020-00082-z (2020-11-27)
[^33^]: ABC Research journal, "Reliability of MRI vs arthroscopy" — https://abcresearch.net/pdf/0fdb9ffe-e838-45c0-b564-25a52c51df96/issues/2026-008-001.pdf (2026)
[^34^]: Hunter DJ, et al. "Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS." Osteoarthritis Cartilage 2011 (PMC4058435) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/ (2011)
[^35^]: Springer J Supercomputing review (2025), summarizing Bien N, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging (MRNet)," PLoS Medicine 2018 — https://link.springer.com/content/pdf/10.1007/s11227-025-07103-2.pdf (2025)
[^36^]: Kohn MD, et al. "Classifications in Brief: Kellgren-Lawrence Classification of Osteoarthritis." Clin Orthop Relat Res 2016 (PMC4925407) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/ (2016)
