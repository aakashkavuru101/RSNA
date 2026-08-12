# Clinical Domain Primer — Knee MRI for the RSNA Knee Abnormality Detection Competition (Dimension 03)

**Audience:** Non-medical ML practitioner. **Compiled:** 2026-08-10. **Scope:** Clinical seed knowledge for detecting/classifying 12 clinically important knee abnormalities from knee MRI exams + multilingual radiology reports.

**Competition context (verbatim):** "RSNA's 2026 Kaggle challenge asks teams to detect 12 knee abnormalities using MRI scans and multilingual radiology reports."
- Source: RuntimeWire, "RSNA opens $77,000 challenge for AI that reads knee MRI and reports"
- URL: https://runtimewire.com/article/rsna-knee-mri-ai-challenge-2026 | Date: 2026-08-07 | Confidence: Medium (trade press; corroborates the brief)

RSNA's own framing (per the brief) targets: ligament/meniscus/cartilage damage, bone marrow lesions, effusion, synovitis, Baker cysts, and osteoarthritis. This primer is organized so each of those maps to a labeled class you may encounter.

---

## 1. Knee Anatomy Essentials (with plain-language analogies)

### 1.1 The big picture

The knee is the body's largest synovial joint — a modified hinge joining three bones.

> **Claim:** The knee joint is a modified hinge joint between the femur, tibia, and patella; the largest synovial joint in the body.
> **Source:** Radiopaedia, "Knee joint" (Su S, Walizai T, et al.)
> **URL:** https://radiopaedia.org/articles/knee-joint-1 | Date: article accessed 2025 (site), accessed for this research 2026-08-10
> **Excerpt (verbatim):** "The **knee joint** is a modified hinge joint between the femur, tibia, and patella. It is the largest synovial joint in the body and allows flexion and extension of the leg as well as some rotation in the flexed position."
> **Confidence:** High

**Analogy:** Think of the knee as a door hinge that also allows a little twist. Two rounded knobs at the bottom of the thigh bone (the femoral condyles) rest on the relatively flat top of the shin bone (the tibial plateau) — like two balls sitting on a table. Because balls-on-a-table is inherently unstable, the knee adds soft-tissue "washers," "ropes," and "padding" to make it work.

There are actually **two joints in one**: the **tibiofemoral joint** (femur–tibia, with medial and lateral compartments) and the **patellofemoral joint** (kneecap–femur).

> **Claim:** The tibiofemoral joint is a modified hinge between distal femur and proximal tibia; menisci deepen and stabilize it.
> **Source:** Radiopaedia, "Tibiofemoral joint"
> **URL:** https://radiopaedia.org/articles/tibiofemoral-joint | Date: accessed 2026-08-10
> **Excerpt (verbatim):** "The **tibiofemoral joint** is a modified hinge synovial joint between the distal femur and the proximal tibia, and forms part of the knee joint. … The medial and lateral menisci increase the depth and stability, and compressive force bearing and absorption of the joint."
> **Confidence:** High

### 1.2 Bones

- **Femur** (thigh bone): its lower end splits into two rounded **condyles** (medial and lateral) that roll on the tibia; the groove between them in front is the **trochlear groove** where the patella glides; the notch between them in the middle is the **intercondylar notch**, home of the cruciate ligaments.
- **Tibia** (shin bone): its flat top is the **tibial plateau** (medial + lateral facets); the small bumps in the middle are the **tibial spines (intercondylar eminence)** where cruciate ligaments and meniscal roots anchor.
- **Patella** (kneecap): a sesamoid bone embedded in the quadriceps tendon; acts like a pulley that increases the leverage of the thigh muscle when straightening the knee.
- **Fibula**: the thin lateral bone of the lower leg; its head anchors the LCL and biceps femoris tendon. (Not part of the knee joint proper but appears in the field of view.)

### 1.3 Ligaments — the "ropes" that stabilize

> **Claim:** Ligamentous anatomy and function of the cruciates.
> **Source:** EPOS/ESR educational exhibit, "Acute trauma of the knee ligaments: Following the contusion pattern" (ECR 2016, C-1617)
> **URL:** https://epos.myesr.org/poster/esr/ecr2016/C-1617/background | Date: 2016-03-02
> **Excerpt (verbatim):** "The ACL is an intraarticular and extrasynovial structure that gives sagittal and rotational stability to the knee. It prevents anterior translation. The ACL originates in the medial aspect of the lateral femoral condyle and inserts on the tibial plateau adjacent to the anterior tibial spine. … The posterior cruciate ligament (PCL) is bigger and stronger than the ACL … prevents subsequent [posterior] translation of the tibia and provides posterolateral rotational stability."
> **Confidence:** High

- **ACL (anterior cruciate ligament):** Runs from the inner wall of the *lateral* femoral condyle to the front of the tibial spine. Stops the tibia sliding forward and controls rotation. The most commonly torn knee ligament; the classic sports injury (pivot-shift mechanism). Two bundles: anteromedial and posterolateral.
- **PCL (posterior cruciate ligament):** Runs from the medial femoral condyle to the back of the tibia. Thicker and stronger than the ACL; stops the tibia sliding backward. Torn by "dashboard" injuries (front of shin driven backward).
- **MCL (medial collateral ligament):** A broad flat band on the inner side of the knee, from medial femur to medial tibia (~5 cm above to ~6–7 cm below the joint line). Resists **valgus** force (knee buckling inward). Has superficial and deep layers; the deep layer is fused to the medial meniscus (which is why MCL and medial meniscus injuries travel together).
- **LCL (lateral collateral ligament):** A cord from lateral femur to fibular head; resists **varus** force (knee bowing outward). Part of the **posterolateral corner** complex (with popliteus tendon, biceps femoris, arcuate ligament) — injuries here are less common but surgically more urgent.

> **Claim:** Normal MCL appearance on MRI.
> **Source:** ESSR Refresher Course 2008, "Knee collateral ligaments" (A. Karantanas)
> **URL:** https://www.essr.org/content-essr/uploads/2016/10/Refresher-Course2008.pdf | Date: 2008
> **Excerpt (verbatim):** "The MCL, a main valgus stabilizer of the knee, consists of two layers separated by a small bursa and peribursal fatty tissue. MCL is normally seen as a linear low-signal intensity structure on all pulse sequences. Its injuries are graded 1-3 based on MR imaging findings and are commonly associated with medial retinaculum injury, medial meniscal tears and meniscocapsular separation."
> **Confidence:** High

### 1.4 Menisci — the "washers/shock absorbers"

> **Claim:** Meniscal shape and attachment.
> **Source:** Radiopaedia, "Knee joint"
> **URL:** https://radiopaedia.org/articles/knee-joint-1 | Date: accessed 2026-08-10
> **Excerpt (verbatim):** "Menisci … fibrocartilaginous, C-shaped in appearance and triangular in cross-section … the medial meniscus is attached to the medial collateral ligament and the lateral meniscus is attached to the popliteus tendon … attached to the femur and tibia via the coronary ligaments."
> **Confidence:** High

Two C-shaped wedges of **fibrocartilage** sit on the tibial plateau: **medial meniscus** (more elongated C, firmly anchored to the MCL → less mobile → torn more often) and **lateral meniscus** (nearly a closed circle, more mobile). Each has an **anterior horn**, **body**, and **posterior horn**, anchored to the central tibia by **root ligaments**.

**Function (key numbers for intuition):**

> **Claim:** Menisci transmit 50% of medial and 70% of lateral compartment load; removal increases contact stress 100% medially and 200–300% laterally.
> **Source:** Chien A, et al. "Magnetic resonance imaging of the knee." Pol J Radiol. 2020 (PMC7571514)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/ | Date: 2020
> **Excerpt (verbatim):** "They function to increase the stability of the knee joint, distribute axial load, absorb shock, and provide lubrication and nutrition to the joint … Studies have shown that 50% of the medial compartment load and 70% of the lateral compartment load are transmitted through the menisci, and removal of the menisci increases contact stress by 100% in the medial compartment and between 200 and 300% in the lateral compartment."
> **Confidence:** High

**Analogy:** the menisci are like rubber washers between two pipe fittings — they deepen the socket, spread the load, and keep the "balls on the table" centered. Only the outer 10–30% has a blood supply in adults (the **"red zone"**, which can heal); the inner portion is avascular (**"white zone"**, tears there don't heal).

### 1.5 Cartilage — the "Teflon coating"

**Articular (hyaline) cartilage** covers the ends of femur, tibia, and the back of the patella — a few millimeters of smooth, water-rich tissue that lets bone glide on bone almost friction-free. It has no blood supply or nerves, so it heals poorly. Damage ranges from softening (**chondromalacia**) through fissuring to full-thickness loss exposing bone.

> **Claim:** Synovial joints have hyaline cartilage covering articular surfaces; menisci are fibrocartilage discs.
> **Source:** Radiopaedia, "Synovial joints" (rID-42705)
> **URL:** https://radiopaedia.org/articles/synovial-joints | Date: accessed 2026-08-10
> **Excerpt (verbatim):** "The articulating surfaces are covered by hyaline cartilage, designed to slide with little friction and to absorb compressive forces. … Additional features within some synovial joints: fibrocartilaginous discs e.g. menisci within the knee joint; intracapsular ligaments e.g. cruciate ligaments within the knee joint."
> **Confidence:** High

### 1.6 Tendons — the "cables from muscle to bone"

- **Quadriceps tendon**: four thigh muscles → top of patella (straightens knee).
- **Patellar tendon (ligament)**: bottom of patella → tibial tuberosity. Site of "jumper's knee."
- Together patella + these tendons = the **extensor mechanism**. Hamstring tendons (semimembranosus, semitendinosus) and popliteus tendon matter posteriorly.

### 1.7 Bursae — the "bubble-wrap cushions"

Small fluid-filled sacs that reduce friction where tendons/skin glide over bone. Key ones: **prepatellar** (in front of kneecap — "housemaid's knee"), **pes anserine** (inner upper tibia), **gastrocnemius–semimembranosus bursa** (behind the knee — this is where a Baker cyst forms), and the **deep infrapatellar bursa**.

> **Claim:** Pes anserine bursitis MRI appearance and differential vs popliteal cyst.
> **Source:** "Bursae around the knee joints," Indian J Radiol Imaging (PMC3354353)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3354353/ | Date: 2012
> **Excerpt (verbatim):** "On MRI, pes anserine bursitis appears as an oblong multiloculated fluid collection seen along the anserine tendons on the posteromedial aspect of the knee … This is best appreciated on T2W axial images. It is commonly confused with a popliteal cyst; the pes anserine bursa is located posteriorly and medially along the semitendinosus, whereas the popliteal cyst is located more often in the midline posteriorly."
> **Confidence:** High

### 1.8 Synovium and joint fluid — the "joint's living space"

The **synovial membrane** lines the joint capsule and secretes **synovial fluid** (lubricant + cartilage nutrition). Normally only a trace of fluid exists. The joint space has named **recesses** (suprapatellar pouch, popliteus recess, perimeniscal recesses, posterior recesses) where fluid collects — important because labels like "effusion" and "synovitis" are scored in these compartments.

> **Claim:** Synovial recess anatomy of the knee.
> **Source:** "Joint effusion of the knee: potentialities and limitations…" Insights Imaging (PMC4630268)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4630268/ | Date: 2015
> **Excerpt (verbatim):** "The synovial membrane … is responsible for the secretion of synovial fluid, which lubricates and nourishes the joint and the removal of intra-articular debris. … The anterior compartment contains the suprapatellar pouch … The popliteal (Baker's) cysts may be considered an articular recess … The knee joint communicates with the proximal tibiofibular joint in approximately 10% of adults."
> **Confidence:** High

### 1.9 Hoffa's fat pad

The **infrapatellar (Hoffa) fat pad** sits behind the patellar tendon; it can become inflamed/impinged ("Hoffitis", fat pad impingement) and is a standard region for grading **Hoffa-synovitis** in OA scoring systems.

---

## 2. MRI Basics for the ML Practitioner

### 2.1 How MRI works (high level)

MRI exploits the magnetic behavior of hydrogen protons (abundant in water and fat). A strong static magnetic field (1.5 T or 3 T in clinical knee imaging) aligns the protons; a tuned radiofrequency (RF) pulse tips them out of alignment; as they "relax" back they emit radio signals that receiver coils detect; gradient fields spatially encode the signal; a computer reconstructs images.

> **Claim:** MRI mechanism (proton alignment, RF pulse, relaxation, spatial encoding).
> **Source:** NIST, "How Does an MRI Machine Work?"
> **URL:** https://www.nist.gov/how-do-you-measure-it/how-does-mri-machine-work | Date: 2025-05-14
> **Excerpt (verbatim):** "The MRI machine then uses a radiofrequency pulse of invisible light to tip the protons' bar magnets out of this alignment. … After the radiofrequency pulse passes through the body, the protons' little magnets move back into their original positions. As the protons 'relax' back into these positions, they release electromagnetic signals that the MRI machine 'sees.' … Each time, the machine turns on spatially varying magnetic fields (known as magnetic field gradients) to encode the location of protons within the MRI signal."
> **Confidence:** High

Two relaxation time constants generate contrast: **T1** (how fast protons realign with the main field) and **T2** (how fast they dephase from each other). By choosing the **TR (repetition time)** and **TE (echo time)**, the scanner emphasizes T1, T2, or proton-density contrast. No ionizing radiation is involved.

> **Claim:** Typical clinical MRI resolves structures of a few millimeters; contrast comes from pulse timing.
> **Source:** Nature Biomedical Engineering, "Making MRI increasingly useful" (2023)
> **URL:** https://www.nature.com/articles/s41551-023-01020-6 | Date: 2023-03-23
> **Excerpt (verbatim):** "By adjusting the strength and timing of the radio and magnetic pulses, a typical clinical MRI scanner can resolve tissue structures of a few millimetres."
> **Confidence:** High

### 2.2 Imaging planes

A knee MRI is not one image but several **series**, each a stack of ~3–4 mm slices acquired in a plane:

| Plane | Orientation (what the slice shows) | Structures best evaluated |
|---|---|---|
| **Sagittal** | Side-view slices, left-to-right | Cruciate ligaments (ACL/PCL), meniscal horns, extensor mechanism (quadriceps/patellar tendon), cartilage step-offs, bone marrow, effusion |
| **Coronal** | Front-view slices, front-to-back | Collateral ligaments (MCL/LCL), meniscal bodies, meniscal extrusion and roots, compartment alignment, cartilage |
| **Axial** | Top-down cross-sections | Patellofemoral joint + patellar cartilage, trochlear groove, retinacula, popliteal fossa / Baker cyst, bursae |

> **Claim:** Plane–structure correspondence for knee MRI.
> **Source:** RadiologyKey, "Lower Limb II: Knee"
> **URL:** https://radiologykey.com/lower-limb-ii-knee/ | Date: 2016-07-24
> **Excerpt (verbatim):** "The sagittal plane is generally the most effective for evaluation of the cruciate ligaments, menisci, patellar ligament, and quadriceps tendon. Coronal sections are needed for evaluation of the medial and lateral collateral ligaments, as well as the menisci. The axial plane is best to evaluate the patellofemoral joint compartment. The axial plane is also helpful in evaluating the popliteal cysts and their relationship to the surrounding structures of the popliteal fossa."
> **Confidence:** High

> **Claim (corroborating, textbook):** Meniscal pathology is evaluated primarily on sagittal; roots on posterior coronal; cruciates best on sagittal with coronal/axial confirmation; collaterals on coronal/axial; patellofemoral cartilage on axial and sagittal.
> **Source:** MusculoskeletalKey, "The Knee" (MRI protocol chapter)
> **URL:** https://musculoskeletalkey.com/the-knee/ | Date: 2016-05-28
> **Excerpt (verbatim):** "Meniscal pathology is evaluated primarily on sagittal plane images. However, the morphology and signal intensity of meniscal fibrocartilage should be assessed secondarily on coronal and axial plane images. The meniscal root attachments are evaluated on posterior coronal images. The cruciate ligaments are best seen on sagittal plane images, with coronal and axial views for secondary visualization and confirmation of pathology. … The medial and lateral collateral ligaments (MCL and LCL) are displayed on coronal and axial images … The patellofemoral joint, including the patellar facets and the trochlear groove chondral surfaces, are assessed on axial and sagittal images."
> **Confidence:** High

### 2.3 Pulse sequences and what tissue looks like

**The core mental model:** water/fluid/edema is the star of pathology detection. Sequences either show anatomy crisply (T1, PD) or make fluid glow (T2, PD-FS, STIR).

| Sequence | Fat | Fluid/edema | Muscle | Tendon/ligament/meniscus/cortical bone | Best for |
|---|---|---|---|---|---|
| **T1-weighted** | Bright | Dark | Intermediate | Dark | Anatomy, marrow fat (marrow replacement = dark on T1), fracture lines, subacute blood, post-contrast imaging |
| **T2-weighted** | Bright (FSE) | **Bright** | Intermediate | Dark | Fluid, edema, cysts, effusion, inflammation |
| **PD (proton density)** | Bright | Intermediate | Intermediate | Dark, high detail | Best SNR/anatomic detail: menisci, ligaments, cartilage |
| **T2/PD fat-suppressed (FS)** | **Dark** | **Bright** (pops against dark fat) | Intermediate | Dark | The workhorse: bone marrow edema, meniscal tears, ligament tears, cartilage, effusion |
| **STIR** | Very dark | Very bright | Intermediate | Dark | Uniform fat suppression; sensitive to edema; robust near metal/inhomogeneous fields |

> **Claim:** Signal characteristics overview.
> **Source:** MXR Imaging, "T1 vs. T2 MRI: Key Differences" (vendor education; consistent with standard teaching)
> **URL:** https://mxrimaging.com/blogs/t1-vs-t2-mri-imaging/ | Date: 2026-03-30
> **Excerpt (verbatim):** "Elements that will appear brighter on a T1 weighted image include: Fat, Blood, MRI contrast elements (gadolinium)… The following are the brighter elements that can be captured on a T2 weighted image: Fluid… Because water and fluids appear brighter on T2 weighted images, T2 imaging is typically used when looking for areas of inflammation. In general, this also means that T2 imaging tends to be used for pathology."
> **Confidence:** Medium-High (commercial source; content is textbook-standard)

> **Claim:** Why fat suppression matters: fat stays bright on FSE T2, masking fluid; FS increases conspicuity.
> **Source:** AJR, "Comparison of Fat-Suppressed T2-Weighted FSE and Modified STIR … Rotator Cuff" (principles generalize to MSK MRI)
> **URL:** https://ajronline.org/doi/10.2214/ajr.185.2.01850371 | Date: 2012 (online)
> **Excerpt (verbatim):** "One inherent disadvantage of the fast spin-echo technique is that fat remains bright on these images … The addition of chemical fat suppression increases the conspicuousness of fluid within small defects … STIR sequences have the advantage of increasing the relative signal intensity of fluid as a result of the additive T1 and T2 contrast effect."
> **Confidence:** High

> **Claim:** STIR vs chemical fat-sat trade-offs (uniformity vs SNR/speed).
> **Source:** RadioGraphics (PMC4359893), "Fat-Suppression Techniques for 3-T MR Imaging of the Musculoskeletal System"
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4359893/ | Date: 2015
> **Excerpt (verbatim):** "STIR is insensitive to B0 and B1 heterogeneity. Thus, STIR sequences are widely used in musculoskeletal imaging because of the improved fat suppression that may be achieved … in the presence of metal … The major clinical limitations of STIR sequences are their relatively long imaging times, low SNR, and high SAR."
> **Confidence:** High

### 2.4 A real-world knee MRI protocol

> **Claim:** Standard knee MRI protocols use 3 orthogonal planes + fluid-sensitive FS sequences + T1; PD FS is the meniscal workhorse; 1.5T/3T typical.
> **Source:** Chien A, et al. "Magnetic resonance imaging of the knee." Pol J Radiol 2020 (PMC7571514)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/ | Date: 2020
> **Excerpt (verbatim):** "MRI protocols of the knee commonly consist of three orthogonal imaging planes of section (axial, coronal, and sagittal), with a combination of fluid-sensitive sequences, either T2-weighted (T2W) fat-saturated (FS) or proton density-weighted (PDW) FS sequences, and T1-weighted (T1W) non-fat-saturated (NFS) imaging. Coronal and sagittal PDW sequences provide high SNR and spatial resolution and are more sensitive than T2W sequences for detecting meniscal pathology … T1W images are usually performed without fat saturation and should be obtained to evaluate the bone marrow fat for marrow replacing processes or to detect fracture lines." Routine 3T protocol table: Sagittal PD FS (TR 3000/TE 37/3 mm), Sagittal T1 (600/17), Coronal PD FS, Axial PD FS.
> **Confidence:** High

> **Claim:** Example 1.5T institutional protocol (6 series).
> **Source:** PMC4651376, "Can a single isotropic 3D FSE sequence replace three-plane standard PD FS knee MRI at 1.5 T?" (Br J Radiol)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4651376/ | Date: 2015
> **Excerpt (verbatim):** "All patients underwent imaging according to the institution's standard knee protocol: sagittal PD FS and T2 FS, coronal PD FS and T1, and axial PD FS." (Slice thickness 3 mm.)
> **Confidence:** High

**ML translation:** expect each exam to contain 4–7 series differing in plane × contrast; series count, scanner vendor, field strength (1.5T vs 3T), and even language/institution vary — a major domain-shift axis across the 16 institutions. Slices within a series are spatially ordered (a volume), but series are not co-registered to each other.

### 2.5 Why radiologists need multiple planes AND sequences

1. **Confirmation rule (two-plane rule):** a true tear should be visible in two planes or on two consecutive images; partial-volume averaging on a single slice mimics pathology.
2. **Geometry:** some structures run obliquely (ACL) or are long bands (MCL) — no single plane contains them.
3. **Contrast complementarity:** T1 shows marrow fat and anatomy; PD gives resolution; FS fluid-sensitive sequences make edema/tears conspicuous. What is invisible on one sequence glows on another.
4. **Pitfall avoidance:** magic-angle artifact, normal variant ligaments (e.g., oblique meniscomeniscal ligament mimicking a bucket-handle tear) are resolved by scrolling through consecutive slices and cross-checking planes.

> **Claim:** The two-image rule quantified: abnormal meniscal signal touching the surface on ≥2 images → PPV of tear 94% (medial) / 96% (lateral); on one image only → 43% / 18%.
> **Source:** Chien A, et al. PMC7571514
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/ | Date: 2020
> **Excerpt (verbatim):** "Using arthroscopy as the gold standard, if an abnormal increased linear signal is identified on at least two images (two consecutive coronal or sagittal images, or one sagittal and one coronal at the same location), the positive predictive value (PPV) of a tear is 94% in the medial meniscus, and 96% in the lateral meniscus. If the abnormal signal touches the articular surface on only one image, the PPV of tear is 43% in the medial meniscus and 18% in the lateral meniscus."
> **Confidence:** High

---

## 3. The Target Abnormalities

For each: what it is / MRI appearance / best plane+sequence / clinical significance / grading.

### 3.1 ACL tear

**What:** Rupture (complete or partial) of the anterior cruciate ligament — the most common knee ligament injury; classic pivot-shift sports mechanism; causes instability ("giving way").

> **Claim:** ACL tears are the most common knee ligament injury; clinical presentation.
> **Source:** Radiopaedia, "Anterior cruciate ligament tear" (rID-12490)
> **URL:** https://radiopaedia.org/articles/anterior-cruciate-ligament-tear | Date: accessed 2026-08-09 (Radiopaedia stamp)
> **Excerpt (verbatim):** "**Anterior cruciate ligament (ACL) tears** are the most common knee ligament injury encountered in radiology and orthopedic practice. … Patients typically present with symptoms of knee instability, usually after acute trauma. … popping sensation at the time of injury, followed by swelling … knee felt to 'giving way', especially during pivoting movements. The classic mechanism is the 'pivot-shift', where the loaded knee is slightly flexed and rotated internally with valgus stress."
> **Confidence:** High

**MRI appearance — primary signs (verbatim list from Radiopaedia):**
- "swelling"
- "increased signal on T2 or fat-saturated PD"
- "fiber discontinuity"
- "abnormal anterior cruciate ligament orientation relative to the intercondylar (Blumensaat) line … ACL angle … >15° … indicating a ruptured and collapsed ligament"
- "empty notch sign: a fluid signal at the site of femoral attachment at the intercondylar notch, denotes avulsion at the femoral attachment."

**Secondary signs (verbatim):** "bone contusion in the lateral femoral condyle and the posterolateral tibial plateau … >7 mm of anterior tibial translation … uncovered posterior horn of the lateral meniscus … Segond fracture (occurs in ~2.5% of ACL tears) … bowing of PCL: reduced PCL angle <105-107° … medial or lateral collateral ligament injury … lateral femoral sulcus deeper than 2 mm."
(Source/URL/date as above; Confidence: High)

**Best plane/sequence:** Sagittal (oblique sagittal along the ligament ideally), PD FS / T2 FS; coronal and axial to confirm. **Significance:** drives instability, often reconstructed; associated meniscal/cartilage damage and post-traumatic OA ("Approximately 10% (range 2-100%) of patients develop post-traumatic osteoarthritis after ACL reconstruction" — Radiopaedia). **Grading:** sprain / partial tear / complete tear; chronic tears may reconstitute low signal with focal angulation.

**Diagnostic performance context:**

> **Claim:** Meta-analyzed MRI sensitivity/specificity vs arthroscopy: ACL 87%/93%, medial meniscus 92%/90%, lateral meniscus 80%/95%.
> **Source:** PMC11214712, "MRI of Internal Derangements and Other Knee Pathologies in Adult Nigerians" (2024), citing meta-analyses
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11214712/ | Date: 2024
> **Excerpt (verbatim):** "Compared to arthroscopy, the respective meta-analyzed sensitivity and specificity of knee MRI for ACL and meniscal tears are ACL (87%; 93%), medial meniscus (92%; 90%), and lateral meniscus (80%; 95%)."
> **Confidence:** High

### 3.2 PCL tear

**What:** Tear of the posterior cruciate ligament; classic "dashboard injury" (posterior force on flexed tibia) or hyperextension. Less common than ACL tears.

**MRI:** fiber discontinuity / increased T2 signal / thickening; best on sagittal PD or T2. Normal PCL is a dark, gently curved ("arcuate") band on all sequences.

> **Claim:** PCL injury epidemiology on MRI and associated injuries.
> **Source:** Sonin AH et al., "Posterior cruciate ligament injury: MR imaging diagnosis and patterns of injury," Radiology 1994 (PubMed)
> **URL:** https://pubmed.ncbi.nlm.nih.gov/8284399/ | Date: 1994-02
> **Excerpt (verbatim):** "Twenty-one patients (45%) had complete PCL tears; 22 patients (47%), partial tears; and four patients (9%), bone avulsion. Associated injuries were seen in 34 patients (72%). Patterns of injuries differed from those seen in anterior cruciate ligament injury and correlated with the mechanism of trauma."
> **Confidence:** High

> **Claim:** Dashboard injury contusion pattern + PCL association.
> **Source:** EPOS/ESSR 2019 poster P-0162, "Bone Contusion Patterns of the Knee at MRI"
> **URL:** https://epos.myesr.org/poster/esr/essr2019/P-0162/imaging%20findings%20or%20procedure | Date: 2019-05-24
> **Excerpt (verbatim):** "Dashboard injury results from a force that is applied to the anterior aspect of the tibia while the knee is a flexed position. … The characteristic bone contusion pattern involves the anterior aspect of the tibia and, occasionally, the posterior surface of the patella. … This injury is very commonly associated with posterior cruciate ligament (PCL) tear (most commonly in its midsubstance) and rupture of the posterior joint capsule."
> **Confidence:** High

**Grading:** partial vs complete; can be hard to separate on standard MRI (flexed-knee MRI described as a trick: partial tears on extension imaging were found to be complete ruptures at 90° flexion — Craddock et al., Knee 2018, PMID 29548815). PCL heals/scars in continuity more often than ACL.

### 3.3 Collateral ligament injuries (MCL, LCL)

**What:** Sprains/tears of the side stabilizers. MCL injured by valgus blows ("clip injury"); very common in athletes. LCL/posterolateral corner injuries rarer but often need surgery.

> **Claim:** MCL injury mechanism and MRI grading (Grade 1: periligamentous edema, intact ligament; Grade 2: partial disruption; Grade 3: complete disruption).
> **Source:** Gaurav Kumar thesis, "Role of MRI in evaluating traumatic knee injury…" (SDUAHER repository), citing standard orthopedic references
> **URL:** https://dspace.sduaher.ac.in/jspui/bitstream/123456789/9342/1/Dr.%20GAURAV%20KUMAR%20%208%20COPY%20BLUE%20SILVER-1.pdf | Date: n.d.
> **Excerpt (verbatim):** "Grade 1: (Minor sprain) high signal is seen medial (superficial) to the ligament, which looks normal. Grade 2: (Severe sprain or partial tear) high signal is seen medial to the ligament, with high signal or partial disruption of the ligament. Grade 3: complete disruption of the ligament. In addition, MRI allows the depiction of associated injuries as bone bruises, posterior oblique ligament and anterior cruciate ligament injuries as well as meniscal tears."
> **Confidence:** Medium-High (thesis citing standard texts; corroborated by Radsource below)

> **Claim:** Caveat: the 3-grade scheme is really a clinical scheme; radiologists apply MR correlates (edema paralleling ligament / attenuation-thickening / loss of continuity).
> **Source:** Radsource, "Medial Supporting Structures of the Knee with Emphasis on the MCL"
> **URL:** https://radsource.us/medial-supporting-structures-knee-emphasis-medial-collateral-ligament/ | Date: 2024-04-24
> **Excerpt (verbatim):** "Although many radiologists refer to grade 1 injuries in the presence of soft tissue edema paralleling the TCL, grade 2 injuries in the presence of periligamentous edema and areas of ligamentous attenuation with focal or segmental areas of ligament thickening, and grade 3 injuries when confronted with complete loss of continuity of ligamentous fibers with or without capsular involvement … the classic three-grade scheme is better applied to the clinical assessment and not the MR imaging assessment of the medial supporting structures."
> **Confidence:** High

**Best plane/sequence:** Coronal PD FS / T2 FS (both collaterals run in the coronal plane); axial as secondary. **Significance:** MCL grades 1–2 usually heal conservatively; grade 3 + multi-ligament injuries (e.g., O'Donoghue unhappy triad = ACL + MCL + medial meniscus) change management. LCL injuries "do not recover well with nonoperative treatment and usually require urgent surgical repair or reconstruction" (CURRENT Medical Diagnosis & Treatment 2024, excerpt captured; Confidence: High).

### 3.4 Meniscal tears

**What:** A cleft in the C-shaped fibrocartilage. Diagnosis = linear high signal inside the normally-black meniscus that reaches an articular surface, or abnormal meniscal shape.

> **Claim:** MRI is the most accurate imaging technique for meniscal lesions; standard knee evaluation tool; distribution of tears.
> **Source:** Chana-Rodríguez F, et al. "Reporting knee meniscal tears: technical aspects, typical pitfalls and how to avoid them," Insights Imaging (PMC4877346)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/ | Date: 2016
> **Excerpt (verbatim):** "Magnetic resonance imaging (MRI) is the most accurate imaging technique in the diagnosis of meniscal lesions and represents a standard tool in knee evaluation. … Half of the meniscal tears involve the medial meniscus, and in 98 % of the cases, the tear is within the posterior horn and the body of the meniscus. … MRI diagnosis is based on the presence of linear signal changes that come in contact with the meniscal surfaces, or is based on the shape and size alterations of the meniscus." Sensitivity/specificity: "a sensitivity of 93.3 % and a specificity of 88.4 % for the medial meniscus and a sensitivity of 79.3 % and a specificity of 95.7 % for the lateral meniscus."
> **Confidence:** High

**Tear orientation taxonomy (verbatim from PMC4877346):**
- "An horizontal tear is defined as a linear signal abnormality involving the surface of the meniscus in a horizontal orientation of less than 30° relative to the adjacent tibial plateau."
- "Radial tears are perpendicular to the long axis of the meniscus and begin in the free edge of the meniscus."
- "Vertical longitudinal tears are parallel to the long axis of the meniscus, away from the free edge."
- "A complex tear refers to a combination of more orientations (e.g. parrot beak tear). Tears with displaced fragments such as bucket-handle tear, flap meniscus tear, or free meniscus fragment are also classified as complex tears."

**Key named patterns:**
- **Bucket-handle tear:** displaced longitudinal tear; the inner fragment flips into the intercondylar notch → **"double PCL sign"** on sagittal images ("the displaced meniscus … that lies anterior to the anterior cruciate ligament" gives a "double anterior cruciate ligament sign" when displaced forward — PMC4877346). Can lock the knee (surgical urgency).
- **Radial/root tears:** "Root failure behaves like total meniscectomy in biomechanics" (moolchand blog summary, Confidence: Medium) — root tears rapidly accelerate OA; 76% of medial meniscus root tears have extrusion (Chien 2020, PMC7571514).
- **Horizontal (cleavage) tears:** typically degenerative, in patients >40, often associated with **parameniscal cysts** (PPV of tear 90% when a cyst is present — Chien 2020).

**Grading — important nuance for label interpretation:** the classic Lotysch/Crues intrameniscal signal grading exists but is deprecated for reporting tears:

> **Claim:** Grade 1–3 meniscal signal system (Lotysch).
> **Source:** Radiopaedia, "MRI grading system for abnormal meniscal signal intensity" (rID-36617)
> **URL:** https://radiopaedia.org/articles/mri-grading-system-for-abnormal-meniscal-signal-intensity | Date: accessed 2026-08-06 (Radiopaedia stamp)
> **Excerpt (verbatim):** "grade 1: small focal area of hyperintensity, no extension to the articular surface … grade 2: linear areas of hyperintensity, no definite extension to the articular surface … grade 3: abnormal hyperintensity extends to at least one articular surface (superior or inferior), on more than one consecutive image, and is referred to as a definite meniscal tear."
> **Confidence:** High

> **Claim:** Grade I/II/III reporting is now considered obsolete clinically; report stable vs unstable, orientation, surfaces involved.
> **Source:** PMC4877346 (Insights Imaging)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/ | Date: 2016
> **Excerpt (verbatim):** "Meniscal grading, e.g. grade I, II, or III, which has been widely used in the past, is now considered obsolete in clinical routine imaging. … Thus, a standard knee report may better distinguish between stable and unstable tears … Stable tears have a potential for healing conservatively, whereas unstable tears often require surgery."
> **Confidence:** High

**ML caution:** intrameniscal signal that does NOT reach a surface is common and not a tear; many tears are asymptomatic in older patients (Englund NEJM 2008, incidental meniscal findings in middle-aged/elderly). Expect noisy labels between "degeneration" and "tear."

**Meniscal extrusion:** meniscal tissue pushed ≥3 mm beyond the tibial plateau margin (best seen on coronal images) — a marker of root tear / advanced degeneration and a driver of OA progression.

> **Claim:** Extrusion definition and significance.
> **Source:** PMC6219866, "Radiographic Evaluation of Meniscal Extrusion" (Cureus 2018)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6219866/ | Date: 2018
> **Excerpt (verbatim):** "Meniscal extrusion has traditionally been defined as meniscal tissue extending 3 mm or greater beyond the edge of the tibial plateau, excluding marginal osteophytes, measured on magnetic resonance imaging (MRI). … Meniscal extrusion is a recognized imaging finding that has been associated with a posterior root tear of the meniscus and osteoarthritis."
> **Confidence:** High

### 3.5 Articular cartilage damage (chondromalacia, chondral/osteochondral defects)

**What:** Softening, fissuring, thinning, or loss of the hyaline cartilage coating. "Chondromalacia patellae" = cartilage damage behind the kneecap, a common cause of anterior knee pain.

**MRI appearance:** focal high signal within cartilage on PD FS, surface fraying/fissures, partial- or full-thickness defects down to subchondral bone; look for underlying marrow edema and subchondral cysts. Best assessed on **fat-saturated PD** sequences; patellofemoral cartilage on **axial**, tibiofemoral on coronal+sagittal.

**Grading — modified Outerbridge (MRI), the most-used clinical scheme:**

> **Claim:** Modified Outerbridge MRI grading of chondromalacia.
> **Source:** pacs.de term entry (mirrors Radiopaedia "Modified Outerbridge grading of chondromalacia"); corroborated by EPOS ESSR 2017 poster P-0316
> **URL:** https://pacs.de/term/outerbridge-grading-system | Date: 2022-11-08
> **Excerpt (verbatim):** "The modified Outerbridge grading of chondromalacia is divided into four grades by MRI, typically using fat-saturated proton density sequences. … grade I — focal areas of hyperintensity with normal contour … grade II — blister-like swelling/fraying of articular cartilage extending to surface … grade III — partial-thickness cartilage loss with focal ulceration … grade IV — full-thickness cartilage loss with underlying bone reactive changes."
> **Confidence:** High

Original (arthroscopic) Outerbridge: grade 1 softening/swelling; 2 fragmentation/fissuring <1.3 cm; 3 fragmentation/fissuring >1.3 cm; 4 erosion down to bone (Outerbridge 1961, via RadiologyKey; Confidence: High). Research datasets may instead use **WORMS** cartilage grades (0–6 scale with half-grades: "grade 1: increased signal … grade 2: partial-thickness focal defect <1 cm; grade 2.5: full-thickness focal defect <1 cm … grade 4: diffuse partial-thickness loss" — Cochrane review excerpt, PMID-linked; Confidence: High) or **MOAKS** subregional size/extent grading.

**Significance:** cartilage loss is the core lesion of osteoarthritis; full-thickness defects may prompt cartilage repair surgery. MRI detects chondromalacia "in the lower stages with a sensitivity of 66%. Moreover, this sensitivity rises to 85–100% for the higher stages" (PMC7276644; Confidence: High).

### 3.6 Bone marrow lesions (BMLs) / bone bruises / contusions

**What:** Ill-defined areas of high signal in the marrow on fluid-sensitive sequences (and low signal on T1), histologically a mix of edema, necrosis, fibrosis, and microfractures of trabeculae — hence the preferred term "bone marrow **lesion**" rather than "edema." Two big flavors:
- **Traumatic (bone bruise/contusion):** trabecular microfracture + hemorrhage after impaction; distribution reveals the injury mechanism (pivot-shift → lateral femoral condyle + posterolateral tibial plateau = classic ACL-tear pattern; "kissing contusions" of anterior femur+tibia in hyperextension).
- **OA-associated (subchondral BML):** under damaged cartilage in overloaded compartments; strongly linked to pain and progression.

> **Claim:** Bone bruise definition, MRI signal, and prognostic implications.
> **Source:** EPOS/ESSR 2019 poster P-0162, "Bone Contusion Patterns of the Knee at MRI"
> **URL:** https://epos.myesr.org/poster/esr/essr2019/P-0162/imaging%20findings%20or%20procedure | Date: 2019-05-24
> **Excerpt (verbatim):** "Bone bruises are focal abnormalities in subchondral bone marrow due to trabecular microfractures as a result from a traumatic force. … radiographically occult osseous injuries are frequently identified at MRI as areas of poorly marginated signal intensity abnormalities in the bone marrow (decreased signal intensity on T1-weighted sequences and increased signal intensity on T2-weighted or proton density (PD) weighted fat-suppressed (FS) or Short Tau Inversion Recovery (STIR) sequences). … The recovery period is quiet variable, ranging from 3 weeks to 2 years. These lesions may have deleterious effect on the overlying articular cartilage evolving to articular cartilage degeneration."
> **Confidence:** High

> **Claim:** OA BMLs predict pain, progression, and total knee arthroplasty (TKA).
> **Source:** PMC6578476, "Clinical and Pathophysiologic Significance of MRI Identified Bone Marrow Lesions Associated with Knee Osteoarthritis" (J Funct Morphol Kinesiol 2019)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6578476/ | Date: 2019
> **Excerpt (verbatim):** "Scher et al. reported that patients with knee OA and a BML had a nine-fold likelihood to progress to TKA when compared to controls without a BML … Roemer et al. reported that the presence of a large BML or an increase in BML size prognosticated both clinical deterioration and performance of TKA in patients with OA (2.5x and 3.4x need for TKA respectively)."
> **Confidence:** High

**Best sequences:** T2 FS / PD FS / STIR (fluid-sensitive); T1 to exclude marrow replacement or fracture line. **Grading:** MOAKS scores BMLs by % of subregion volume involved (0 = none; 1 <33%; 2 = 33–66%; 3 >66%); WORMS 0–3 by extent.

---

### 3.7 Joint effusion

**What:** Excess synovial fluid inside the joint — the joint's nonspecific distress signal (trauma, meniscal/ligament injury, arthritis, infection). Blood in the joint = **hemarthrosis** (fluid–fluid levels on MRI); fat+blood = **lipohemarthrosis** (implies fracture).

> **Claim:** Causes of knee effusion.
> **Source:** PMC4630268, "Joint effusion of the knee" (Insights Imaging)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4630268/ | Date: 2015
> **Excerpt (verbatim):** "The first manifestation of synovial disease is joint effusion. Knee effusion may be the result of trauma, overuse or systemic disease. Overuse syndromes, ligamentous, osseous and meniscal injuries are the most common causes of effusion. Arthritis, infections, crystal deposition, pigmented villonodular synovitis, osteochondromatosis and tumors are other possible causes of effusion."
> **Confidence:** High

**MRI appearance:** bright fluid distending the suprapatellar pouch and recesses on T2 FS/PD FS (best judged on sagittal and axial); simple effusion is dark on T1. Reports grade it subjectively as **small/moderate/large** (see structured-report template in §4.3). **Significance:** a marker of internal derangement or inflammation; acute large effusion after trauma strongly suggests ACL tear or fracture. Effusion-synovitis on MRI predicts future cartilage loss (MOST study; Roemer 2011, Ann Rheum Dis — reference captured).

### 3.8 Synovitis

**What:** Inflammation and thickening of the synovial lining, with increased fluid production. Seen in OA (low-grade, "effusion-synovitis"), inflammatory arthritis, infection, trauma, and special entities (PVNS, lipoma arborescens, synovial chondromatosis).

**MRI appearance:** thickened, frond-like synovium outlined by bright fluid on fluid-sensitive sequences; **definitive diagnosis needs IV gadolinium** — inflamed synovium enhances on post-contrast T1 FS while fluid does not. Non-contrast MRI surrogates: synovial thickening in the suprapatellar pouch and Hoffa's fat pad ("Hoffa-synovitis").

> **Claim:** Synovitis imaging: classic signs are effusion + synovial thickening; enhancement post-contrast.
> **Source:** EPOS/ECR 2013 poster C-2443, "Synovial knee disease: MRI differential diagnosis"
> **URL:** https://epos.myesr.org/poster/esr/ecr2013/C-2443/background | Date: 2013-03-07
> **Excerpt (verbatim):** "synovitis is described as irritation and inflammation of the joint lining, anatomically called synovium. … The most classical radiological signs are joint effusion and synovial thickening."
> **Confidence:** High

> **Claim (case-level corroboration):** synovitis appearance on post-contrast MRI.
> **Source:** Radiopaedia case, "Synovitis - knee" (rID 68579)
> **URL:** https://radiopaedia.org/cases/synovitis-knee | Date: 2019-02-25
> **Excerpt (verbatim):** "The synovium is abnormally thickened returning low signal on both T1 and T2 weighted image with intense post contrast enhancement."
> **Confidence:** High

**Grading:** in OA research, effusion-synovitis and Hoffa-synovitis are graded 0–3 (MOAKS). **Significance:** synovitis in knee OA correlates with pain and radiographic progression (Burke CJ 2019, "MRI of Synovitis and Joint Fluid," PMC6504589 — "Synovitis in knee osteoarthritis assessed by contrast-enhanced magnetic resonance imaging (MRI) is associated with radiographic tibiofemoral osteoarthritis"; Confidence: High).

### 3.9 Baker (popliteal) cyst

**What:** A fluid-filled outpouching of the joint into the gastrocnemius–semimembranosus bursa in the popliteal fossa (back of knee). Not a true cyst — it's joint fluid squeezed through a one-way valve into the bursa, almost always secondary to an intra-articular problem (OA, meniscal tear) that raises joint fluid production.

> **Claim:** Definition, location, MRI signal.
> **Source:** Radiopaedia, "Baker cyst" (rID-21117)
> **URL:** https://radiopaedia.org/articles/baker-cyst-2 | Date: accessed 2026-08-06 (Radiopaedia stamp)
> **Excerpt (verbatim):** "**Baker cysts**, or **popliteal cysts**, are fluid-filled distended synovial-lined lesions arising in the popliteal fossa between the medial head of the gastrocnemius and the semimembranosus tendons via a communication with the knee joint. … Signal characteristics — T1: low; T2: high. … Recognized complications include: dissection … rupture: leaking of cyst fluid into the popliteal fossa … compression: of the popliteal vessels and tibial nerve."
> **Confidence:** High

> **Claim:** Association with internal derangement.
> **Source:** PMC10846661, "Intramuscular Dissecting Baker's Cysts: A Case Series" (2024), citing Miller 1996 & Fielding 1991
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10846661/ | Date: 2024
> **Excerpt (verbatim):** "Baker's cysts are frequently associated with degenerative knee arthropathy. MRI studies of Baker's cysts reveal an association with intra-articular lesions in 87% to 98% of cases."
> **Confidence:** High

**Best plane/sequence:** axial T2 FS shows the "speech-bubble" neck between the medial gastrocnemius head and semimembranosus tendon; sagittal/coronal PD FS confirm. **Significance:** usually incidental and asymptomatic; rupture mimics deep vein thrombosis (calf pain/swelling). In children they usually resolve spontaneously.

### 3.10 Osteoarthritis (OA)

**What:** Chronic "wear-and-tear" whole-joint disease: articular cartilage loss, subchondral bone changes (sclerosis, cysts, BMLs), osteophytes (bone spurs), meniscal degeneration/extrusion, low-grade synovitis. The knee's medial compartment and patellofemoral joint are typical sites.

**X-ray vs MRI:** the classic **Kellgren–Lawrence (KL)** grade is radiographic (it cannot be assigned from MRI alone — important if the competition includes radiographs or report-derived KL labels):

> **Claim:** Kellgren–Lawrence grades 0–4.
> **Source:** PMC11624959, "Grading of Knee Osteoarthritis Based on Kellgren-Lawrence Classification…" (Cureus 2024)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11624959/ | Date: 2024
> **Excerpt (verbatim):** "Grade 0: No radiographic findings of osteoarthritis • Grade 1: Minute osteophytes of doubtful clinical significance • Grade 2: Definite osteophytes with unimpaired joint space • Grade 3: Definite osteophytes with moderate joint space narrowing • Grade 4: Definite osteophytes with severe joint space narrowing and subchondral sclerosis."
> **Confidence:** High

> **Claim:** KL is the most widely used clinical radiographic OA tool; reliability history (knee had the highest interobserver correlation r=0.83 in the original 1957 study; later studies ICC 0.51–0.89).
> **Source:** Kohn MD, et al. "Classifications in Brief: Kellgren-Lawrence Classification of Osteoarthritis," Clin Orthop Relat Res 2016 (PMC4925407)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/ | Date: 2016
> **Excerpt (verbatim):** "Currently, the KL classification is the most widely used clinical tool for the radiographic diagnosis of OA. … They found that the tibiofemoral joint of the knee had the highest interobserver correlation coefficient of r = 0.83 … the KL system … had an interobserver reliability intraclass correlation coefficient of 0.51 to 0.89 … from studies since the original KL article."
> **Confidence:** High

**MRI features of OA (whole-organ view):**

> **Claim:** Characteristic MRI features of knee OA.
> **Source:** Radsource, "Osteoarthritis (OA) of the Knee"
> **URL:** https://radsource.us/osteoarthritis-oa-of-the-knee/ | Date: 2023-05-08
> **Excerpt (verbatim):** "MRI features characteristic of OA include focal loss of articular (hyaline) cartilage, osteophytes, subchondral marrow lesions, and joint effusion. Frequently seen with OA and with a probable association are meniscal tears, especially meniscal extrusion, and periligamentous edema at the MCL."
> **Confidence:** High

**Research grading systems you may see in labels or literature:** **WORMS** (Whole-Organ MRI Score; Peterfy 2004, Osteoarthritis Cartilage — scores cartilage, BMLs, menisci, osteophytes, synovitis, effusion, ligaments per subregion), **BLOKS** (Boston-Leeds; Hunter 2008), and **MOAKS** (MRI Osteoarthritis Knee Score; Hunter 2011 — the current standard; reliability mostly kappa 0.61–1.0, see §4.4).

**Significance:** leading cause of chronic knee pain and arthroplasty; MRI detects OA years before radiographs (early cartilage/BML changes precede joint space narrowing). Note the poor correlation between pain and KL grade — imaging severity ≠ symptoms (PMC11624959: "no significant correlation between VAS pain score and severity of osteoarthritis knee as per Kellgren-Lawrence grading"; Confidence: High).

### 3.11 Other findings that may appear among the 12 labels or in reports

- **Patellar/quadriceps tendinopathy & tears** ("jumper's knee"): thickened proximal patellar tendon with increased T1/PD and T2 signal; normal AP diameter ≤7 mm (Radiopaedia "Patellar tendinopathy," rID-92897; Confidence: High). Best on sagittal.
- **Bursitis** (prepatellar, pes anserine, MCL bursa, deep infrapatellar): fluid-distended bursa, bright on T2 FS; axial often best (PMC3354353).
- **Osteochondral lesions / osteochondritis dissecans (OCD):** subchondral bone fragment ± overlying cartilage injury, classically the lateral aspect of the medial femoral condyle; stability assessment guides surgery.
- **Loose bodies / intra-articular bodies:** free fragments of bone/cartilage in the joint — a cause of locking; part of "internal derangement."
- **Plica syndrome:** thickened synovial fold (esp. medial patellar plica) causing impingement.
- **Fractures (occult):** tibial plateau, patella; low T1 line + surrounding edema; Segond fracture = lateral capsular avulsion nearly pathognomonic for ACL tear.

---

## 4. How a Musculoskeletal Radiologist Reads a Knee MRI

### 4.1 "Internal derangement" — the umbrella term in the indication line

Most knee MRIs are ordered for "internal derangement of the knee" (IDK) — shorthand for suspected mechanical/structural problems inside the joint.

> **Claim:** Definition of internal derangement of the knee.
> **Source:** PMC11214712, "MRI of Internal Derangements and Other Knee Pathologies in Adult Nigerians" (2024), quoting the standard definition
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11214712/ | Date: 2024
> **Excerpt (verbatim):** "Internal derangement of the knee (IDK) is an inclusive term used to indicate (alone or in combination) certain disorders of the joint, including (alone or in combination) torn meniscus, loose bodies in the knee, and damaged collateral or cruciate ligaments. The term does not signify chronic disorders such as chondromalacia patellae, congenital discoid meniscus, meniscal cysts, or degenerative processes such as knee osteoarthritis."
> **Confidence:** High

MRI is the non-invasive test of choice for IDK and has largely replaced diagnostic arthroscopy (PMC11214712: "MRI is now the non-invasive imaging method of choice for virtually all clinical indications involving the knee."; Confidence: High).

### 4.2 The systematic search pattern

Radiologists do not free-browse; they sweep a fixed checklist so nothing is missed. A representative expert workflow (synthesized from two sources below):

1. **Technical check:** motion artifact, fat-suppression homogeneity, correct planes/slice thickness — a failed FS sequence can fake or hide edema.
2. **Fluid first:** effusion volume/distribution (suprapatellar pouch, recesses), Baker cyst, bursal fluid.
3. **Bone marrow:** edema patterns (their map reveals mechanism — pivot shift vs dashboard vs clip), occult fracture lines, marrow-replacing lesions.
4. **Ligaments:** ACL, PCL (sagittal), MCL, LCL/posterolateral corner (coronal) — continuity, caliber, signal, periligamentous edema; secondary signs (anterior tibial translation, PCL bowing, bone bruise pattern).
5. **Menisci:** horns and bodies on sagittal + coronal; surface-reaching signal, morphology, displaced fragments, roots, extrusion.
6. **Cartilage:** all six surfaces (patella, trochlea, medial/lateral femoral condyles, medial/lateral tibial plateaus) — signal, thinning, defects, delamination.
7. **Extensor mechanism & tendons:** quadriceps, patellar tendon, popliteus, pes anserine, semimembranosus.
8. **Synovium & fat pads:** synovitis, plicae, Hoffa's fat pad.
9. **Periphery:** muscles, neurovascular bundle, soft tissues, incidental masses ("don't-forget" areas).

> **Claim:** Systematic reading order (quality → marrow → menisci → ligaments → cartilage → effusion).
> **Source:** Moolchand Healthcare blog, "Knee Anatomy MRI Radiology Explained with Images" (practicing radiologist author; use as pattern evidence, not authority)
> **URL:** https://blog.moolchandhealthcare.com/knee-anatomy-mri-radiology-explained-with-images-a/ | Date: 2026-06-15
> **Excerpt (verbatim):** "I assess field homogeneity, motion, and slice thickness first. … I scroll for marrow oedema patterns under weight-bearing cartilage. … I then look for a line that reaches an articular surface on two planes. … I trace every fibre bundle in two planes. Waviness, discontinuity, and oedema are weighed together. … I assess thickness, surface contour, and the opposing surface. … I describe volume, distribution, and synovial features."
> **Confidence:** Medium (non-peer-reviewed but consistent with standard teaching)

> **Claim:** Corroborating checklist (bones → ligaments → menisci → cartilage → tendons/muscles → soft tissues → report).
> **Source:** Chinmay Gupte (orthopedic surgeon), "How do you read a knee MRI"
> **URL:** https://www.chinmaygupte.com/how-do-you-read-a-knee-mri | Date: n.d., accessed 2026-08-10
> **Excerpt (verbatim):** "A common approach is to assess bones, ligaments, menisci, cartilage, tendons, muscles, and surrounding soft tissues. … Summarize findings: After completing the image review, summarize the findings in a structured report that outlines any identified abnormalities and their potential clinical implications."
> **Confidence:** Medium

**ML translation:** the radiologist's checklist is essentially a multi-task classifier over anatomically indexed regions; each abnormality class has a characteristic (plane, sequence, region) where its signal-to-noise is highest. Models that pool all planes/series mimic the cross-confirmation step.

### 4.3 Report structure: Findings → Impression

Reports follow: **Clinical indication → Technique (sequences) → Findings (systematic body) → Impression (numbered, most-important-first diagnoses)**. Referring clinicians often read only the impression.

> **Claim:** Clinicians skip to the impression.
> **Source:** ScienceDirect/JACR, "Analysis of Different Levels of Structured Reporting in Knee MRI" (2020)
> **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1076633220300131 | Date: 2020-10-01
> **Excerpt (verbatim):** "A large percentage of respondents (47% overall, 44% of attendings, 60% of residents, and 35% of PAs) replied that they skipped the body and went straight to the impression section."
> **Confidence:** High

Structured knee reports itemize compartments — a useful schema for understanding label taxonomies (verbatim checklist excerpt):

> **Claim:** Structured knee MRI report template (fields).
> **Source:** MusculoskeletalKey, "The Knee" — "BOX 1: The Structured Report: Knee"
> **URL:** https://musculoskeletalkey.com/the-knee-9/ | Date: 2016-12-21
> **Excerpt (verbatim):** "FINDINGS: Fluid: [<Normal> <Small effusion> <Moderate effusion> <Large effusion> <Baker cyst (present / partially ruptured)> <Synovial hypertrophy> <Cartilaginous or osteochondral bodies>] … Medial meniscus: [<Normal> <Degenerative free edge fraying> <Incomplete or complete radial tear/Oblique tear/Horizontal tear/Longitudinal tear/Flap tear/Displaced bucket-handle tear>] Medial collateral ligament: [<Normal> <Thickened> <Acute sprain>] Medial femoral condyle cartilage … Lateral compartment … Posteromedial corner …"
> **Confidence:** High

**ML translation:** in multilingual report data, expect (a) the same Findings/Impression skeleton across the ~12 languages and 16 institutions, (b) labels of varying granularity (structure × side × severity), and (c) impression-section sentences to be the highest-yield text for weak supervision. A normal-study report (Radiopaedia case rID-147131) reads: "The anterior cruciate ligament (ACL) and posterior cruciate ligament (PCL) are intact with normal signal intensity. … Both medial and lateral menisci are normal. … No evidence of fracture, dislocation or significant joint effusion. No marrow edema. … Impression: Normal left knee MRI." (Confidence: High — Radiopaedia normal case accessed 2025-10-30.)

### 4.4 Inter-reader variability (why labels are noisy)

Radiologists disagree, especially on cartilage and subtle ligament findings. Agreement is measured with **Cohen's kappa** (chance-corrected; Landis & Koch 1977 scale):

| Kappa | Agreement |
|---|---|
| <0 | Poor |
| 0.0–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.0 | Almost perfect |

(Source: NCBI Bookshelf, "Interpretation of Fleiss' kappa (from Landis and Koch 1977)," https://www.ncbi.nlm.nih.gov/books/NBK92287/table/executivesummary.t2/ — verbatim table. Confidence: High.)

Representative kappa values for knee MRI:

> **Claim:** Two-reader agreement on a standard 1.5T knee protocol: medial meniscus κ=0.91, lateral meniscus κ=0.89, ACL κ=0.98, cartilage κ=0.84.
> **Source:** PMC4651376 (Br J Radiol 2015)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4651376/ | Date: 2015
> **Excerpt (verbatim):** "Agreement among the readers was for the standard protocol: MM kappa = 0.91, LM = 0.89, ACL = 0.98 and cartilage = 0.84; and for the 3D protocol: MM = 0.86, LM = 0.77, ACL = 0.94 and cartilage = 0.64."
> **Confidence:** High

> **Claim:** MRI-vs-arthroscopy reliability (a proxy for ground-truth noise): ACL κ=0.75 (strong), medial meniscus κ=0.60 (moderate), lateral meniscus κ=0.35 (weak), cartilage lesions κ=0.03–0.32 (none–weak), plica κ=0.01 (none).
> **Source:** ABC Research journal PDF (2026 issue), "Reliability of MRI vs arthroscopy" table
> **URL:** https://abcresearch.net/pdf/0fdb9ffe-e838-45c0-b564-25a52c51df96/issues/2026-008-001.pdf | Date: 2026
> **Excerpt (verbatim):** "Anterior cruciate ligament 0.75 … Strong; Medial meniscus 0.60 … Moderate; Lateral meniscus 0.35 … Weak; … Lateral femoral condyle cartilage lesion 0.03 … None; … Plica existence 0.01 … None."
> **Confidence:** Medium (journal of uncertain indexing; values consistent with literature)

> **Claim:** Acute ACL injury reading agreement between two radiologists κ=0.89–0.93; ALL tear detection varies with injury-to-MRI interval (κ 0.86 within 1 month vs 0.62 at 1–2 months).
> **Source:** Springer, "Timing of MRI affects the accuracy and interobserver agreement of anterolateral ligament tears detection in ACL deficient knees" (2020)
> **URL:** https://link.springer.com/article/10.1186/s43019-020-00082-z | Date: 2020-11-27
> **Excerpt (verbatim):** "Park also reported K values of 0.89–0.93 for agreement between two radiologists in assessing acute anterior cruciate ligament (ACL) injury in the knee. … In the first group in which MRI scans were performed within 1 month of injury, the ALL tear was identified by the radiologist in 92% of patients and by the surgeon in 90% of patients (Κ = 0.86). In the second group … (K = 0.62)."
> **Confidence:** High

> **Claim:** MOAKS (OA whole-joint score) reliability: mostly κ 0.61–1.0, with weak spots — tibial cartilage area κ=0.36, tibial osteophytes κ=0.49, Hoffa-synovitis intra-rater κ=0.42.
> **Source:** Hunter DJ et al., "Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS," Osteoarthritis Cartilage 2011 (PMC4058435)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/ | Date: 2011
> **Excerpt (verbatim):** "With the exception of inter-rater reliability for tibial cartilage area (kappa=0.36) and tibial osteophytes (kappa=0.49); and intra-rater reliability for tibial BML number of lesions (kappa=0.54), Hoffa-synovitis (kappa=0.42) all measures of reliability using kappa statistics were very good (0.61-0.8) or reached near perfect agreement (0.81-1.0)."
> **Confidence:** High

> **Claim:** Meniscal extrusion reading agreement κ=0.61.
> **Source:** PMC6219866 (Cureus 2018)
> **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6219866/ | Date: 2018
> **Excerpt (verbatim):** "Kappa test for inter-reader agreement = 0.61."
> **Confidence:** High

**ML translation:** expect label noise to be lowest for ACL tears and gross meniscal tears, and highest for cartilage grading, synovitis grading, lateral meniscus, and plica. This argues for soft labels / ordinal losses for graded findings and ensembling across institutions.

### 4.5 ML state of the art anchor (for calibration of expectations)

> **Claim:** Stanford MRNet (Bien et al. 2018, Nature Medicine): deep learning on 1,370 knee MRIs achieved AUC 0.965 (ACL), 0.847 (meniscus), 0.937 (abnormality); external-validation ACL accuracy ~82.4%, near-radiologist; model assistance increased radiologist sensitivity.
> **Source:** Springer J Supercomputing review (2025), summarizing Bien et al.
> **URL:** https://link.springer.com/content/pdf/10.1007/s11227-025-07103-2.pdf | Date: 2025
> **Excerpt (verbatim):** "In 2018, Bien et al. created MRNet, a deep learning model to analyze knee MRI data. The model was evaluated using 1,370 knee MRI scans from Stanford University to identify ACL injuries, meniscal tears, and general abnormalities. MRNet demonstrated exceptional performance with AUC values of 96.5% for ACL tears, 84.7% for meniscal tears, and 93.7% for overall abnormalities. In the external validation dataset, the model identified ACL tears with an accuracy of 82.4%, comparable to that of radiologists."
> **Confidence:** High (secondary summary of the primary Nature Medicine paper)

---

## 5. Terminology Glossary (45 terms you'll meet in reports and labels)

**Anatomy**
1. **Condyle** — rounded articular knob at bone ends (medial/lateral femoral condyles).
2. **Tibial plateau** — flat top of the tibia bearing the menisci.
3. **Intercondylar (tibial) spine / eminence** — central tibial bumps; cruciate and meniscal-root attachments.
4. **Intercondylar notch** — groove between femoral condyles housing the cruciate ligaments.
5. **Trochlea (trochlear groove)** — femoral groove the patella glides in; dysplasia predisposes to dislocation.
6. **Sesamoid** — bone embedded in a tendon (the patella is the largest).
7. **Meniscus (medial/lateral)** — C-shaped fibrocartilage shock absorber; parts: anterior horn, body, posterior horn, roots.
8. **Cruciate ligaments (ACL/PCL)** — crossing central ligaments limiting anterior/posterior tibial translation.
9. **Collateral ligaments (MCL/LCL)** — side stabilizers against valgus/varus stress.
10. **Posterolateral corner (PLC)** — LCL + popliteus + popliteofibular + arcuate complex; posterolateral stability.
11. **Extensor mechanism** — quadriceps tendon → patella → patellar tendon.
12. **Hoffa's (infrapatellar) fat pad** — fat cushion behind patellar tendon; impingement = Hoffitis.
13. **Bursa** — friction-reducing fluid sac (prepatellar, pes anserine, gastrocnemius–semimembranosus).
14. **Synovium / synovial fluid** — joint lining and its lubricant secretion.
15. **Hyaline (articular) cartilage** — smooth load-bearing coating of bone ends.
16. **Fibrocartilage** — tougher cartilage of the menisci.
17. **Plica** — embryologic synovial fold; medial patellar plica can impinge.
18. **Meniscofemoral ligaments (Humphrey/Wrisberg)** — normal variant ligaments near the PCL; can mimic tears.
19. **Red/white zone** — vascular outer vs avascular inner meniscus (healing potential).

**Pathology / report vocabulary**
20. **Internal derangement** — umbrella term for structural intra-articular knee pathology (tears, loose bodies).
21. **Sprain / partial tear / complete tear (rupture)** — ligament injury severity grades 1/2/3.
22. **Midsubstance tear vs avulsion** — tear within the ligament vs pulling off its bony attachment.
23. **Tear orientations** — horizontal (cleavage), longitudinal vertical, radial, oblique/parrot-beak, complex, root tear.
24. **Bucket-handle tear** — displaced longitudinal meniscal tear into the notch; may lock the knee.
25. **Meniscal extrusion** — meniscus bulging ≥3 mm past tibial plateau margin; marker of root tear/OA.
26. **Parameniscal cyst** — fluid cyst at meniscal margin, implies underlying (usually horizontal) tear.
27. **Discoid meniscus** — congenital disc-shaped (over-wide) meniscus, prone to tearing; usually lateral.
28. **Chondromalacia** — cartilage softening/degeneration, graded (modified) Outerbridge I–IV.
29. **Delamination** — cartilage peeling away from bone along the cartilage–bone interface.
30. **Osteochondral lesion / OCD** — lesion of cartilage + underlying bone; osteochondritis dissecans may fragment.
31. **Loose (intra-articular) body** — free cartilage/bone fragment in the joint ("joint mouse").
32. **Bone marrow lesion (BML) / bone marrow edema(-like) signal** — ill-defined high signal on fluid-sensitive sequences; traumatic (bruise) or degenerative.
33. **Bone bruise / contusion / kissing contusion** — impaction microtrabecular injury; pattern maps mechanism.
34. **Effusion** — excess joint fluid (small/moderate/large); **hemarthrosis** = blood; **lipohemarthrosis** = fat+blood (fracture sign).
35. **Synovitis** — synovial inflammation/thickening; enhances post-contrast; **Hoffa-synovitis** = in fat pad.
36. **Baker (popliteal) cyst** — fluid in gastrocnemius–semimembranosus bursa behind the knee; may dissect/rupture (mimics DVT).
37. **Bursitis** — inflamed, fluid-distended bursa (e.g., prepatellar "housemaid's knee", pes anserine).
38. **Tendinopathy / tendinosis** — degenerative tendon thickening + signal change (e.g., jumper's knee); vs **tear**.
39. **Osteophyte** — marginal bone spur of OA.
40. **Subchondral sclerosis / cyst (geode)** — hardened bone / fluid cavity beneath damaged cartilage.
41. **Joint space narrowing (JSN)** — radiographic proxy for cartilage loss (asymmetric in OA, medial > lateral).
42. **Segond fracture** — lateral tibial capsular avulsion fracture; strongly associated with ACL tear.
43. **O'Donoghue (unhappy) triad** — ACL + MCL + medial meniscus injury combination.
44. **Pivot shift / dashboard / clip / hyperextension** — named injury mechanisms with characteristic bone-bruise maps.
45. **Kellgren–Lawrence (KL) grade 0–4** — radiographic OA severity scale; WORMS/BLOKS/MOAKS = MRI whole-organ OA scores.

**MRI/technical**
46. **T1W / T2W / PD** — weightings: anatomy / fluid-bright / high-detail intermediate contrast.
47. **Fat suppression (FS/sat), STIR** — makes fat dark so fluid/edema "lights up."
48. **Sagittal / coronal / axial** — the three orthogonal imaging planes.
49. **Signal hyperintense/hypointense** — brighter/darker than reference tissue ("high signal in the ACL" etc.).
50. **TR/TE, FOV, slice thickness, FSE/TSE, GRE** — acquisition parameters/sequence types shaping contrast and resolution.

---

## 6. Best Educational Resources (curated)

**Radiopaedia.org (free, peer-reviewed reference articles + cases):**
- "Knee joint" — anatomy overview — https://radiopaedia.org/articles/knee-joint-1
- "Anterior cruciate ligament tear" — primary/secondary MRI signs — https://radiopaedia.org/articles/anterior-cruciate-ligament-tear
- "Medial collateral ligament injury (MRI grading)" — https://radiopaedia.org/articles/medial-collateral-ligament-injury-mri-grading
- "MRI grading system for abnormal meniscal signal intensity" — https://radiopaedia.org/articles/mri-grading-system-for-abnormal-meniscal-signal-intensity
- "Baker cyst" — https://radiopaedia.org/articles/baker-cyst-2
- "Patellar tendinopathy" — https://radiopaedia.org/articles/patellar-tendinopathy
- "Normal MRI knee" case for reference appearances — https://radiopaedia.org/cases/normal-mri-knee-1

**Structured review articles (peer-reviewed, open access):**
- Chien A, Weaver JS, et al. "Magnetic resonance imaging of the knee." Pol J Radiol 2020 — best single primer: protocol table, meniscus/ligament/cartilage sections, PPV rules. https://pmc.ncbi.nlm.nih.gov/articles/PMC7571514/
- Chana-Rodríguez F, et al. "Reporting knee meniscal tears: technical aspects, typical pitfalls and how to avoid them." Insights Imaging 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4877346/
- "How I Diagnose Meniscal Tears on Knee MRI." AJR 2012 — tear-by-tear pictorial. https://ajronline.org/doi/10.2214/AJR.12.8663
- Mohankumar R, et al. "Pitfalls and Pearls in MRI of the Knee." AJR 2014 — normal variants that fake tears. https://ajronline.org/doi/10.2214/AJR.14.12969
- "Joint effusion of the knee: potentialities and limitations…" Insights Imaging 2015 — recess anatomy, effusion. https://pmc.ncbi.nlm.nih.gov/articles/PMC4630268/
- Burke CJ, et al. "MRI of Synovitis and Joint Fluid." 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6504589/
- Kohn MD, et al. "Classifications in Brief: Kellgren-Lawrence." Clin Orthop Relat Res 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC4925407/
- Hunter DJ, et al. "MOAKS." Osteoarthritis Cartilage 2011. https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/
- Roemer FW, et al. "MRI-detected subchondral bone marrow signal alterations of the knee joint: terminology, imaging appearance, relevance and radiological differential diagnosis." Osteoarthritis Cartilage 2009 (citation captured via Nat Rev Rheumatol 2023 reference list, https://www.nature.com/articles/s41584-023-00971-z).
- "Clinical and Pathophysiologic Significance of MRI Identified BMLs Associated with Knee OA." 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6578476/
- EPOS/ESR posters: "Bone Contusion Patterns of the Knee at MRI" (ESSR 2019, P-0162); "Synovial knee disease: MRI differential diagnosis" (ECR 2013, C-2443); "Acute trauma of the knee ligaments: Following the contusion pattern" (ECR 2016, C-1617).

**Teaching-file sites:** Radsource MRI Web Clinic (radsource.us — MCL, knee OA, protocols); Radiology Assistant (radiologyassistant.nl — "Meniscal pathology" knee module); Radiology Masterclass (radiologymasterclass.co.uk — OA radiography); AuntMinnie.com (news + education channels; account needed).

**ML context:** Bien N, et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging (MRNet)." Nature Medicine 2018; Stanford MRNet dataset (1,370 exams; abnormal/ACL/meniscus labels in axial/coronal/sagittal) — the methodological ancestor of this competition.

---

## Appendix: Search Log (independent searches executed 2026-08-10)

1. Radiopaedia knee anatomy MRI ligaments menisci overview
2. Radiopaedia ACL tear MRI appearance
3. Radiopaedia meniscal tear MRI classification
4. Radiopaedia bone marrow lesion MRI knee
5. MRI pulse sequences T1 T2 PD fat-suppressed fluid bright/dark
6. Kellgren-Lawrence osteoarthritis grading radiographic
7. Knee MRI planes which structures best seen
8. Knee joint effusion MRI radiopaedia
9. Baker cyst popliteal cyst MRI radiopaedia
10. Synovitis MRI knee radiopaedia
11. Bone marrow edema MRI knee OA BML review (Roemer/Guermazi)
12. Articular cartilage injury grading Outerbridge MRI
13. MCL injury grading MRI radiopaedia
14. How to read knee MRI systematic search pattern
15. Interobserver agreement knee MRI kappa ACL meniscus
16. Radiology report structure findings impression knee MRI
17. Knee MRI protocol sagittal coronal axial PD FS standard
18. PCL tear MRI radiopaedia
19. Osteoarthritis knee MRI features osteophytes sclerosis
20. Knee joint effusion suprapatellar recess MRI
21. Menisci anatomy anterior/posterior horn radiopaedia
22. Fat suppression STIR vs chemical fat-sat MSK MRI
23. Kappa reliability MOAKS/WORMS meniscus cartilage
24. Best resources to learn knee MRI interpretation
25. T1/T2 signal characteristics table tissues
26. Which plane for collaterals/cruciates/patellofemoral cartilage
27. Bone bruise/contusion patterns knee MRI
28. RSNA 2026 knee abnormality detection competition
29. Chondromalacia patellae MRI grading
30. Patellar tendinopathy jumper's knee MRI
31. Meniscal extrusion MRI significance
32. Landis-Koch kappa interpretation
33. Knee bursae/bursitis MRI
34. WORMS whole-organ MRI score knee OA
35. Osteochondritis dissecans knee MRI
36. Internal derangement knee definition
37. MRNet Stanford knee MRI deep learning
38. MRI physics how it works (protons/RF/gradients)

Plus direct full-text opens: Radiopaedia ACL tear article; PMC7571514 (Chien 2020); RuntimeWire competition article; Radiopaedia bone-contusion article (blocked); Radiopaedia meniscal-tear article (bot-blocked; substitute sources used); Radiology Assistant (access-rejected).
