# Hysteresis-aware simulation and vision-guided remount recovery for atomic force microscopy site repositioning

## Abstract

Precise repositioning in atomic force microscopy (AFM) remains difficult when a sample is removed and remounted, because nanoscale targeting is affected by scanner hysteresis, creep, view rotation, and repeated local patterns. Existing workflows often depend on manual operator recall, repeated trial-and-error scanning, or isolated correction modules that do not connect scanner behavior with image-based verification. Here, we present a simulation-centered AFM platform that integrates two coupled capabilities: a Prandtl-Ishlinskii (PI)-based piezo-stage motion model with creep, and a coarse-to-fine vision pipeline for recovering previously saved scan sites after remounting. The platform stores structured site memories composed of low-magnification overviews, high-magnification reference templates, and multi-scale landmarks, and then combines affine remount estimation, normalized cross-correlation (NCC) matching, landmark consensus, and optional deep-feature-based verification for site recovery. In the current implementation, the simulator maintains 26 saved site memories across two sample image sources, with each memory containing 8 low-magnification landmarks and 6 high-magnification landmarks. A saved hysteresis characterization dataset comprising 1,440 motion samples across six sweep conditions shows a maximum absolute positioning deviation of 4.13 um in the present PI configuration. These results establish the platform as a practical foundation for studying AFM repositioning under nonideal scanner behavior while also defining a reproducible framework for future quantitative evaluation on larger remount datasets. The present manuscript reports the system design, current implementation status, and a figure-led experimental blueprint for subsequent real-data validation.

**Keywords:** atomic force microscopy, hysteresis simulation, remount recovery, image registration, site repositioning, vision-guided relocation

## 1. Introduction

Atomic force microscopy is widely used for nanoscale topography, defect analysis, and surface characterization. In many practical workflows, however, the most challenging step is not the initial scan but the return to the same physical site after the sample has been removed, reinstalled, or repositioned. Even when the user has prior knowledge of the target region, exact revisit remains difficult because the relation between commanded and realized stage motion is not ideal, the observed image may rotate after remounting, and locally repeated textures can produce ambiguous matches.

This problem becomes more severe when the scanner exhibits hysteresis and creep. In piezo-actuated stages, the commanded displacement is not perfectly equal to the realized displacement, and the mismatch depends on motion history. As a result, replaying a previously used coordinate is often insufficient for true site recovery. In addition, remounting can introduce translation, in-plane rotation, and slight appearance change, so that a single stored reference patch is often too fragile for robust relocation.

Several components relevant to this problem are already well known in isolation. Hysteresis modeling has been studied extensively for piezo-driven positioning systems, and image-based registration is widely used for template matching and visual alignment. Deep learning has also been applied to feature extraction and pattern recognition in microscopy. However, a practical AFM repositioning workflow requires these elements to operate together: the system must remember where a site was, compensate for nonideal stage behavior, identify the approximate remount transform, refine the recovered location, and verify that the predicted site is indeed the same physical region.

The present work addresses this need with a unified simulation and relocation environment. Rather than treating hysteresis compensation and sample recall as separate research tasks, we combine them in a single platform that supports interactive AFM-like navigation, site-memory storage, remount simulation, and closed-loop recovery. The design follows a classical-first, AI-augmented philosophy. Deterministic methods such as affine estimation, NCC template matching, and landmark geometry remain the primary recovery mechanisms, while machine learning is used selectively for transform estimation, embedding-based retrieval, and same-site verification.

This manuscript reports the current version of the platform and its research framing. The main contributions are as follows. First, we establish a PI-based AFM stage simulator with an explicit creep component for realistic command-to-motion mismatch. Second, we implement a structured multi-scale site-memory representation consisting of overview images, reference templates, and low- and high-magnification landmarks. Third, we develop a coarse-to-fine relocation strategy that combines affine remount estimation, landmark consensus, fine template matching, and verification logic. Fourth, we provide initial ML hooks based on ResNet18 features and shallow regressors/classifiers without replacing the explainable classical baseline. Finally, we summarize the current evidence already available in the repository and define the next quantitative experiments required to convert the platform into a complete AFM repositioning study.

## 2. System overview

The platform is organized around two interacting subsystems: motion simulation and visual site recovery. The motion subsystem models the AFM scanner as a stateful XY piezo-stage with hysteresis and creep. The visual subsystem renders the corresponding AFM view, stores selected regions as site memories, simulates remount-induced changes, and attempts to relocate the original site.

At the workflow level, the user first navigates to a region of interest and saves a site memory. The site memory includes the current reference template, low-magnification overview, high-magnification local appearance, landmark patches, tip-relative geometry, and metadata such as zoom level, field of view, tilt, and motion history. After a simulated remount event, the recovery logic estimates a coarse transform between the stored and current views, moves to the predicted region, performs fine alignment, and accepts the result only if verification thresholds are satisfied.

This design is intentionally hybrid. The repositioning logic is not delegated entirely to an AI model. Instead, classical image-matching methods provide the main localization signal, while the ML components act as optional support modules that can narrow the search space or provide a second opinion when confidence is borderline.

## 3. Materials and methods

### 3.1 Software environment

The simulator is implemented in Python and uses `numpy`, `matplotlib`, `opencv-python`, `joblib`, `pandas`, `scikit-learn`, `torch`, and `torchvision` through project modules such as `afm_control_panel.py`, `afm_callbacks.py`, `afm_relocation.py`, `afm_ml_recognition.py`, and `hysteresis.py`. The interactive interface is launched from `afm_control_panel.py`.

### 3.2 PI-based hysteresis and creep model

The stage model in `hysteresis.py` uses a multi-operator Prandtl-Ishlinskii formulation. For a commanded input sequence, each play operator contributes a thresholded state response, and the weighted sum of these operators defines the base hysteresis output. The implementation maintains independent state for the X and Y axes and therefore preserves path dependence during interactive movement.

To reflect short-term post-motion relaxation, an additional creep term is included. In the current configuration, the creep model uses a gain of 0.18, a per-frame decay of 0.14, and a nonlinearity scale of 40.0 um. The realized position is computed as the sum of the PI base response and the evolving creep residual. Motion history is logged to `movement_log.csv`, and stage trajectories can be visualized directly from the simulator.

The repository also contains a saved hysteresis characterization file, `collected_data/hysteresis_data_20260526_115443.csv`, comprising 1,440 motion samples across six sweep conditions: 400, 800, and 1,200 um ranges, each evaluated at slow and fast settings. In this dataset, the maximum absolute command-tracking deviation is 4.13 um and the mean absolute deviation is 4.06 um under the present PI configuration.

### 3.3 Site-memory construction

When the user saves a site, the system creates a structured memory record in `collected_data/site_memories/`. Each record includes:

- a low-magnification overview image,
- a high-magnification reference template,
- an optional origin template,
- 8 low-magnification landmarks,
- 6 high-magnification landmarks,
- tip-relative landmark geometry,
- motion-history metadata,
- field-of-view information,
- zoom, magnification, focus, tilt, sample ID, and session ID.

At the time of writing, the repository contains 26 saved site memories spanning two sample image sources. The landmark counts are consistent across the current records, indicating that the capture routine is already standardized at the implementation level.

### 3.4 Low-magnification overview and landmark extraction

The relocation system constructs a reduced overview representation using `build_overview()` and extracts candidate landmarks using `extract_landmarks()`. The current implementation limits the low-magnification set to 8 landmarks and the high-magnification set to 6 landmarks. Landmark patches are annotated with absolute coordinates and, when available, with tip-relative distance and angle. This geometry is later reused during relocation verification.

### 3.5 Coarse remount estimation

The first relocation stage estimates the coarse transform between the saved and current sample states. Two main mechanisms are available. The first is affine estimation from image correspondences using `estimate_affine_transform()`. The second is landmark consensus from low-magnification patches using `estimate_landmark_consensus()`. If an affine solution is sufficiently supported, it is converted into the full-resolution coordinate frame and used to initialize the fine-search region. Otherwise, the system falls back to landmark-guided coarse localization.

This design is important because remounting can induce both translation and in-plane rotation. A pure local template match is often brittle under those conditions. By estimating the coarse remount transform first, the system reduces the search ambiguity before attempting high-resolution site recovery.

### 3.6 Fine registration and verification

After coarse localization, the system performs fine alignment using NCC template matching and local affine checks. Verification is then performed using four complementary signals:

1. reference-template match score and score gap,
2. high-magnification landmark consensus,
3. landmark-geometry consistency,
4. optional same-site ML probability.

Acceptance requires agreement across the deterministic criteria unless the implementation is explicitly configured to let the ML verifier override a borderline classical failure. This conservative acceptance logic is intended to reduce false-positive relocation in samples that contain repeated structures.

### 3.7 Machine learning modules

The AI layer uses a pretrained ResNet18 as a fixed feature extractor that produces 512-dimensional normalized image descriptors. These descriptors are then used in three ways.

First, `MLPatternMatcher` applies sliding-window deep-feature matching as an alternative scoring mechanism. Second, an MLP regressor can be trained to predict remount transforms from concatenated deep pair features. Third, an MLP classifier can be trained to assess whether two views belong to the same site.

Two training directions are implemented in the repository. The synthetic branch in `train_remount_5w.py` generates 50,000 warped examples per anchor template, with rotation, translation, scale, and brightness variation applied in template pixel space. The real-pair branch in `train_remount_real.py` mines frame pairs from the saved site-memory corpus and uses affine estimation as a pseudo-ground-truth label source. Based on the current site-memory set, the synthetic branch can draw from 25 unique anchor templates after deduplication.

## 4. Results

### 4.1 The platform couples scanner nonideality with visual site recovery

The first outcome of this work is not a standalone predictor but an integrated environment in which motion nonideality and visual recovery can be studied together. The simulator does not assume an ideal relation between command and realized position. Instead, it explicitly injects hysteresis and creep, then asks whether the original site can still be found after remounting using a structured memory and visual verification pipeline. This is the core conceptual contribution of the current platform.

**Suggested Figure 1.** Overall workflow of the platform, including site saving, remount simulation, coarse transform estimation, fine relocation, and verification.

### 4.2 The current PI configuration produces a measurable command-to-position mismatch

The saved hysteresis characterization data show that the present stage model generates a visible and quantifiable offset between commanded and realized motion. Across 1,440 recorded samples and six sweep conditions, the maximum absolute tracking deviation reaches 4.13 um, with a mean absolute deviation of 4.06 um. These values confirm that the simulator is not effectively operating as an ideal scanner; rather, it creates a sufficiently strong motion mismatch to serve as a meaningful testbed for repositioning studies.

Although the current saved file reports one-dimensional characterization along the X command axis, the implementation itself is stateful in both X and Y and therefore supports two-dimensional repositioning experiments. A future quantitative study should extend the exported characterization to bidirectional raster scans and recovery trajectories after closed-loop correction.

**Suggested Figure 2.** Representative PI hysteresis loops and creep-affected trajectories for different command ranges and sweep rates.

### 4.3 Structured site memories enable multi-scale relocation logic

The second practical result is the establishment of a standardized site-memory structure for remount recovery. The current repository contains 26 site memories distributed across two sample sources. Each memory includes a low-magnification overview, a high-magnification reference template, and fixed-size sets of low- and high-magnification landmarks. This is a stronger representation than a single stored patch because it supports both coarse global search and local geometric verification.

Visual inspection of the stored records shows that the overview image captures contextual sample appearance, while the reference template and high-magnification landmarks preserve local structural details. In the current implementation, this representation is sufficient to support coarse affine estimation, fine NCC registration, and landmark-geometry checks within a single recovery pipeline.

**Suggested Figure 3.** Example site memory showing overview image, reference template, low-magnification landmarks, and high-magnification landmark patches.

### 4.4 The relocation pipeline is explainable and staged rather than monolithic

The relocation logic encoded in `afm_callbacks.py` and `afm_relocation.py` proceeds through a clearly interpretable sequence: coarse affine remount estimation, low-magnification landmark fallback, high-magnification template recovery, and final verification. This staged design is preferable to a monolithic black-box predictor for two reasons. First, each stage provides intermediate evidence that can be inspected when recovery fails. Second, the classical baseline remains usable even if the trained models are absent, mismatched, or unreliable.

An important implementation detail is that the system does not rely only on the best raw template-match score. It also evaluates support count, confidence, affine inliers, and geometry consistency. This is particularly important for AFM-like textures, where repeated patterns can otherwise produce visually plausible but incorrect matches.

**Suggested Figure 4.** Coarse-to-fine relocation sequence showing remounted view, coarse transform estimate, fine search window, recovered template match, and pass/fail verification indicators.

### 4.5 The machine learning layer is currently auxiliary rather than definitive

The present ML components are best viewed as augmentation modules, not as the final scientific claim of the project. The ResNet18-based feature extractor, MLP remount regressor, embedding index, and same-site classifier are already integrated at the code level, and the data pipeline supports both synthetic transform generation and real-pair harvesting from stored site memories. However, the current repository does not yet contain a formal evaluation table demonstrating statistically robust gains over the classical baseline on held-out real remount cases.

This is an important limitation and should be stated directly. The current manuscript therefore presents the ML layer as a prepared experimental branch whose main roles are rotation hinting, retrieval support, and same-site verification. A complete follow-up study should compare at least three modes: classical-only, classical plus ML hinting, and ML-dominant recovery. Metrics should include recovery success rate, translation error, rotation error, verification precision, and failure behavior under repeated patterns.

**Suggested Figure 5.** ML module diagram with ResNet18 feature extraction, pair-feature construction, remount regression, same-site classification, and dataset generation from stored anchors.

## 5. Discussion

The current platform is useful because it converts AFM repositioning from an informal manual task into a structured computational problem. It represents a site with context rather than with a single patch, models scanner nonideality explicitly rather than assuming perfect motion, and separates recovery from verification so that false confidence is less likely. These design choices are aligned with the practical reality of AFM revisit experiments, where operator memory alone is often insufficient and repeated local patterns can make visual matching ambiguous.

At the same time, the present work should be interpreted as a platform paper and an initial methods manuscript rather than a finished performance paper. The available evidence already supports the feasibility of the simulator architecture, the hysteresis-aware motion model, and the multi-stage recovery logic. However, several elements still require systematic validation. The current site-memory corpus is small, being limited to 26 saved records across two source images. The hysteresis characterization file demonstrates nonideal motion, but a larger suite of two-dimensional recovery benchmarks is still needed. Likewise, the ML branch is implemented and trainable, yet it has not been benchmarked rigorously against a fully defined remount test set.

Future experiments should therefore address four questions. First, how accurately can the system recover a site after controlled translation and rotation perturbations of increasing magnitude? Second, how robust is the verification logic when the sample contains repeated motifs or low contrast? Third, how much does the ML layer improve recovery speed or success rate relative to the classical baseline? Fourth, how well do results transfer from simulated remounts to real AFM revisit data?

The platform is already well positioned for those studies because the codebase stores the required metadata, supports repeated save-and-recover trials, and preserves motion history and site-memory records on disk. The most immediate next step is not architectural redesign but disciplined evaluation.

## 6. Conclusion

We have presented a hysteresis-aware AFM simulation platform that combines PI-based stage modeling, creep-aware motion mismatch, structured multi-scale site memory, and a coarse-to-fine visual remount-recovery pipeline. The current implementation stores 26 site memories with standardized landmark content and includes a saved hysteresis dataset showing up to 4.13 um command-tracking deviation under the present PI configuration. The system also integrates optional ResNet18-based ML modules for transform prediction and same-site verification while preserving an explainable classical baseline.

The main value of the present work is that it establishes a coherent framework for studying AFM site repositioning under realistic scanner nonideality. The next phase should focus on quantitative benchmarking, controlled remount experiments, and figure-driven comparison of classical and ML-augmented recovery modes. With those additions, the platform can evolve from a strong simulation and methods contribution into a full repositioning manuscript with experimentally supported performance claims.

## Figure plan for the next revision

### Figure 1

Overall workflow:

- save site memory
- remount perturbation
- coarse transform recovery
- fine relocation
- verification

### Figure 2

Hysteresis characterization:

- ideal vs actual trajectory
- forward/backward loop
- creep relaxation after step motion
- error versus command displacement

### Figure 3

Site-memory structure:

- low-mag overview
- reference template
- low-mag landmark grid
- high-mag landmark grid

### Figure 4

Relocation example:

- saved reference
- remounted input
- coarse affine recovery
- fine matched output
- overlay of recovered vs target region

### Figure 5

Quantitative recovery evaluation:

- translation error
- rotation error
- success rate within tolerance
- verification pass/fail matrix

### Figure 6

ML branch:

- ResNet18 feature extractor
- pair-feature design
- synthetic and real-pair training paths
- same-site classifier outputs

## Data and claim status

The following statements are directly supported by the current repository state:

- the system implements a PI hysteresis model with creep,
- the relocation pipeline is coarse-to-fine and hybrid classical/ML,
- 26 saved site memories are present,
- each current memory contains 8 low-mag and 6 high-mag landmarks,
- 25 unique reference anchors are available after deduplication,
- the saved hysteresis dataset contains 1,440 samples across six sweep conditions,
- the maximum absolute error in that file is 4.13 um.

The following items still require formal experiment tables before journal submission:

- remount recovery success rate,
- mean and median relocation error,
- comparison between classical and ML-assisted modes,
- robustness under repeated patterns,
- transfer from simulator-only data to real AFM revisit data.

## Candidate references to add in the final version

The final journal version should cite at least:

- AFM repositioning and revisit literature,
- Prandtl-Ishlinskii hysteresis modeling papers,
- piezo-stage creep compensation papers,
- microscopy or AFM image-registration papers,
- NCC / affine registration baselines,
- ResNet and feature-matching references,
- any directly related AFM remount or correlative imaging studies.

## Notes for the next writing pass

- Replace the placeholder figure descriptions with actual composite figures exported from the simulator.
- Add a formal experiment section once remount trials have been executed and logged.
- Convert all `um` notation to the exact journal-preferred symbol handling during final formatting.
- If a target journal requires `Results` before `Methods`, move Sections 3 and 4 accordingly without changing the argument sequence.
