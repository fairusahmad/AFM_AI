# AI Repositioning Plan for AFM Sample Recovery

## Goal

Build a repositioning workflow that helps the user:

1. recognize the sample at low magnification,
2. move to a scan location and define a meaningful local origin,
3. remove and place the sample back,
4. recover the previous scan position automatically with AI and image matching,
5. verify that the recovered position is truly the same physical site.


## High-Level Idea

The system should use a coarse-to-fine strategy:

1. Low magnification for coarse localization on the sample.
2. Mid magnification for regional refinement.
3. High magnification for exact scan-site recovery.

This should not rely on only one memorized patch. It should use:

- a global sample view,
- multiple landmarks,
- a named local origin,
- high-magnification reference patterns,
- confidence scoring before motion and before final confirmation.


## Intended User Workflow

### Phase 1: Before Removal

1. User views the sample at low magnification.
2. System captures a low-magnification overview image.
3. AI or image processing recognizes distinctive features and stores candidate landmarks.
4. User moves the sample stage to the region of interest.
5. System records the stage movement history during the search.
6. User zooms in to higher magnification and reaches the exact scan location.
7. User defines a new local origin at the scan site.
8. System stores:
   - the local origin,
   - the current scan position,
   - the current high-magnification image,
   - several nearby landmark patches,
   - the low-mag to high-mag relationship.

### Phase 2: Sample Removal

1. User lifts or removes the sample.
2. The system preserves all reference data from the previous session.

### Phase 3: After Replacement

1. User places the sample back on the stage.
2. User optionally gives a rough starting location.
3. System uses low-magnification recognition to estimate:
   - translation,
   - in-plane rotation,
   - possible tilt-related view change.
4. System moves near the predicted old region.
5. System refines position using stored landmarks and reference patches.
6. System moves to the predicted high-magnification scan location.
7. System verifies the final location by comparing the current image to the stored reference.
8. Only after passing the verification threshold does the system mark repositioning as successful.


## Coordinate System Design

The system should maintain at least two coordinate frames.

### 1. Global Coordinate Frame

Used for low-magnification navigation across the sample.

- stage X, Y
- sample overview landmarks
- coarse relocation

### 2. Local Coordinate Frame

Used for exact AFM scan work around the area of interest.

- named origin
- target scan coordinates relative to origin
- high-magnification landmarks
- local revisit points

### 3. Transform Between Frames

The system must estimate and store the transformation between sessions. At minimum:

- translation
- rotation

Potentially also:

- slight scale change,
- small affine or perspective distortion if mounting angle changes.


## What the Current Simulator Already Has

The current codebase already contains useful building blocks:

- named origin support,
- origin template capture,
- supervised origin search with template matching,
- reference image save and relocate flow,
- manual surface tilt entry,
- trajectory and stage-motion recording.

These are a good foundation, but not yet a complete AI repositioning system.


## Current Implementation Status

The simulator now includes a practical Phase 1 baseline aligned with this plan:

- structured site-memory save during reference capture,
- low-magnification overview extraction,
- multiple low-mag and high-mag landmark capture,
- site-memory reload from disk for later sessions,
- low-mag affine remount estimation,
- coarse-to-fine relocation flow,
- rotation-aware remount simulation,
- local affine fine registration,
- ambiguity-aware labeled-origin search using score-gap checks,
- relocation verification using reference confidence plus high-mag landmark support,
- iterative fine relocation passes before acceptance.

Phase 1 checklist status:

- multi-landmark save/load: implemented
- coarse template or keypoint matching: implemented
- affine transform estimation: implemented
- fine template matching: implemented
- confidence scoring: implemented
- iterative correction: implemented as repeated coarse/fine refinement inside relocation

Phase 2 has now started with initial AI augmentation scaffolding:

- trainable same-site / wrong-site classifier,
- trainable remount transform predictor,
- low-mag embedding index builder,
- runtime hooks that load these models when present.

Still missing from the full vision are:

- perspective or full projective remount estimation,
- learned landmark scoring dedicated to patch selection,
- stronger low-mag retrieval on real multi-sample data,
- robust real-dataset training and validation workflow,
- final closed-loop relocation using learned models as the primary decision layer.


## Missing Pieces in the Full Plan

### 1. Multi-Landmark Memory

Do not rely on only one saved patch.

Need to store:

- low-mag landmarks,
- high-mag landmarks,
- landmark class or descriptor,
- coordinates relative to local origin,
- patch uniqueness score.

### 2. Registration Across Sessions

After sample removal and re-placement, the system must estimate:

- XY shift,
- in-plane rotation,
- possible appearance change from tilt,
- confidence of the transform.

### 3. Confidence Scoring

The system should never move aggressively using a weak match.

Need thresholds for:

- coarse localization confidence,
- fine localization confidence,
- safe-to-scan confidence.

If confidence is too low, ask for user guidance instead of forcing relocation.

### 4. Ambiguity Rejection

Many samples have repeated patterns. The system must detect when two places look too similar.

Useful checks:

- top-1 vs top-2 match score gap,
- geometric consistency of multiple landmarks,
- residual alignment error after transform fitting.

### 5. Closed-Loop Correction

Relocation should not be one-shot.

Recommended loop:

1. estimate location,
2. move,
3. capture image,
4. compare again,
5. refine,
6. stop when residual error is below threshold.

### 6. AI-Based Tilt / Rotation Recognition

Manual tilt input exists now, but the future system should estimate at least:

- image-plane rotation,
- apparent tilt-related distortion,
- focus or blur changes caused by remount differences.


## Recommended Algorithm Strategy

Use a hybrid system, not AI alone.

### Coarse Localization

Good choices:

- classical keypoints and descriptors,
- image retrieval embeddings,
- landmark detector,
- ORB / SIFT-style matching if texture permits.

Output:

- approximate region,
- coarse transform estimate,
- confidence score.

### Fine Relocation

Good choices:

- normalized cross-correlation,
- template matching,
- phase correlation,
- keypoint matching with RANSAC,
- ECC or affine registration,
- learned patch matcher later if needed.

Output:

- `dx`, `dy`,
- possibly `dtheta`,
- match confidence,
- residual alignment error.

### Where AI Helps Most

AI is especially useful for:

- selecting robust landmarks,
- recognizing landmark classes,
- handling changing illumination or blur,
- scoring confidence,
- distinguishing true matches from lookalikes,
- learning remount transform patterns from historical data.


## Training Problems

This project should be split into at least two training problems.

### Problem A: Coarse Localization

Question:

`Where is this low-magnification view on the sample?`

Model output:

- stage coordinate,
- region ID,
- nearest known landmark set,
- coarse translation and rotation.

### Problem B: Fine Registration

Question:

`How far is the current view from the previously scanned location?`

Model output:

- `dx`,
- `dy`,
- optionally `dtheta`,
- confidence,
- same-site / wrong-site classification.

### Optional Problem C: Remount Transform Prediction

Question:

`After replacing the sample, what global transform best explains the new pose?`

Model output:

- translation,
- rotation,
- tilt-related transform parameters,
- uncertainty.


## What Data Should Be Trained

### A. Low-Magnification Localization Data

Collect:

- low-mag images from across the sample,
- true stage coordinates `(x, y)`,
- zoom / magnification,
- illumination setting,
- focus condition,
- session ID,
- sample ID,
- known rotation or remount orientation if available.

Use this to train:

- region recognition,
- coarse localization,
- remount-aware retrieval.

### B. High-Magnification Fine-Matching Data

Collect:

- reference high-mag patch at scan site,
- revisit images of the same site,
- images from nearby wrong locations,
- ground-truth offset between reference and revisit,
- optional rotation difference.

Use this to train:

- offset estimation,
- same-site vs wrong-site matching,
- fine reposition refinement.

### C. Remount / Reinstallation Data

Collect paired before/after-remount data:

- low-mag overview before removal,
- low-mag overview after replacement,
- known shift,
- known rotation,
- known tilt if measurable,
- final manually confirmed recovered position.

Use this to train:

- remount transform prediction,
- robust low-mag recovery under reposition changes.


## Essential Labels

For each captured image or pair, store as many of these as possible:

- `sample_id`
- `session_id`
- `site_id`
- `image_id`
- `stage_x_um`
- `stage_y_um`
- `origin_x_um`
- `origin_y_um`
- `relative_x_um`
- `relative_y_um`
- `zoom_level`
- `magnification`
- `tilt_angle_deg`
- `rotation_deg`
- `focus_state`
- `illumination_state`
- `is_reference`
- `is_same_site`
- `gt_dx_um`
- `gt_dy_um`
- `gt_dtheta_deg`
- `confidence_manual`


## Data the System Must Save Per Scan Site

For each important scan site, save a structured memory record.

Recommended contents:

- sample ID
- site ID
- session ID
- low-mag overview image
- low-mag landmark patches
- low-mag landmark coordinates
- high-mag reference image
- high-mag secondary landmark patches
- local origin coordinates
- target scan coordinates
- coordinates relative to origin
- zoom level
- calibration in `um/px`
- tilt angle
- motion history summary
- final verified human-confirmed location


## Hard Cases That Must Be Included in Training Data

Do not train only on perfect images.

Include:

- small positioning error,
- large positioning error,
- repeated textures,
- rotated re-mounts,
- tilted samples,
- low contrast,
- blur / defocus,
- brightness changes,
- partial contamination,
- partial occlusion,
- hysteresis-related motion mismatch,
- slightly wrong user-provided starting guesses.


## Recommended Data Collection Protocol

### Session Template

For one sample:

1. Capture low-mag overview.
2. Move to several regions and record stage coordinates.
3. For each important region:
   - capture low-mag patch,
   - capture high-mag patch,
   - define local origin if needed,
   - save neighboring landmarks.
4. Remove and replace the sample.
5. Repeat capture after replacement.
6. Manually confirm the correct relocated site.
7. Save the before/after pair.

### Minimum Useful Dataset

Start small but structured:

- 10 to 20 samples,
- multiple sites per sample,
- multiple remount sessions per sample,
- multiple magnification levels,
- both easy and difficult repeated-pattern regions.


## Proposed System Pipeline

### Stage 1: Before Removal

1. Save global overview.
2. Detect or select robust low-mag landmarks.
3. Save high-mag scan-site reference.
4. Save local origin and local coordinates.
5. Save secondary nearby landmarks.

### Stage 2: After Replacement

1. Capture new low-mag overview.
2. Estimate coarse transform to previous session.
3. Move to predicted old region.
4. Match mid/high-mag landmarks.
5. Estimate residual offset.
6. Correct position iteratively.
7. Validate against the stored reference.

### Stage 3: Verification

Report:

- predicted coordinates,
- residual `dx`, `dy`,
- residual rotation if estimated,
- confidence score,
- pass/fail for scan readiness.


## Safe Failure Behavior

If the system is unsure:

- do not declare success,
- do not jump directly to scan,
- ask the user for one more guide point,
- show candidate matches,
- allow manual correction and continue from there.


## Practical Development Roadmap

### Phase 1: Classical Baseline

Implement first:

- multi-landmark save/load,
- coarse template or keypoint matching,
- affine transform estimation,
- fine template matching,
- confidence scoring,
- iterative correction.

This gives a strong non-AI baseline.

### Phase 2: AI Augmentation

Add:

- learned landmark scoring,
- same-site / wrong-site patch classifier,
- remount transform predictor,
- low-mag retrieval embedding model.

### Phase 3: Integrated Closed Loop

Combine:

- motion model,
- hysteresis compensation,
- vision-based correction,
- final scan-site verification.


## Suggested File Outputs for Future Data Collection

Recommended project structure:

```text
collected_data/
  sample_001/
    session_001_before/
      lowmag/
      highmag/
      landmarks/
      metadata.csv
    session_002_after/
      lowmag/
      highmag/
      landmarks/
      metadata.csv
```

Recommended metadata fields:

```csv
sample_id,session_id,site_id,image_id,view_type,stage_x_um,stage_y_um,origin_x_um,origin_y_um,relative_x_um,relative_y_um,zoom_level,tilt_angle_deg,rotation_deg,is_reference,is_same_site,gt_dx_um,gt_dy_um,gt_dtheta_deg,image_path
```


## Key Principle

The system should not only remember a pattern. It should estimate:

- what location is being matched,
- how confident the match is,
- what geometric transform exists between sessions,
- what residual error remains after repositioning.

That is what will make AI-assisted AFM repositioning reliable instead of fragile.
