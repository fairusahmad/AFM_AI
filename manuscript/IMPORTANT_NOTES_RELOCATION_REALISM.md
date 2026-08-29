# Important Notes: AFM Relocation Realism Fixes

Date: August 16, 2026

## Purpose

These notes document the important relocation-realism issues that were resolved in the AFM hysteresis and remount-recovery simulator.

The central purpose of these fixes is to keep the relocation pipeline faithful to real AFM operation. In a real AFM system, relocation should be based on the optical camera point of view (POV), not on a hidden ground-truth image of the entire sample. The camera has limited pixel resolution, sees the cantilever from the top, and cannot observe the sample region hidden under the cantilever body and tip.

If the simulator stores or matches against the actual clean sample image, the relocation task becomes unrealistically easy and gives the algorithm information that would not exist in practice.

## Core Principle

The correct relocation model is:

- save site memory from the camera POV,
- include the cantilever in that saved POV,
- apply limited camera resolution,
- use the same camera-style POV again during recovery,
- accept that relocation must succeed or fail based only on what the simulated AFM camera can realistically observe.

## Resolved Issues

### 1. Low-magnification relocation memory was saving the actual sample image

Problem:

- The saved low-magnification overview image could be built from the full sample surface image.
- This meant the coarse relocation memory was based on ground truth rather than camera observation.

Why this was wrong:

- A real AFM operator does not have access to the full clean sample image during relocation.
- Coarse remount recovery should depend on the top-view optical camera image.

Resolution:

- The saved low-magnification overview was changed to a simulated camera capture generated from the current AFM POV around the remembered site.
- New saved site memories should therefore store a camera-based low-mag overview rather than the actual sample image.

Important manuscript meaning:

- The “overview image” in the relocation memory should be described as a simulated low-magnification camera POV, not as a direct sample map.

### 2. Coarse relocation was still using the full stage image internally

Problem:

- Even when a camera-style reference existed, coarse relocation still used the full `surface_image` in parts of the pipeline.
- This preserved an unrealistic shortcut during recovery.

Why this was wrong:

- It violated the intended AFM imaging constraint.
- It reduced the realism of relocation difficulty and performance assessment.

Resolution:

- Coarse recall was redirected to use low-magnification camera-overview imagery and camera-derived landmarks.
- The matching path was moved toward a camera-POV-based coarse recovery strategy.

Important manuscript meaning:

- The relocation pipeline should be framed as camera-view-based coarse recall followed by high-magnification refinement, not as direct full-sample image registration.

### 3. The saved camera recognition image did not visibly show the cantilever

Problem:

- The simulated recognition frame used an occlusion mask, but the cantilever region could be flattened toward background intensity.
- As a result, the saved relocation image did not clearly show the cantilever body and tip.

Why this was wrong:

- In a real AFM optical camera, the cantilever is visible from the top view.
- The cantilever is not just an invisible mask. It is an object in the observed image and also blocks part of the sample.

Resolution:

- The recognition rendering was changed so the cantilever appears as a visible silhouette in the saved and live camera-style relocation image.
- The outside invalid region and the cantilever region are now treated differently.

Important manuscript meaning:

- The simulated camera POV should be described as containing both sample context and the projected cantilever silhouette.

### 4. Camera-resolution realism was incomplete

Problem:

- The relocation-recognition image path could still inherit unrealistically dense image detail from the raw stage crop if camera resolution was not enforced.

Why this was wrong:

- Real AFM relocation depends on finite pixel resolution.
- Excess pixel detail makes matching easier than it should be.

Resolution:

- The recognition view generation was updated to use explicit camera resolution for stored and live relocation imagery.

Important manuscript meaning:

- The relocation memory and matching views should be described as finite-resolution camera images rather than unrestricted crops.

### 5. AI recall crashed when coarse low-mag NCC returned no match

Problem:

- `AI Recall & Recover` could crash when coarse low-magnification matching returned `None`.

Why this mattered:

- Besides being a software bug, it interrupted the intended fallback logic and prevented robust staged recovery.

Resolution:

- Null handling was fixed.
- The recovery flow now continues into fallback logic rather than failing on that condition.

Important manuscript meaning:

- The relocation pipeline supports failure-aware staged fallback, rather than assuming every coarse match succeeds.

### 6. AI recall fallback behavior was too weak when coarse low-mag matching failed

Problem:

- If low-magnification NCC failed, the later refinement path could still be centered too rigidly on the old saved location.

Why this was wrong:

- After remounting, a realistic system should tolerate approximate coarse placement and continue refining from the best available local evidence.

Resolution:

- The AI recall flow now falls back explicitly to the saved coarse region.
- High-magnification refinement is re-centered more sensibly and the fine search window can expand progressively.

Important manuscript meaning:

- The recovery logic should be described as resilient and staged, with explicit fallback when coarse localization is ambiguous.

### 7. Best Focus control was missing from the relocation dock

Problem:

- Best-focus adjustment existed, but it was not directly available in the relocation dock.

Why this mattered:

- Focus is part of realistic relocation operation.
- During practical AFM relocation, focus recovery is tightly coupled to recognizing the site.

Resolution:

- A `Best Focus` control was added to the relocation dock while preserving existing binding behavior.

Important manuscript meaning:

- The relocation workflow includes focus recovery as an operational step, not only XY image matching.

## Practical Interpretation for the Manuscript

The manuscript should now consistently present the relocation system as:

- an AFM-like optical relocation simulator,
- constrained by camera POV, limited resolution, and cantilever visibility,
- using structured memory from realistic observations rather than hidden ground truth,
- combining coarse camera-based recall, landmark fallback, fine local matching, and final verification.

## Wording Guidance

Preferred wording:

- “camera POV”
- “top-view optical camera image”
- “cantilever-visible relocation image”
- “finite-resolution low-magnification overview”
- “camera-based site memory”
- “coarse-to-fine AFM relocation under realistic optical constraints”

Wording to avoid:

- “full sample overview” if it implies ground-truth access
- “actual sample image” for the saved relocation memory
- “direct sample map” unless explicitly discussing non-realistic legacy behavior

## Important Limitation

Although the relocation pipeline has been moved significantly toward realistic AFM camera behavior, the codebase should still be checked carefully whenever new relocation logic is added. Any future use of the clean `surface_image` in saved memory generation, coarse matching, or verification can reintroduce unrealistic shortcuts.

## Recommended Use in Writing

These notes should inform:

- the system-overview description,
- the relocation-memory explanation,
- the figure captions for saved overview and reference images,
- the limitations section,
- and any claims about realism, robustness, or AFM-likeness.
