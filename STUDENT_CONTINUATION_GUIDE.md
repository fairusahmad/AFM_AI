# Student Continuation Guide

This note is for the student who started the project in the initial commit, `gyxiii` (`229c702`, dated 2026-05-18).

The goal is to help you continue the AFM hysteresis simulation and AI-assisted repositioning work without needing to rediscover the current structure from scratch.

## 1. Project Purpose

This project is building an AFM simulation environment with two main directions:

1. hysteresis-aware motion and visualization, and
2. vision-assisted sample/site recovery after remounting.

The long-term research goal is not only to move the simulated stage, but to recover the same physical scan site again after the sample is removed and replaced.

## 2. What Exists Now

Compared with the initial commit, the codebase now includes a much stronger repositioning workflow:

- saved site memory for a scan site,
- named origin support,
- low-magnification overview capture,
- low-mag and high-mag landmark extraction,
- coarse-to-fine relocation,
- affine remount estimation,
- verification before accepting relocation,
- HUD landmark overlays,
- distance and angle display from tip to landmark,
- user-drawn multiple ROI selection for landmark filtering,
- one-ROI-to-one-pattern matching behavior.

The planning document is here:

- [AI_REPOSITIONING_PLAN.md](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\AI_REPOSITIONING_PLAN.md)

## 3. Main Files You Should Know First

- [afm_control_panel.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_control_panel.py)
  Main UI entry point. Builds the figure, HUD overlays, ROI drawing, dock panels, and starts animation.

- [afm_callbacks.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_callbacks.py)
  Most interaction logic lives here: motion, zoom, save reference, load images, remount simulation, relocation, and HUD mode switching.

- [afm_state.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_state.py)
  Shared simulation state. If you add new UI or relocation features, this is usually where new state variables belong.

- [afm_relocation.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_relocation.py)
  Core relocation logic: landmark extraction, template matching, affine estimation, geometry checks, site-memory save/load.

- [afm_utils.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_utils.py)
  Rendering helpers, FOV generation, camera frame generation, probe occlusion logic, and outside-sample masking.

- [artefact_detector.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\artefact_detector.py)
  YOLO-based artefact detection support. Right now the HUD landmark overlay is more tightly connected to `afm_relocation.py` landmark matching than to this detector.

## 4. How To Run

Install the dependencies from [requirements.txt](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\requirements.txt), then run:

```powershell
python afm_control_panel.py
```

The control panel starts the simulation UI and calls `plt.show()` from the main script.

## 5. Current User Workflow In The Simulator

The current intended workflow is roughly:

1. Load a default image or sample image.
2. Navigate to the region of interest.
3. Use `Stop Here` to cancel active travel if needed, or `Go Now` to jump immediately to the current destination.
4. Set a named origin if needed.
5. Save the site memory.
6. Simulate sample removal/remount.
7. Run relocation.
8. Inspect the HUD and landmark verification result.

In the Motion Dock:

- `Motion: ON/OFF` pauses or resumes motion updates globally.
- `Stop Here` cancels active movement and snaps the destination to the current position.
- `Go Now` moves the view immediately to the current destination without waiting for smooth travel.

For HUD-assisted work:

1. Use the `HUD` button to cycle overlay modes.
2. Right-click the viewport to draw detection ROIs.
3. Draw multiple circular ROIs.
4. Each ROI keeps one best detection pattern.
5. Right-click near an ROI to delete that ROI only.

## 6. Important Design Decisions Already In The Code

These are worth preserving unless you intentionally want to redesign them:

- The relocation system is hybrid, not AI-only.
  Classical matching is still the baseline.

- The saved memory uses both low-mag and high-mag information.
  This is important because one patch alone is fragile.

- Verification matters as much as prediction.
  A relocation result should not be accepted only because a match exists.

- The HUD is now separated into detection and distance layers.
  This makes it easier to reduce clutter and reduce runtime cost.

- ROI filtering is used to help the user constrain landmark matching.
  This is useful when repeated patterns exist in the view.

- Remount-created empty padding is treated as outside-sample area during rendering.
  This keeps transformed blank regions visually consistent with the red outside-stage region instead of showing median-gray affine padding.

## 7. Suggested Next Tasks

These are good next steps for you to continue the work:

### A. Make ROI handling more complete

- support rectangle ROI in addition to circle ROI,
- allow renaming ROI labels,
- color-code ROI labels and matched landmarks,
- persist ROI definitions if that becomes useful for a session.

### B. Improve runtime performance

- update HUD matching every `N` frames instead of every frame,
- skip contour extraction when HUD detection is off,
- cache outline extraction when the viewport does not change much.

### C. Strengthen relocation robustness

- improve repeated-pattern rejection,
- expand geometric consistency scoring,
- test on more real microscope images,
- add a better fallback path when confidence is low.

### D. Expand the AI side carefully

- build a real dataset for same-site vs wrong-site classification,
- validate the remount transform predictor,
- improve low-mag retrieval using real samples instead of only simulator logic,
- compare AI predictions against the classical baseline instead of replacing it immediately.

## 8. Recommended Research Mindset

When you continue this project, try to keep these questions in mind:

1. Does the system still work when the sample has repeated patterns?
2. Does it fail safely when confidence is weak?
3. Can a user understand why a relocation was accepted or rejected?
4. Is the AI actually improving the baseline, or just making it more complicated?

That last question is especially important for research quality.

## 9. Good First Reading Order

If you are returning to the project after some time, this is a practical order:

1. Read [AI_REPOSITIONING_PLAN.md](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\AI_REPOSITIONING_PLAN.md)
2. Open [afm_state.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_state.py)
3. Read relocation-related methods in [afm_callbacks.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_callbacks.py)
4. Read matching logic in [afm_relocation.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_relocation.py)
5. Then inspect HUD/ROI logic in [afm_control_panel.py](C:\Users\fairu\OneDrive%20-%20unimap.edu.my\Unimap\HP%20documents\Research\AFM_yixuan\AFM-Hysteresis-Simulation\afm_control_panel.py)

## 10. Final Advice

You already gave the project a strong start with the initial commit. The current codebase is larger now, but the direction is still consistent with that original idea: combine AFM motion simulation with reliable vision-based repositioning.

The best way to continue is:

- keep the system explainable,
- test every new idea against the existing baseline,
- prefer robust incremental improvements over big rewrites,
- document assumptions when adding new AI components.

If you continue from that mindset, the project will stay research-useful and easier for the next person to inherit too.
