# AFM Hysteresis Simulation & AI-Assisted Repositioning

A Python-based simulation environment for AFM (Atomic Force Microscopy) that combines
**PI hysteresis stage modeling** with **vision-based sample relocation** after
remounting. Built with OpenCV, PyTorch, and Matplotlib.

## Overview

This project has two main directions:

1. **Hysteresis-aware motion simulation** — Prandtl-Ishlinskii (PI) model for
   realistic piezo-stage X/Y behavior including creep.
2. **AI-assisted site recovery** — After removing and replacing a sample on the
   stage, automatically recover the original scan position using image matching
   and deep learning.

## Architecture

```
User clicks "Recover Site"
        |
        v
+---- Coarse Localization ----+
|  ML estimates rotation      |
|  NCC sweeps angles with     |
|    rotation-compensated     |
|    template matching        |
+-----------------------------+
        |
        v
+---- Fine Registration ------+
|  NCC template matching      |
|  with expanding search      |
|  range (700 → 2800 um)      |
+-----------------------------+
        |
        v
+---- Verification -----------+
|  NCC score + Landmark       |
|  consensus + ML site        |
|  classifier                 |
+-----------------------------+
```

| Step | Method | Purpose |
|------|--------|---------|
| Coarse rotation | 5w ML model (ResNet18 + MLP) | Estimate remount rotation, narrow NCC sweep |
| Coarse translation | NCC template matching with rotation sweep | Find approximate XY position |
| Fine registration | NCC template matching (cv2.matchTemplate) | Precise sub-micron alignment |
| Verification | NCC + landmark geometry + ML classifier | Confirm same-site before accepting |

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
python afm_control_panel.py
```

### Workflow
1. **Save Region** — Navigate to your area of interest, click "Save Region Memory".
   This stores the current FOV, landmarks, and a low-mag overview.
2. **Remount Sample** — Simulates physically removing and replacing the sample
   (random translation + rotation + tilt).
3. **Recover Site** — Runs coarse-to-fine relocation to find the original position.
4. Press **`m`** to toggle ML mode (narrows NCC angle sweep from ±10° to ±4°).

## Key Features

- **PI Hysteresis Model** — Multi-operator Prandtl-Ishlinskii model with creep
  for realistic piezo-stage behavior.
- **Coarse-to-Fine Relocation** — Low-mag landmark consensus → NCC rotation sweep
  → fine NCC alignment.
- **Deep ML Recognition** — ResNet18 feature extractor + MLP for rotation
  estimation and same-site verification.
- **Real-Frame Training** — Train on actual AFM frame pairs (not synthetic warps)
  via `train_remount_real.py`.
- **Optical Simulation** — Defocus blur, zoom-dependent camera lift, variable
  NA/wavelength focus model.
- **Interactive UI** — Draggable dock panels, HUD overlay, ROI drawing, real-time
  status logging.

## File Structure

```
afm_control_panel.py    # Main entry point & UI construction
afm_callbacks.py        # All interaction logic (motion, zoom, relocation, remount)
afm_state.py            # Shared simulation state
afm_relocation.py       # Landmark extraction, template matching, affine estimation
afm_ml_recognition.py   # ResNet18 feature extractor, MLP models, deep_pair_features
afm_phase2_ml.py        # Phase 2 ML: same-site classifier, remount predictor
afm_animation.py        # Per-frame update loop (auto-scan, zoom, FOV rendering)
afm_utils.py            # FOV cropping, probe occlusion, defocus blur
afm_optics_model.py     # Optical equations (DoF, blur, zoom-lift)
afm_data.py             # Trajectory recording and visualization
afm_ui.py               # Dockable panels, button/radio widgets
hysteresis.py           # PI hysteresis model + creep (NanoPositioner)
sample_generation.py    # Synthetic sample + artifact generation
image_matching.py       # NCC template matching utilities
artefact_detector.py    # YOLO-based artefact detection (experimental)

train_remount_5w.py     # Train 5w ML model on synthetic warp data
train_remount_real.py   # Train on real AFM frame pairs
test_5w_model.py        # Standalone model evaluation script

collected_data/
  models/               # Trained model files (.pkl)
  site_memories/        # Saved scan-site data (images + landmarks + metadata)
```

## Model Loading Priority

Models are loaded from `collected_data/models/` in this order:

1. `deep_remount_predictor_real.pkl` — Trained on real frame pairs (highest priority)
2. `deep_remount_predictor_5w.pkl` — 50K synthetic samples per anchor
3. `deep_remount_predictor_final.pkl` — Manual rename fallback
4. `deep_remount_predictor.pkl` — Legacy model

## Dependencies

- numpy, matplotlib, opencv-python
- pytorch, torchvision (ResNet18)
- scikit-learn (MLPRegressor/Classifier)
- joblib, pandas, ultralytics

## Research Notes

The system follows a **classical-first, AI-augmented** design:

- NCC template matching is the primary localization method (deterministic, reliable).
- The ML model narrows the rotation search space (±4° instead of ±10°).
- ML verification adds a second opinion without replacing the baseline.
- Key design question: *"Is the AI improving the baseline, or just making it more complicated?"*

## License

Research project — contact the authors for usage terms.
