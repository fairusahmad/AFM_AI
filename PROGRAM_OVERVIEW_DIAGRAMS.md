# AFM Hysteresis Simulation: Visual Overview

Mermaid is a good fit here because it is lightweight, readable in plain text, and easy to update as the project changes.

## 1. Big Picture

```mermaid
flowchart LR
    U[User / Student] --> UI[UI Controls<br/>afm_ui.py + afm_control_panel.py]
    UI --> CB[Callback Logic<br/>afm_callbacks.py]

    CB --> ST[Simulation State<br/>afm_state.py]
    CB --> STAGE[Stage Motion Model<br/>hysteresis.py]
    CB --> OPTICS[Optics / Focus Model<br/>afm_optics_model.py]
    CB --> RENDER[Image Rendering<br/>afm_animation.py + afm_utils.py]
    CB --> RELOC[Relocation / Matching<br/>afm_relocation.py + image_matching.py]
    CB --> ML[Optional ML Models<br/>afm_ml_recognition.py + afm_phase2_ml.py]

    STAGE --> ST
    OPTICS --> ST
    ST --> RENDER
    ST --> RELOC
    ST --> ML

    RENDER --> OUT[Rendered AFM View]
    RELOC --> OUT
    ML --> OUT
```

## 2. What The Program Is Simulating

```mermaid
flowchart TD
    A[Desired XY movement] --> B[PI hysteresis model]
    B --> C[Actual stage movement<br/>not perfectly equal to command]
    C --> D[Sample image seen by virtual AFM]
    D --> E[Optics and focus effects]
    E --> F[Camera / scan output shown in UI]

    G[Optional AI inverse model] --> B
    G -. tries to compensate .-> C
```

## 3. Main Runtime Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as UI
    participant CB as Callbacks
    participant Stage as Hysteresis Model
    participant State as AFM State
    participant View as Renderer

    User->>UI: Click button / drag / recall site
    UI->>CB: Trigger callback
    CB->>Stage: Request move / scan / recovery
    Stage-->>CB: Actual motion with hysteresis
    CB->>State: Update x, y, zoom, saved memory
    CB->>View: Re-render surface/camera view
    View-->>User: Updated simulation output
```

## 4. Save Region And Recall Concept

```mermaid
flowchart TD
    A[User finds interesting region] --> B[Save Region]
    B --> C[Store site_memory]
    C --> C1[reference_template]
    C --> C2[overview image]
    C --> C3[position / target / landmarks]

    D[Later: sample remounted] --> E[AI Recall / Recover]
    E --> F[Use overview / landmarks / template matching]
    F --> G[Estimate coarse location]
    G --> H[Run fine template matching]
    H --> I[Move back to saved site]
```

## 5. Relocation Logic

```mermaid
flowchart TD
    A[Load saved site_memory] --> B[Get reference_template]
    B --> C[Estimate coarse transform]

    C --> C1[NCC rotation sweep]
    C --> C2[Landmark consensus]
    C --> C3[Optional ML rotation hint]

    C1 --> D[Approximate current location]
    C2 --> D
    C3 --> D

    D --> E[Fine template match]
    E --> F[Verify score / geometry / same-site classifier]
    F -->|Pass| G[Accept recovered position]
    F -->|Fail| H[Fallback / manual correction]
```

## 6. Where Machine Learning Fits

```mermaid
flowchart LR
    A[Collected site memories] --> B[Training scripts]
    B --> B1[train_remount_5w.py]
    B --> B2[train_remount_real.py]
    B --> B3[train_ml_models.py]
    B --> B4[train_repositioning_ai.py]

    B1 --> C1[5w remount model]
    B2 --> C2[real-pair remount model]
    B3 --> C3[legacy deep models]
    B4 --> C4[phase-2 classical ML models]

    C1 --> D[Runtime model loading]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E1[Template-based rotation hint]
    D --> E2[Same-site verification]
    D --> E3[Low-mag retrieval]
```

## 7. Important Clarification About The New 5w Model

```mermaid
flowchart TD
    A[5w model training] --> B[Input: reference_template patch]
    B --> C[Create warped patch]
    C --> D[Predict dx_px, dy_px, angle]

    E[Correct runtime use] --> F[reference_template vs candidate patch]
    F --> G[Useful for rotation hint / patch comparison]

    H[Incorrect runtime use] --> I[overview image vs overview image]
    I --> J[Prediction may look valid but be wrong in the app]
```

## 8. File Map

```mermaid
flowchart TD
    UI[UI Layer] --> UI1[afm_ui.py]
    UI --> UI2[afm_control_panel.py]
    UI --> UI3[afm_callbacks.py]

    CORE[Simulation Core] --> C1[hysteresis.py]
    CORE --> C2[afm_state.py]
    CORE --> C3[afm_animation.py]
    CORE --> C4[afm_optics_model.py]

    VISION[Vision / Relocation] --> V1[afm_relocation.py]
    VISION --> V2[image_matching.py]
    VISION --> V3[afm_phase2_ml.py]

    ML[ML / Training] --> M1[afm_ml_recognition.py]
    ML --> M2[train_remount_5w.py]
    ML --> M3[train_remount_real.py]
    ML --> M4[test_5w_model.py]
    ML --> M5[train_ml_models.py]
```

## 9. Plain-English Summary

This project is not only a machine learning project.

It is mainly:

1. A simulation of AFM stage behavior with hysteresis.
2. A virtual imaging system that shows what the AFM would see.
3. A relocation system that tries to return to the same sample site after remounting.
4. An optional ML-assisted layer that helps with recognition and recovery, but does not replace the full classical pipeline.

