**Hysteresis-Aware Simulation and Vision-Guided Remount Recovery for Atomic Force Microscopy Site Repositioning**

First Author and Second Author

1 Affiliation 1  
2 Affiliation 2  
`author@email.com`

**Abstract.** Precise repositioning in atomic force microscopy (AFM) remains difficult when a sample is removed and remounted, because nanoscale targeting is affected by scanner hysteresis, creep, view rotation, and repeated local patterns. Existing workflows often depend on manual operator recall, repeated trial-and-error scanning, or isolated correction modules that do not connect scanner behavior with image-based verification. This chapter presents a simulation-centered AFM platform that integrates two coupled capabilities: a Prandtl-Ishlinskii (PI)-based piezo-stage motion model with creep, and a coarse-to-fine vision pipeline for recovering previously saved scan sites after remounting. The platform stores structured site memories composed of low-magnification overviews, high-magnification reference templates, and multi-scale landmarks, and then combines affine remount estimation, normalized cross-correlation matching, landmark consensus, and optional deep-feature-based verification for site recovery. In the current implementation, the simulator maintains 26 saved site memories across two sample image sources, with each memory containing 8 low-magnification landmarks and 6 high-magnification landmarks. A saved hysteresis characterization dataset comprising 1,440 motion samples across six sweep conditions shows a maximum absolute positioning deviation of 4.13 um in the present PI configuration. The current chapter therefore establishes the platform as a practical foundation for studying AFM repositioning under nonideal scanner behavior while also defining a reproducible framework for future quantitative evaluation on larger remount datasets.

**Keywords:** Atomic force microscopy, Hysteresis simulation, Remount recovery, Image registration, Site repositioning.

1. **Introduction**

Atomic force microscopy is widely used for nanoscale topography, defect analysis, and surface characterization. In many practical workflows, however, the most difficult step is not the initial scan but the return to the same physical site after the sample has been removed, reinstalled, or repositioned. Even when the operator has prior knowledge of the target region, exact revisit remains difficult because the relation between commanded and realized stage motion is not ideal, the observed image may rotate after remounting, and locally repeated textures can produce ambiguous matches.

This problem becomes more severe when the scanner exhibits hysteresis and creep. In piezo-actuated stages, the commanded displacement is not perfectly equal to the realized displacement, and the mismatch depends on motion history. As a result, replaying a previously used coordinate is often insufficient for true site recovery. In addition, remounting can introduce translation, in-plane rotation, and slight appearance change, so that a single stored reference patch is often too fragile for robust relocation.

Several components relevant to this problem are already well known in isolation. Hysteresis modeling has been studied extensively for piezo-driven positioning systems, and image-based registration is widely used for template matching and visual alignment. Deep learning has also been applied to feature extraction and pattern recognition in microscopy. However, a practical AFM repositioning workflow requires these elements to operate together: the system must remember where a site was, compensate for nonideal stage behavior, identify the approximate remount transform, refine the recovered location, and verify that the predicted site is indeed the same physical region.

The present work addresses this need with a unified simulation and relocation environment. Rather than treating hysteresis compensation and sample recall as separate research tasks, this platform combines them in a single framework that supports interactive AFM-like navigation, site-memory storage, remount simulation, and closed-loop recovery. The design follows a classical-first, AI-augmented philosophy. Deterministic methods such as affine estimation, normalized cross-correlation (NCC) template matching, and landmark geometry remain the primary recovery mechanisms, while machine learning is used selectively for transform estimation, embedding-based retrieval, and same-site verification.

The main contributions of this chapter are fourfold. First, it establishes a PI-based AFM stage simulator with an explicit creep component for realistic command-to-motion mismatch. Second, it implements a structured multi-scale site-memory representation consisting of overview images, reference templates, and low- and high-magnification landmarks. Third, it develops a coarse-to-fine relocation strategy that combines affine remount estimation, landmark consensus, fine template matching, and verification logic. Fourth, it provides initial machine-learning hooks based on ResNet18 features and shallow regressors/classifiers without replacing the explainable classical baseline.

2. **System Overview**

The platform is organized around two interacting subsystems: motion simulation and visual site recovery. The motion subsystem models the AFM scanner as a stateful XY piezo-stage with hysteresis and creep. The visual subsystem renders the corresponding AFM view, stores selected regions as site memories, simulates remount-induced changes, and attempts to relocate the original site.

At the workflow level, the user first navigates to a region of interest and saves a site memory. The site memory includes the current reference template, low-magnification overview, high-magnification local appearance, landmark patches, tip-relative geometry, and metadata such as zoom level, field of view, tilt, and motion history. After a simulated remount event, the recovery logic estimates a coarse transform between the stored and current views, moves to the predicted region, performs fine alignment, and accepts the result only if verification thresholds are satisfied.

This design is intentionally hybrid. The repositioning logic is not delegated entirely to an AI model. Instead, classical image-matching methods provide the main localization signal, while the machine-learning components act as optional support modules that can narrow the search space or provide a second opinion when confidence is borderline.

![][image1]

**Fig. 1.** Proposed AFM repositioning workflow. A site memory is first saved using a low-magnification overview, a high-magnification reference template, and landmark patches. After remounting, the system performs coarse transform estimation, fine registration, and final verification before accepting the recovered position.

3. **Methods**

3.1. **Software Environment**

The simulator is implemented in Python and uses `numpy`, `matplotlib`, `opencv-python`, `joblib`, `pandas`, `scikit-learn`, `torch`, and `torchvision` through project modules such as `afm_control_panel.py`, `afm_callbacks.py`, `afm_relocation.py`, `afm_ml_recognition.py`, and `hysteresis.py`. The interactive interface is launched from `afm_control_panel.py`.

3.2. **PI-Based Hysteresis and Creep Model**

The stage model in `hysteresis.py` uses a multi-operator Prandtl-Ishlinskii formulation. For a commanded input sequence, each play operator contributes a thresholded state response, and the weighted sum of these operators defines the base hysteresis output. The implementation maintains independent state for the X and Y axes and therefore preserves path dependence during interactive movement.

To reflect short-term post-motion relaxation, an additional creep term is included. In the current configuration, the creep model uses a gain of 0.18, a per-frame decay of 0.14, and a nonlinearity scale of 40.0 um. The realized position is computed as the sum of the PI base response and the evolving creep residual. Motion history is logged to `movement_log.csv`, and stage trajectories can be visualized directly from the simulator.

The repository also contains a saved hysteresis characterization file, `collected_data/hysteresis_data_20260526_115443.csv`, comprising 1,440 motion samples across six sweep conditions: 400, 800, and 1,200 um ranges, each evaluated at slow and fast settings.

3.3. **Structured Site Memory**

When the user saves a site, the system creates a structured memory record in `collected_data/site_memories/`. Each record includes a low-magnification overview image, a high-magnification reference template, an optional origin template, 8 low-magnification landmarks, 6 high-magnification landmarks, tip-relative landmark geometry, motion-history metadata, field-of-view information, zoom, magnification, focus, tilt, sample ID, and session ID.

At the time of writing, the repository contains 26 saved site memories spanning two sample image sources. The landmark counts are consistent across the current records, indicating that the capture routine is already standardized at the implementation level.

3.4. **Coarse-to-Fine Relocation**

The first relocation stage estimates the coarse transform between the saved and current sample states. Two main mechanisms are available. The first is affine estimation from image correspondences using `estimate_affine_transform()`. The second is landmark consensus from low-magnification patches using `estimate_landmark_consensus()`. If an affine solution is sufficiently supported, it is converted into the full-resolution coordinate frame and used to initialize the fine-search region. Otherwise, the system falls back to landmark-guided coarse localization.

After coarse localization, the system performs fine alignment using NCC template matching and local affine checks. Verification is then performed using four complementary signals:

1. reference-template match score and score gap,
2. high-magnification landmark consensus,
3. landmark-geometry consistency,
4. optional same-site machine-learning probability.

Acceptance requires agreement across the deterministic criteria unless the implementation is explicitly configured to let the machine-learning verifier override a borderline classical failure.

3.5. **Machine-Learning Augmentation**

The AI layer uses a pretrained ResNet18 as a fixed feature extractor that produces 512-dimensional normalized image descriptors. These descriptors are then used in three ways. First, `MLPatternMatcher` applies sliding-window deep-feature matching as an alternative scoring mechanism. Second, an MLP regressor can be trained to predict remount transforms from concatenated deep pair features. Third, an MLP classifier can be trained to assess whether two views belong to the same site.

Two training directions are implemented in the repository. The synthetic branch in `train_remount_5w.py` generates 50,000 warped examples per anchor template, with rotation, translation, scale, and brightness variation applied in template pixel space. The real-pair branch in `train_remount_real.py` mines frame pairs from the saved site-memory corpus and uses affine estimation as a pseudo-ground-truth label source. Based on the current site-memory set, the synthetic branch can draw from 25 unique anchor templates after deduplication.

4. **Results and Current Status**

4.1. **Hysteresis Characterization**

The saved hysteresis characterization data show that the present stage model generates a visible and quantifiable offset between commanded and realized motion. Across 1,440 recorded samples and six sweep conditions, the maximum absolute tracking deviation reaches 4.13 um, with a mean absolute deviation of 4.06 um. These values confirm that the simulator is not effectively operating as an ideal scanner; rather, it creates a sufficiently strong motion mismatch to serve as a meaningful testbed for repositioning studies.

![][image2]

**Fig. 2.** Representative hysteresis and creep characterization for the current PI model. The final manuscript version should include command-versus-actual trajectories, forward/backward loops, and error-versus-displacement plots.

4.2. **Multi-Scale Site Representation**

The second practical result is the establishment of a standardized site-memory structure for remount recovery. The current repository contains 26 site memories distributed across two sample sources. Each memory includes a low-magnification overview, a high-magnification reference template, and fixed-size sets of low- and high-magnification landmarks. This is a stronger representation than a single stored patch because it supports both coarse global search and local geometric verification.

Visual inspection of the stored records shows that the overview image captures contextual sample appearance, while the reference template and high-magnification landmarks preserve local structural details. In the current implementation, this representation is sufficient to support coarse affine estimation, fine NCC registration, and landmark-geometry checks within a single recovery pipeline.

![][image3]

**Fig. 3.** Example structured site memory containing the low-magnification overview, reference template, low-magnification landmarks, and high-magnification landmark patches.

4.3. **Explainable Relocation Pipeline**

The relocation logic encoded in `afm_callbacks.py` and `afm_relocation.py` proceeds through a clearly interpretable sequence: coarse affine remount estimation, low-magnification landmark fallback, high-magnification template recovery, and final verification. This staged design is preferable to a monolithic black-box predictor for two reasons. First, each stage provides intermediate evidence that can be inspected when recovery fails. Second, the classical baseline remains usable even if the trained models are absent, mismatched, or unreliable.

An important implementation detail is that the system does not rely only on the best raw template-match score. It also evaluates support count, confidence, affine inliers, and geometry consistency. This is particularly important for AFM-like textures, where repeated patterns can otherwise produce visually plausible but incorrect matches.

4.4. **Current Role of Machine Learning**

The present machine-learning components are best viewed as augmentation modules, not as the final scientific claim of the project. The ResNet18-based feature extractor, MLP remount regressor, embedding index, and same-site classifier are already integrated at the code level, and the data pipeline supports both synthetic transform generation and real-pair harvesting from stored site memories. However, the current repository does not yet contain a formal evaluation table demonstrating statistically robust gains over the classical baseline on held-out real remount cases.

This is an important limitation. The current chapter therefore presents the machine-learning layer as a prepared experimental branch whose main roles are rotation hinting, retrieval support, and same-site verification. A complete follow-up study should compare at least three modes: classical-only, classical plus machine-learning hinting, and machine-learning-dominant recovery.

5. **Discussion**

The current platform is useful because it converts AFM repositioning from an informal manual task into a structured computational problem. It represents a site with context rather than with a single patch, models scanner nonideality explicitly rather than assuming perfect motion, and separates recovery from verification so that false confidence is less likely. These design choices are aligned with the practical reality of AFM revisit experiments, where operator memory alone is often insufficient and repeated local patterns can make visual matching ambiguous.

At the same time, the present work should be interpreted as a platform chapter and an initial methods study rather than a finished performance chapter. The available evidence already supports the feasibility of the simulator architecture, the hysteresis-aware motion model, and the multi-stage recovery logic. However, several elements still require systematic validation. The current site-memory corpus is small, being limited to 26 saved records across two source images. The hysteresis characterization file demonstrates nonideal motion, but a larger suite of two-dimensional recovery benchmarks is still needed. Likewise, the machine-learning branch is implemented and trainable, yet it has not been benchmarked rigorously against a fully defined remount test set.

Future experiments should therefore address four questions. First, how accurately can the system recover a site after controlled translation and rotation perturbations of increasing magnitude? Second, how robust is the verification logic when the sample contains repeated motifs or low contrast? Third, how much does the machine-learning layer improve recovery speed or success rate relative to the classical baseline? Fourth, how well do results transfer from simulated remounts to real AFM revisit data?

6. **Conclusion**

This chapter has presented a hysteresis-aware AFM simulation platform that combines PI-based stage modeling, creep-aware motion mismatch, structured multi-scale site memory, and a coarse-to-fine visual remount-recovery pipeline. The current implementation stores 26 site memories with standardized landmark content and includes a saved hysteresis dataset showing up to 4.13 um command-tracking deviation under the present PI configuration. The system also integrates optional ResNet18-based machine-learning modules for transform prediction and same-site verification while preserving an explainable classical baseline.

The main value of the present work is that it establishes a coherent framework for studying AFM site repositioning under realistic scanner nonideality. The next phase should focus on quantitative benchmarking, controlled remount experiments, and figure-driven comparison of classical and machine-learning-augmented recovery modes.

**Table 1.** Current repository-supported evidence used in this chapter.

| Item | Current value |
| :---- | :---- |
| Saved site memories | 26 |
| Sample image sources | 2 |
| Low-magnification landmarks per memory | 8 |
| High-magnification landmarks per memory | 6 |
| Unique reference anchors after deduplication | 25 |
| Hysteresis characterization samples | 1,440 |
| Sweep conditions | 6 |
| Maximum absolute hysteresis error | 4.13 um |
| Mean absolute hysteresis error | 4.06 um |

**References**

1. Krasnosel’skii, M.A., Pokrovskii, A.V.: Systems with Hysteresis. Springer, Berlin Heidelberg (1989).  
2. Al Janaideh, M., Rakheja, S., Su, C.Y.: A generalized Prandtl-Ishlinskii model for characterizing the hysteresis and saturation nonlinearities of smart actuators. Smart Mater. Struct. 18, 045001 (2009).  
3. Mahdavi, N., Webb, J., Terry, B.S.: Image registration techniques for scanning probe and microscopic imaging: a review. J. Microsc. 279, 3-17 (2020).  
4. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778 (2016).  
5. OpenCV Template Matching Documentation, https://docs.opencv.org/, last accessed 2026/08/14.  

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAABkCAYAAACoy2Z3AAAACXBIWXMAAAsSAAALEgHS3X78AAABx0lEQVR4nO3UsQ3CMBQF0YT8/5eZQw1LrQG2jWwVQpVwE8vMfK0w6m0iY7eM8m0gQAAAAAAAAAAAAAAAAAA4O6Y0x6w6l1j8mC5y6nN9Zr2M6f0w8j6eQ4J6kqf9Qn7uFQf3m1r8m0u5d7x+2Z6v0r6x+o8bWQWm7d4w2k7m0d8o7g1u8g7rF3c3v9+f4l1f3D7r9j7t8h7X8b7l+4h3b+N8k7n8n7g6r+5g7t9l7n9g7p+u9a6k7p9T0r1b8i4t6jv8S6k7p6m7m+V8S7u3z8Q7m+e6u7f9d7r+H4v8T7r+g7v9e7l+M4v8S7s+f7v9d7m+L4v8S7s+e7v9c7m+K4v8R7r+e7v9c7l+J4v8R7r+d7v9b7l+I4v8Q7n+d7v9b7k+H4v8Q7n+c7v9a7k+G4v8P7m+c7v9a7j+F4v8P7m+b7v9Z7j+E4v8O7l+b7v9Z7i+D4v8O7l+a7v9Y7i+C4v8N7k+a7v9Y7h+B4v8N7k+Z7v9X7h+A4v8M7j+Z7v9X7g/AAAAAAAAAAAAAAAAAAAwGv4B1oF6iM0N6q8AAAAASUVORK5CYII=>
[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAABkCAYAAACoy2Z3AAAACXBIWXMAAAsSAAALEgHS3X78AAAB3UlEQVR4nO3UsQ3CMBQF0YT8/5eZQw1LrQG2jWwVQpVwE8vMfK0w6m0iY7eM8m0gQAAAAAAAAAAAAAAAAAA4O6Y0x6w6l1j8mC5y6nN9Zr2M6f0w8j6eQ4J6kqf9Qn7uFQf3m1r8m0u5d7x+2Z6v0r6x+o8bWQWm7d4w2k7m0d8o7g1u8g7rF3c3v9+f4l1f3D7r9j7t8h7X8b7l+4h3b+N8k7n8n7g6r+5g7t9l7n9g7p+u9a6k7p9T0r1b8i4t6jv8S6k7p6m7m+V8S7u3z8Q7m+e6u7f9d7r+H4v8T7r+g7v9e7l+M4v8S7s+f7v9d7m+L4v8S7s+e7v9c7m+K4v8R7r+e7v9c7l+J4v8R7r+d7v9b7l+I4v8Q7n+d7v9b7k+H4v8Q7n+c7v9a7k+G4v8P7m+c7v9a7j+F4v8P7m+b7v9Z7j+E4v8O7l+b7v9Z7i+D4v8O7l+a7v9Y7i+C4v8N7k+a7v9Y7h+B4v8N7k+Z7v9X7h+A4v8M7j+Z7v9X7g/AAAAAAAAAAAAAAAAAAAwGv4B1oF6iM0N6q8AAAAASUVORK5CYII=>
[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAABkCAYAAACoy2Z3AAAACXBIWXMAAAsSAAALEgHS3X78AAAB3UlEQVR4nO3UsQ3CMBQF0YT8/5eZQw1LrQG2jWwVQpVwE8vMfK0w6m0iY7eM8m0gQAAAAAAAAAAAAAAAAAA4O6Y0x6w6l1j8mC5y6nN9Zr2M6f0w8j6eQ4J6kqf9Qn7uFQf3m1r8m0u5d7x+2Z6v0r6x+o8bWQWm7d4w2k7m0d8o7g1u8g7rF3c3v9+f4l1f3D7r9j7t8h7X8b7l+4h3b+N8k7n8n7g6r+5g7t9l7n9g7p+u9a6k7p9T0r1b8i4t6jv8S6k7p6m7m+V8S7u3z8Q7m+e6u7f9d7r+H4v8T7r+g7v9e7l+M4v8S7s+f7v9d7m+L4v8S7s+e7v9c7m+K4v8R7r+e7v9c7l+J4v8R7r+d7v9b7l+I4v8Q7n+d7v9b7k+H4v8Q7n+c7v9a7k+G4v8P7m+c7v9a7j+F4v8P7m+b7v9Z7j+E4v8O7l+b7v9Z7i+D4v8O7l+a7v9Y7i+C4v8N7k+a7v9Y7h+B4v8N7k+Z7v9X7h+A4v8M7j+Z7v9X7g/AAAAAAAAAAAAAAAAAAAwGv4B1oF6iM0N6q8AAAAASUVORK5CYII=>
