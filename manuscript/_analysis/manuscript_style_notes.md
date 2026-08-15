# Manuscript Style Study

## Source papers reviewed

1. `s12951-025-03923-9.pdf`
   - Journal of Nanobiotechnology
   - Style type: application-driven biomedical paper
2. `s43593-026-00127-y.pdf`
   - eLight
   - Style type: engineering/optics paper with stronger theory emphasis

Supporting extracted assets:

- Text extraction: `manuscript/_analysis/*_pages.txt`
- Caption extraction: `manuscript/_analysis/*_captions.txt`
- Embedded figure sheets:
  - `manuscript/_analysis/figures/s12951-025-03923-9_contact.jpg`
  - `manuscript/_analysis/figures/s43593-026-00127-y_contact.jpg`

## Shared manuscript style

These two papers are different in domain, but they share a clear Springer-style logic:

1. Start from a practical bottleneck, not from the algorithm alone.
2. State why current methods are insufficient in speed, cost, complexity, or scalability.
3. Introduce the physical phenomenon or device principle as the key enabler.
4. Convert that phenomenon into a structured workflow or platform.
5. Validate with a staged results sequence:
   - principle
   - experiment
   - quantitative evaluation
   - interpretation
   - practical significance
6. Use figures as the main carrier of the argument, with text mainly guiding the reader through the figures.

## Writing style observations

## 1. Opening style

The introductions are problem-first and impact-first.

- The first paper opens with a clinical urgency, mortality, and need for rapid identification.
- The second paper opens with a scaling bottleneck in neural networks and optical hardware.

The common pattern is:

1. Big real-world problem
2. Why existing solutions are limited
3. Why a physical or computational alternative is promising
4. The specific gap that remains
5. "Here we..." statement with the contribution

For your AFM manuscript, the introduction should likely open with:

- the practical difficulty of precise AFM remounting/repositioning
- why operator-dependent relocation is slow or unreliable
- why hysteresis/nonlinearity matters for nanoscale targeting
- why a simulation or AI-assisted framework is useful
- what this project contributes beyond a generic simulator

## 2. Sentence style

Both papers use:

- formal, compressed sentences
- high information density
- few rhetorical flourishes
- strong use of contrast words: "however", "thus", "therefore", "consequently", "furthermore"
- quantitative claims embedded directly in the sentence

Typical tone:

- assertive but not exaggerated
- technical and efficient
- focused on measurable outcomes

This means the AFM manuscript should avoid casual explanation. It should sound like:

- "To address this limitation, ..."
- "We established ..."
- "The results demonstrate ..."
- "These findings suggest ..."

## 3. Section architecture

### Paper 1 structure

- Abstract
- Introduction
- Materials and methods
- Results
- Discussion
- Conclusion

### Paper 2 structure

- Abstract
- Introduction
- Results
- Discussion
- Methods

The first paper is closer to the style you probably want for this AFM project, because it is strongly application-led and image-led.

Recommended AFM structure:

1. Abstract
2. Introduction
3. System design / simulation framework
4. Materials and methods
5. Results
6. Discussion
7. Conclusion

If targeting a more engineering journal, you could also use:

1. Abstract
2. Introduction
3. Results
4. Discussion
5. Methods

## 4. How the papers use figures

This is the strongest shared pattern.

The figures are not decorative. Each figure advances one argument step.

### Figure grammar in paper 1

Observed sequence:

1. Workflow / concept figure
2. Time-evolution image grid
3. Quantitative plot from image evolution
4. Higher-magnification image grid
5. Microstructural SEM comparison
6. Dataset summary
7. Model architecture
8. Performance metrics
9. Interpretability visualization

This is an excellent template for your AFM paper because it moves from phenomenon to mechanism to dataset to model to validation.

### Figure grammar in paper 2

Observed sequence:

1. Device/system principle
2. Benchmark performance across tasks
3. Extension to more complex model behavior
4. Real-world application

This paper uses larger multi-panel composite figures with stronger systems-engineering storytelling.

## 5. Visual style of the images

Across both papers, the image style has consistent traits:

- Multi-panel figures labeled `a`, `b`, `c`, ...
- Left-to-right narrative flow
- Workflow diagrams at the top or first panel
- Image grids with controlled alignment and repeated scale
- Quantitative plots placed next to representative images
- Color used sparingly to encode categories, not for decoration
- Clean white background
- Consistent iconography and panel spacing
- Figures mix raw images, schematics, plots, and summary tables in one plate

Paper 1 in particular uses a very effective information-sharing style:

- representative microscopy images
- time-series montages
- species/category color coding
- simple arrows showing process flow
- compact summary tables embedded in figures
- model outputs shown alongside image examples

For AFM, this suggests we should build figures that combine:

- scan trajectory diagrams
- hysteresis loops
- before/after repositioning images
- error maps
- timeline or step-flow schematic
- quantitative metrics tables or confusion-style summaries

## 6. Caption style

Captions are functional and dense. They usually do three jobs:

1. Name the figure's main point in the first clause
2. Enumerate each subpanel
3. Include critical conditions such as objective, scale bar, temperature, or dataset split

This is important. The caption is not short. It carries technical metadata.

For AFM figures, captions should include:

- scan size
- step size or displacement range
- hysteresis model condition
- simulation or experimental setting
- number of samples/runs
- error metric definition

## 7. Role of quantitative evidence

Both papers alternate between visual evidence and numerical evidence.

Pattern:

- show the image phenomenon
- convert it to a measurable descriptor
- compare categories or methods
- summarize with a metric

That means an AFM paper in this style should not show only screenshots from the simulator. Each visual should pair with a metric such as:

- repositioning error
- path deviation
- hysteresis compensation error
- convergence time
- robustness across noise or drift
- success rate within tolerance

## 8. What is most reusable for your AFM manuscript

The most reusable style is from `s12951-025-03923-9.pdf`.

Why:

- It is image-led.
- It tells a clear practical story.
- It moves from physical behavior to computational interpretation.
- It uses multi-panel figures to explain a workflow cleanly.
- It is close to what an AFM relocation / hysteresis project needs.

The second paper is still useful, but mainly for:

- high-level system framing
- compact statement of contribution
- stronger engineering tone
- broader scalability claims

## Suggested AFM manuscript blueprint

## Title direction

Use a title that combines:

- AFM task
- underlying limitation
- computational idea
- practical outcome

Example directions:

- "Simulation-guided compensation of hysteresis for accurate AFM repositioning"
- "An image-informed AFM remounting framework for robust nanoscale relocation under hysteresis"
- "Modeling and compensation of AFM hysteresis for precise tip repositioning after sample remounting"

## Abstract blueprint

Follow this 5-part pattern:

1. Problem
   - AFM repositioning after remounting is difficult because hysteresis and drift degrade accuracy.
2. Gap
   - Existing workflows rely heavily on manual operation, repeated scanning, or limited correction models.
3. Method
   - We developed an AFM simulation and/or compensation framework that models hysteresis and guides repositioning.
4. Results
   - Report the most important quantitative gains.
5. Significance
   - State why this matters for repeatable nanoscale measurements.

## Recommended figure sequence for the AFM paper

1. **Figure 1: Overall AFM workflow**
   - remounting/repositioning problem
   - where hysteresis enters
   - simulator or compensation pipeline

2. **Figure 2: Hysteresis behavior**
   - forward and backward scan trajectories
   - ideal vs distorted path
   - parameterized hysteresis curves

3. **Figure 3: Simulation environment**
   - UI or system schematic
   - coordinate frames
   - motion model
   - data flow between user input, actuator model, and output image

4. **Figure 4: Representative relocation outcomes**
   - target position
   - uncompensated result
   - compensated result
   - zoomed error view

5. **Figure 5: Quantitative performance**
   - error vs displacement
   - error vs hysteresis strength
   - success rate
   - repeatability across trials

6. **Figure 6: If AI is involved**
   - model architecture
   - training data split
   - prediction examples
   - confusion matrix or regression error

7. **Figure 7: Interpretation / ablation**
   - which signals or features matter
   - what happens when compensation terms are removed

8. **Figure 8: Practical use case**
   - a realistic AFM revisit scenario
   - step-by-step relocation to the same surface feature

## Visual design rules to copy

- Use consistent panel labels and spacing.
- Keep white backgrounds and avoid heavy ornament.
- Use 1 accent color per category, not many gradients.
- Put workflow schematics before quantitative plots.
- Pair every plot with at least one representative image or trajectory.
- Keep axes and legends minimal but readable.
- Use scale bars and coordinate references wherever spatial interpretation matters.
- Make summary tables compact and embed them inside figures only when they add speed.

## Writing rules to copy

- Lead each subsection with the result, then explain the evidence.
- Put the experimental or simulation condition close to the claim.
- Keep the narrative cumulative: each result should unlock the next one.
- Avoid long literature reviews once the gap is established.
- Use the Discussion to explain limits, robustness, and transfer to real AFM operation.

## Next step when you want the AFM manuscript

When you are ready, I can turn this style study into:

1. a journal-style manuscript outline
2. a figure plan tied to your current AFM code and outputs
3. a draft abstract and introduction
4. a full manuscript in the style closest to paper 1

## Bottom line

The strongest style match for your future AFM manuscript is:

- prose structure from `s12951-025-03923-9.pdf`
- selected engineering framing from `s43593-026-00127-y.pdf`

In practice, that means:

- problem-first writing
- figure-led storytelling
- multi-panel visual summaries
- quantitative validation paired with representative images
- a direct, formal, high-density tone
