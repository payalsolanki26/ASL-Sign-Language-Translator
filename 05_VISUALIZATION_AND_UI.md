# 5. Data visualization and UI design

## Figure catalogue (`reports/figures/`)
| File | Question it answers | Made by |
|---|---|---|
| `01_class_distribution.png` | Is the data balanced? | `viz/eda.py` |
| `02_sample_grid.png` | What does one frame per class look like? | `viz/eda.py` |
| `03_capture_timeline.png` | Are frames independent? (No: continuous 30 fps takes.) | `viz/eda.py` |
| `04_quality_distributions.png` | Brightness, sharpness and landmark-pixel spread | `viz/eda.py` |
| `05_split_composition.png` / `06_split_strips.png` | Which frames went where, and where is the embargo? | `viz/eda.py` |
| `07_leakage_demo.png` | What does the split choice cost or hide? | `viz/eda.py` |
| `08_confusion_matrix.png` | Which letters get confused? | `viz/results.py` (after `evaluate`) |
| `09_per_class_f1.png` | Which classes are weak? | `viz/results.py` |
| `10_reliability.png` | Are confidence scores trustworthy? (ECE) | `viz/results.py` |
| `11_training_curves.png` | Over- or under-fitting? | `viz/results.py` |
| `12_gradcam.png` | Is the model looking at the hand? | `viz/explain.py` |

`reports/dashboard.html` (from `dashboard`) embeds all of them in a single file and shows the numbers beside them.
Figures 08-12 appear after training; the result plotting code is unit-tested on synthetic predictions.

Style rules: one palette (teal = train / good, amber = validation, brick = test / problem, grey = embargo);
titles state the question; no chart junk.

## App UI design (`design/ui_mockup.html`)
Open it in a browser. The switch at the top previews three states; **Play demo** spells HELLO with the hold ring.

**Concept.** A dark camera stage with the same red-dot skeleton the model was trained on, a very large letter
with its confidence, and a calm side column for the word being spelled.

**Behaviour**
* A letter is committed after it stays the top guess for 0.8 s (hold ring shows progress).
* Below the confidence threshold (default 55%, user-adjustable) the UI says "not sure" and adds nothing.
* "No hand" shows a plain instruction instead of a stale letter.
* Top-3 guesses are always visible so the user can see near-misses (M vs N).
* Privacy is on screen, not buried: "Reading on this device", a server switch that is off by default.
* Accessibility: state is text, not only colour; keyboard focus rings; reduced-motion respected; responsive to phone width.
* J and Z: show "motion sign, check twice" (specified, not yet in the mockup).
