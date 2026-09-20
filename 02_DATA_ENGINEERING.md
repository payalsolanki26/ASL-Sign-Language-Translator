# 2. Data engineering

## Stages

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | Ingest + validate | `ingest` | `artifacts/manifest.csv`, `reports/data_quality.json` |
| 2 | Leak-safe split | `split` | `artifacts/manifest_split.csv`, `reports/split_audit.json`, `data/manifests/manifest_public.csv` |
| 3 | Privacy scan | `privacy-scan` | `reports/privacy_scan.json` |
| 4 | Figures | `eda` | `reports/figures/01-06` |
| 5 | Leakage demo | `leakage-demo` | `reports/leakage_demo.json`, `07_leakage_demo.png` |

## Manifest schema (`manifest.csv`)
`rel_path, label, label_id, capture_ts, sha256, width, height, mode, blur_var, skeleton_px, brightness, valid, issues, take_id`
and, after splitting, `frame_idx, split, block`.

## Validation rules
Hard fail (excluded): unreadable file, size not 300x300, mode not RGB, fewer than 25 red landmark pixels
(a proxy for "a hand was detected"), exact duplicate. Soft flag (kept, reported): blur variance below 20.

## Findings on your dataset

| Check | Result |
|---|---|
| Files / valid | 9,677 / 9,676 (1 exact duplicate) |
| Classes, per-class range | 27; 297 (C) to 486 (S), ratio 1.64 |
| Image geometry | all 300x300 RGB |
| Median gap between frames | 0.0335 s (30 fps video) |
| Takes (gap > 2 s) | 53 in total, 1-4 per class |
| Blurry / EXIF | 0 / 0 |
| Mean brightness / median sharpness | 163 / 847 (Laplacian variance) |

![timeline](../reports/figures/03_capture_timeline.png)

## The split
Each class recording is cut into 6 contiguous blocks: block 2 is validation, block 4 is test, the rest is
train, and 10 train frames beside each held-out block are embargoed.

| Split | Images |
|---|---|
| train | 5,373 |
| validation | 1,613 |
| test | 1,610 |
| embargoed (dropped) | 1,080 |

Audit: the closest train-to-eval pair is 11 frames (0.37 s) apart, and no content hash appears on both sides.

![strips](../reports/figures/06_split_strips.png)

### How much did leakage matter here?
A deliberately simple baseline (32x32 pixels, PCA to 80 dimensions, then k-NN or logistic regression):

| Model | Random split | Leak-safe split |
|---|---|---|
| 1-NN | 100.0% | 96.9% |
| Logistic regression | 99.9% | 98.5% |

The gap is real but small. The bigger message is that a trivial model scores about 97% even on the leak-safe
split, which means a test block from the same recordings is **an easy exam**. Report the deep model's score
alongside a fresh new-signer hold-out, and treat the within-session number as a sanity check only.
Even the 0.37 s embargo leaves the same pose, lighting and signer on both sides.

## Imbalance
1.6x is mild. The loss is class-weighted (inverse frequency); report macro-F1, not just accuracy.

## Recommended data improvements
* Record 5+ signers, several rooms and lighting conditions, both hands, and different camera distances.
* Record J and Z as short clips if motion recognition is wanted.
* Keep the raw frames (before the skeleton is drawn) if you can: they allow retraining on raw pixels or
  re-extracting landmarks later. Today only the overlay version exists.
