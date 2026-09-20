# 1. Architecture

![pipeline map](../design/01_pipeline_map.png)

## The idea in one paragraph
Raw images enter a **trust boundary**. Inside it they are validated, cleaned of personal metadata, encrypted,
and described by a **manifest**. Every later stage reads the manifest, never the raw folder, so what a model
trained on is always reproducible and auditable. The split is done by *time*, not randomly, because the frames
are video. The trained model is signed, and the serving layer refuses an unsigned model.

## Layers and where the code lives

| Layer | Job | Code |
|---|---|---|
| Data engineering | validate, hash, dedupe, detect takes, split without leakage, feed the model | `src/asl/data/` |
| Security & privacy | strip metadata, blur faces, pseudonymise IDs, encrypt at rest, verify integrity, audit | `src/asl/security/` |
| Deep learning | augment, train, evaluate, explain | `src/asl/models/` `src/asl/training/` |
| Visualization | EDA, split evidence, result figures, dashboard, live UI | `src/asl/viz/` `design/ui_mockup.html` |
| Serving | secured inference API, live-frame preprocessing | `src/asl/serve/` |

## Key design decisions (and why)

1. **Manifest-first.** One CSV row per image (hash, quality signals, take id, split). Anything that changes the
   training set changes the manifest, and the change is visible in a diff.
2. **Temporal-block split with an embargo.** At 30 fps, twin frames straddle any random split. Blocks of
   consecutive frames go to val/test; train frames within 10 frames of them are dropped. The audit function
   fails loudly if any train frame is closer than that.
3. **Keep the skeleton, mask the box.** The skeleton is the informative part. The magenta box is drawn by the
   capture tool and encodes hand size and shape, a shortcut, so it is painted white (`mask_magenta_box`).
4. **Small custom CNN first.** About 2.8 M parameters trains on a laptop GPU and runs on a phone. A
   MobileNetV3-small option exists for comparison.
5. **Say "not sure".** Below a confidence threshold the app answers "not sure" instead of guessing. A wrong
   confident letter is worse than a pause.
6. **On-device by default.** The most private system is one where frames never leave the device
   (`design/03_deployment_design.png`).

## Trust boundaries
![deployment](../design/03_deployment_design.png)

## Known limits of the design
* Single-frame classification cannot capture **J and Z**, which are motion signs. A landmark-sequence model
  (Phase 2 in `04_MODEL_DESIGN.md`) is the proper fix.
* `space` is a project-specific gesture, not a standard ASL letter.
* All data appears to come from very few recording sessions; see the new-signer hold-out recommendation.
