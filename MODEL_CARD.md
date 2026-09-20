# Model card - AslNet (template, fill in after training)

**Status: not trained yet.** No performance figure exists. The fields below marked TODO must be filled from
`reports/metrics.json` and the new-signer hold-out after you run `train` and `evaluate`.

| Field | Value |
|---|---|
| Task | Classify one hand-shape image into 27 classes (A-Z, space) |
| Architecture | Residual CNN + squeeze-excite, ~2.8 M parameters, 128x128 input |
| Training data | `own_dataset`; temporal-block split (train 5,373 / val 1,613 / test 1,610) |
| Intended use | Learning aids, demos, research on fingerspelling recognition, with a human in the loop |
| Out of scope | Replacing interpreters; medical, legal, emergency or safety-critical communication; identifying people |
| Input requirement | Image must be produced by the same landmark-overlay preprocessing as training |
| Test-block accuracy / macro-F1 | TODO |
| New-signer hold-out accuracy / macro-F1 | TODO |
| Weakest classes | TODO (from `per_class_metrics.csv`) |
| Calibration (ECE) | TODO |
| Known limitations | Static frames only (J, Z are motion signs); few signers; single-hand; lighting and background range unknown |
| Ethical notes | Fingerspelling is a small part of ASL, a full language with its own grammar. Involve Deaf people in evaluating and shaping any product. |
| Security | Checkpoint is HMAC-signed and loaded with `weights_only=True` |
