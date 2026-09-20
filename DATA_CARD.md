# Data card - `own_dataset`

| Field | Value |
|---|---|
| Content | Static hand images for ASL fingerspelling: letters A-Z and a `space` gesture |
| Size | 9,677 files (9,676 after removing one exact duplicate), 27 classes, 297-486 per class |
| Format | JPEG, 300x300 RGB |
| Appearance | Hand crop with the MediaPipe landmark skeleton drawn on it (red dots, white lines), a 4 px magenta box, on a white letterboxed canvas |
| Origin | Self-collected ("own dataset"); frames extracted from video at ~30 fps. The file names carry Unix capture times; recorded on two days (29 and 30 Sep 2024) |
| Recording | 53 continuous takes overall (1-4 per class). Everything suggests very few sessions; the number of signers is unknown |
| Known issues | Near-duplicate consecutive frames; skeleton and box are burned in; unknown signer diversity; J and Z are motion signs shown as stills |
| Personal data | Hand geometry (biometric), some background and clothing. No EXIF. Automated face check found no confirmed faces in a spot check, but it is weak on partial faces |
| Recommended use | Prototype and pipeline development; **not** evidence of real-world accuracy |
| Recommended additions | New signers, rooms, cameras, both hands; raw frames before overlay; consent and retention records |
