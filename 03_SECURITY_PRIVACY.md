# 3. Data security & privacy

## What is being protected
Images of hands are **personal data**: hand geometry is biometric, and frames can include a face, clothing,
and a room. Model files are also assets: a swapped checkpoint can be a backdoor. Scan results on your data:

| Check | Result |
|---|---|
| EXIF metadata (GPS, device, time) | none in 9,676 images |
| Automated face detector | flagged 123 (1.3%); a spot-check of 12 showed **hand knuckle false positives, no faces** |
| Backgrounds | walls, clothing and parts of the signer's body are visible in many frames |

The Haar detector misses partial and side faces, so treat these numbers as a floor, not a guarantee.
Review `reports/privacy_flagged_faces.csv` and a random sample by eye before sharing the data.

## Threat model

| Threat | Control | Code |
|---|---|---|
| Dataset copy leaks | AES-256-GCM per file, key from environment / secrets manager | `security/crypto.py` |
| Someone edits or swaps training files | SHA-256 manifest, `verify_dataset` | `security/integrity.py` |
| Encrypted file moved under another ID | ID bound as GCM associated data | `security/crypto.py` |
| Filenames reveal capture time / person | keyed-hash pseudonymous IDs; public manifest keeps only relative order | `security/privacy.py`, `split` command |
| Hidden metadata in images | metadata stripped; clean files pass through byte-identical | `security/privacy.py` |
| Faces in frames | detect + blur before data leaves the ingest boundary | `security/privacy.py` |
| Malicious or swapped model file | HMAC signature checked before load; `torch.load(weights_only=True)` | `security/integrity.py`, `training/evaluate.py`, `serve/api.py` |
| Silent tampering with history | hash-chained audit log; `verify_chain()` detects edits and deletions | `security/audit.py` |
| API abuse / DoS | API key, rate limit, size cap, pixel cap (decompression bombs), type allow-list | `serve/api.py` |
| Leaking images through logs or disk | processed in memory only; logs hold id, latency, label, confidence bucket | `serve/api.py` |
| Browser abuse | strict CORS allow-list, `nosniff`, `no-store`, CSP headers | `serve/api.py` |

Tested (`tests/test_core.py`): encrypt/decrypt round-trip, unique nonces, tamper detection, wrong key, wrong ID,
signature verification and tamper, unsigned model rejected, audit-chain tamper detection, EXIF removal,
keyed pseudonyms. The encryption stage was also run on all 9,676 images and 200 random ones decrypted
byte-for-byte to the originals.

## Key management (the part code cannot do for you)
* Three secrets: `ASL_DATA_KEY` (encrypt), `ASL_SIGNING_KEY` (sign models), `ASL_PSEUDO_SALT` (IDs). `keygen` prints fresh ones.
* Keep them in a secrets manager or a git-ignored `.env`. Never in the repo, notebooks, or chat.
* **Back up `ASL_DATA_KEY`.** Losing it means the encrypted copy is unrecoverable by design.
* Rotate by decrypting and re-encrypting under a new key; keep the old key until the job finishes.
* `.gitignore` already excludes data, keys, checkpoints and `artifacts/`.

## Privacy by design decisions
1. **On-device inference is the default recommendation.** Nothing leaves the device, so most risks disappear.
2. **Data minimisation.** The training manifest needs `label, split, frame_idx, take_id`, not filenames or absolute time.
3. **Purpose limitation and retention.** Write down what the data is for, who can access it, and when it is deleted.
4. **Consent.** If the images show real people (or you collect more), get informed, revocable consent; note that
   hand shape can identify a person.
5. **Encrypted training path.** Set `data.source: encrypted` and the loader decrypts in memory only.

## Optional, not implemented
* **Differential privacy (DP-SGD, e.g. Opacus):** limits what a model can reveal about single images. It costs
  accuracy and time, and it is mainly worth it if the model is published or trained on sensitive third-party data.
* **Landmark-only pipeline:** store and process 21 keypoints instead of images. See `04_MODEL_DESIGN.md`, Phase 2.
* **Adversarial robustness testing** for the public API.

## Residual risks
* Face blur is best-effort. * Encryption at rest does not protect data while it is decrypted in memory on a
compromised machine. * A signed model proves origin, not quality. * If the service is exposed to the internet,
put it behind a TLS reverse proxy and a WAF, and keep dependencies patched.

## Legal note
If real people's images are collected or the app is offered publicly, check your obligations under India's
Digital Personal Data Protection Act, 2023 (and GDPR if you have EU users). This is engineering guidance, not legal advice.
