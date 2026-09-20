"""Run:  python -m unittest discover -s tests -v      (no PyTorch needed)"""
import io, json, os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from cryptography.exceptions import InvalidTag

from asl.data import split as spl
from asl.security import crypto, integrity, privacy
from asl.security.audit import AuditLog


def fake_manifest(n_classes=3, n=300):
    rows = []
    for c in range(n_classes):
        for i in range(n):
            rows.append(dict(label=f"c{c}", label_id=c, capture_ts=1e9 + i / 30, sha256=f"{c}-{i}", valid=True))
    return pd.DataFrame(rows)


class TestSplit(unittest.TestCase):
    def test_temporal_split_has_embargo_and_all_splits(self):
        d = spl.temporal_block_split(fake_manifest(), 6, (2,), (4,), 10)
        a = spl.audit_split(d)
        self.assertGreaterEqual(a["min_train_to_eval_gap_frames"], 11)
        for s in ("train", "val", "test"):
            self.assertGreater(a["counts"][s], 0)

    def test_random_split_leaks_adjacent_frames(self):
        d = spl.random_split(fake_manifest())
        self.assertEqual(spl.audit_split(d)["min_train_to_eval_gap_frames"], 1)   # twins exist

    def test_bad_block_config(self):
        with self.assertRaises(ValueError):
            spl.temporal_block_split(fake_manifest(), 6, (2,), (2,), 10)


class TestCrypto(unittest.TestCase):
    def setUp(self):
        self.key = crypto.generate_key()

    def test_roundtrip(self):
        blob = crypto.encrypt_bytes(b"hand", "img_1", self.key)
        self.assertEqual(crypto.decrypt_bytes(blob, "img_1", self.key), b"hand")

    def test_nonce_unique(self):
        self.assertNotEqual(crypto.encrypt_bytes(b"x", "a", self.key), crypto.encrypt_bytes(b"x", "a", self.key))

    def test_tamper_and_wrong_id_fail(self):
        blob = bytearray(crypto.encrypt_bytes(b"hand", "img_1", self.key)); blob[-1] ^= 1
        with self.assertRaises(InvalidTag):
            crypto.decrypt_bytes(bytes(blob), "img_1", self.key)
        with self.assertRaises(InvalidTag):
            crypto.decrypt_bytes(crypto.encrypt_bytes(b"hand", "img_1", self.key), "img_2", self.key)

    def test_wrong_key_fails(self):
        blob = crypto.encrypt_bytes(b"hand", "img_1", self.key)
        with self.assertRaises(InvalidTag):
            crypto.decrypt_bytes(blob, "img_1", crypto.generate_key())

    def test_store(self):
        with tempfile.TemporaryDirectory() as t:
            st = crypto.EncryptedImageStore(t, self.key); st.put("img_9", b"abc")
            self.assertEqual(st.get("img_9"), b"abc")
            self.assertNotIn(b"abc", (Path(t) / "img_9.enc").read_bytes())


class TestIntegrity(unittest.TestCase):
    def test_sign_verify_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "m.pt"; f.write_bytes(b"weights")
            integrity.sign_file(f, "k"); self.assertTrue(integrity.verify_file(f, "k"))
            self.assertFalse(integrity.verify_file(f, "other-key"))
            f.write_bytes(b"weights+backdoor"); self.assertFalse(integrity.verify_file(f, "k"))

    def test_unsigned_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "m.pt"; f.write_bytes(b"w"); self.assertFalse(integrity.verify_file(f, "k"))

    def test_dataset_verification(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "a.jpg").write_bytes(b"1")
            rows = [dict(rel_path="a.jpg", sha256=integrity.sha256_file(Path(t) / "a.jpg")),
                    dict(rel_path="gone.jpg", sha256="x")]
            self.assertEqual(integrity.verify_dataset(t, rows), ["missing: gone.jpg"])


class TestAudit(unittest.TestCase):
    def test_chain_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            log = AuditLog(Path(t) / "a.jsonl")
            for i in range(3):
                log.log("evt", i=i)
            self.assertEqual(log.verify_chain(), (True, 3))
            lines = (Path(t) / "a.jsonl").read_text().splitlines()
            e = json.loads(lines[1]); e["details"]["i"] = 99; lines[1] = json.dumps(e)
            (Path(t) / "a.jsonl").write_text("\n".join(lines) + "\n")
            self.assertFalse(log.verify_chain()[0])


class TestPrivacy(unittest.TestCase):
    def test_pseudonym_is_keyed_and_stable(self):
        a, b = privacy.pseudonymize_id("A/Image_1.jpg", "s1"), privacy.pseudonymize_id("A/Image_1.jpg", "s2")
        self.assertEqual(a, privacy.pseudonymize_id("A/Image_1.jpg", "s1"))
        self.assertNotEqual(a, b); self.assertNotIn("Image", a)

    def test_strip_metadata_removes_exif(self):
        im = Image.new("RGB", (32, 32), (10, 200, 30)); ex = im.getexif(); ex[0x010F] = "SecretPhoneMaker"
        buf = io.BytesIO(); im.save(buf, "JPEG", exif=ex)
        self.assertIn(b"SecretPhoneMaker", buf.getvalue())
        clean = privacy.strip_metadata(buf.getvalue())
        self.assertNotIn(b"SecretPhoneMaker", clean)
        self.assertEqual(len(Image.open(io.BytesIO(clean)).getexif()), 0)

    def test_blur_faces_runs_on_blank(self):
        out, n = privacy.blur_faces(np.full((120, 120, 3), 128, np.uint8))
        self.assertEqual((out.shape, n), ((120, 120, 3), 0))


class TestResultsViz(unittest.TestCase):
    def test_figures_and_metrics(self):
        from asl.viz import results
        rng = np.random.default_rng(0); classes = list("ABCDE")
        y = rng.integers(0, 5, 200); p = np.where(rng.random(200) < 0.9, y, rng.integers(0, 5, 200))
        conf = np.clip(rng.normal(0.85, 0.1, 200), 0, 1)
        with tempfile.TemporaryDirectory() as t:
            cm = results.plot_confusion(y, p, classes, Path(t) / "cm.png")
            results.plot_per_class_f1(y, p, classes, Path(t) / "f1.png")
            results.plot_reliability(conf, (y == p).astype(float), Path(t) / "rel.png")
            self.assertEqual(cm.sum(), 200)
            self.assertTrue((Path(t) / "cm.png").stat().st_size > 1000)
            s = results.summarize(y, p, conf, classes); self.assertTrue(0.8 < s["accuracy"] < 1.0)


if __name__ == "__main__":
    unittest.main()
