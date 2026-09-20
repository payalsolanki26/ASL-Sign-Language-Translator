"""
STAGE 9 - DASHBOARD  (Data Visualization)

Builds ONE self-contained HTML file (figures embedded as base64) that summarises the data,
the split, the privacy scan and - when available - the model results. Runs without PyTorch;
sections for training/evaluation appear automatically once those stages have been run.
"""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path

from asl.config import resolve

SECTIONS = [
    ("Data", "What is in the dataset", ["01_class_distribution.png", "02_sample_grid.png", "04_quality_distributions.png"]),
    ("Leakage", "Why the split strategy matters", ["03_capture_timeline.png", "06_split_strips.png", "05_split_composition.png", "07_leakage_demo.png"]),
    ("Model", "How the network performs", ["11_training_curves.png", "08_confusion_matrix.png", "09_per_class_f1.png", "10_reliability.png", "12_gradcam.png"]),
]


def _img(path: Path) -> str:
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'


def _kv(d: dict) -> str:
    return "".join(f"<div><dt>{html.escape(str(k).replace('_', ' '))}</dt><dd>{html.escape(str(v))}</dd></div>" for k, v in d.items())


def build(cfg: dict) -> Path:
    rep, fig = resolve(cfg, "reports_dir"), resolve(cfg, "figures_dir")
    def load(name):
        f = rep / name
        return json.loads(f.read_text()) if f.exists() else None
    blocks = []
    for title, sub, files in SECTIONS:
        figs = [_img(fig / f) for f in files if (fig / f).exists()]
        if not figs:
            blocks.append(f"<section><h2>{title}</h2><p class='muted'>Not generated yet - run the matching pipeline stage.</p></section>")
            continue
        blocks.append(f"<section><h2>{title}</h2><p class='muted'>{sub}</p>{''.join(f'<figure>{g}</figure>' for g in figs)}</section>")
    facts = []
    for label, name in (("Data quality", "data_quality.json"), ("Split audit", "split_audit.json"),
                        ("Privacy scan", "privacy_scan.json"), ("Baseline: random vs leak-safe split", "leakage_demo.json"),
                        ("Test metrics", "metrics.json")):
        d = load(name)
        if d:
            flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}
            facts.append(f"<div class='card'><h3>{label}</h3><dl>{_kv(flat)}</dl></div>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASL classification - project dashboard</title><style>
:root{{--ink:#22262B;--muted:#6B7280;--bg:#F7F8F8;--card:#fff;--line:#E1E4E8;--teal:#2F6F73}}
@media (prefers-color-scheme:dark){{:root{{--ink:#E8EAED;--muted:#9AA0A6;--bg:#15181B;--card:#1D2125;--line:#2E343A;--teal:#6FB7BB}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 20px 64px}}h1{{font-size:1.9rem;margin:0 0 4px}}h2{{margin:40px 0 2px;border-top:2px solid var(--teal);padding-top:14px}}
.muted{{color:var(--muted);margin:0 0 14px}}figure{{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto}}
figure img{{max-width:100%;height:auto;display:block;margin:0 auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:14px 16px}}h3{{margin:0 0 8px;font-size:1rem}}
dl{{margin:0;display:grid;gap:4px}}dl div{{display:flex;justify-content:space-between;gap:12px;border-bottom:1px dashed var(--line);padding:2px 0}}
dt{{color:var(--muted)}}dd{{margin:0;font-variant-numeric:tabular-nums;text-align:right;word-break:break-word}}
</style></head><body><main><h1>ASL fingerspelling classifier</h1>
<p class="muted">Generated from the pipeline outputs in <code>reports/</code>. Nothing here is hand-edited.</p>
<div class="cards">{''.join(facts)}</div>{''.join(blocks)}</main></body></html>"""
    out = rep / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out
