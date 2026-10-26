"""Inspection report exports: a self-contained HTML report (print it to PDF from the browser), Markdown and JSON.

Everything comes from the files a walkthrough already has: result.json (OpenCV pipeline), agent.json (the agent's
trace, decision, risk assessment and the person's approval) and the images next to them.
"""

from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path

import cv2

STATUS_LABEL = {"confirmed": "Confirmed", "needs_review": "Needs review", "candidate": "Not reviewed",
                "dismissed": "Dismissed"}
KIND_LABEL = {"crack": "Crack", "stain": "Stain", "mould": "Mould", "rust": "Rust", "peeling": "Peeling paint"}


def load(folder: Path) -> dict:
    folder = Path(folder)
    r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    a = json.loads((folder / "agent.json").read_text(encoding="utf-8")) if (folder / "agent.json").exists() else {}
    if not a.get("risk"):
        from . import risk
        a["risk"] = risk.assess_result(a.get("findings") or r.get("findings", []), r)
    return {"folder": folder, "result": r, "agent": a}


def bundle(folder: Path) -> dict:
    """Everything about one inspection in one JSON document (images by relative path)."""
    d = load(folder)
    r, a = d["result"], d["agent"]
    return {"name": r["name"], "property": r.get("property"), "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "decision": a.get("decision"), "approval": a.get("approval"), "risk": a.get("risk"),
            "integrity": r.get("integrity"), "coverage": r["coverage"], "gaps": r.get("gaps"),
            "refilm": a.get("refilm", []), "findings": a.get("findings") or r.get("findings", []),
            "rejected": r.get("rejected", []), "labels": r.get("labels", []), "quality": r.get("quality"),
            "privacy": r.get("privacy"), "layout": r.get("layout"), "timings_s": r.get("timings_s"),
            "agent_engine": a.get("engine"), "trace": a.get("trace", []), "opencv": r.get("opencv")}


def _img(folder: Path, rel: str | None, max_side: int = 900, quality: int = 82) -> str | None:
    if not rel:
        return None
    img = cv2.imread(str(Path(folder) / rel))
    if img is None:
        return None
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def _money(v: float, cur: str = "USD") -> str:
    return f"{cur} {v:,.0f}" if cur else f"{v:,.0f}"


def _e(v) -> str:
    return html.escape(str(v if v is not None else "–"))


def _decision(a: dict) -> tuple[str, str, str]:
    ap, dec = a.get("approval"), a.get("decision") or {}
    if ap:
        return ap["decision"], f"Approved by {ap.get('by', 'inspector')} {ap.get('at', '')}".strip(), ap.get("note", "")
    return dec.get("proposed", "–"), "Proposed by the agent, awaiting approval", dec.get("summary", "")


def to_html(folder: Path) -> str:
    d = load(folder)
    folder, r, a = d["folder"], d["result"], d["agent"]
    rk = a["risk"]
    cur = rk["repair_cost"]["currency"]
    findings = [f for f in (a.get("findings") or r.get("findings", [])) if f.get("status") != "dismissed"]
    findings.sort(key=lambda f: ({"confirmed": 0, "needs_review": 1}.get(f.get("status"), 2), -f.get("score", 0)))
    dismissed = [f for f in (a.get("findings") or []) if f.get("status") == "dismissed"]
    decision, who, note = _decision(a)
    cov = r["coverage"]
    integ = r.get("integrity") or {"level": "low", "flags": []}

    def row(*cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

    cov_rows = "".join(row(_e(k.title()), f"{v['seen_pct']}%", f"{v['inspectable_pct']}%", f"<b>{v['inspected_pct']}%</b>",
                           f"{v.get('area_m2', '–')} m²") for k, v in cov["by_kind"].items())
    gaps = "".join(f"<li>{_e(g['label'])}: " + (f"{g['length_m']} m of wall" if g.get("length_m") else
                                                 f"{g.get('uncovered_m2', '–')} m²") + " not inspectable</li>"
                   for g in r.get("gaps", [])[:6])
    refilm = "".join(f"<li><b>{_e(x['surface'])}</b>: {_e(x['instruction'])}</li>" for x in a.get("refilm", []))
    perils = "".join(row(_e(p["label"]), _e(p["peril"]), p["findings"], p["risk_points"],
                         f"{_money(p['repair_cost'][0], cur)} – {_money(p['repair_cost'][1], cur)}")
                     for p in rk["insurance"]["perils"]) or row("No defects counted", "", "", "", "")
    cards = []
    for f in findings[:24]:
        src = _img(folder, f.get("crop"), 360)
        size = f.get("size") or {}
        dims = ", ".join(x for x in [f"{size['length_m']:.2f} m long" if size.get("length_m") else "",
                                     f"{size['width_mm']:.1f} mm wide" if f.get("kind") == "crack" and size.get("width_mm")
                                     else "", f"{size['area_m2']:.2f} m²" if size.get("area_m2") else ""] if x)
        cards.append(f"""<div class="finding {f.get('status', 'candidate')}">{f'<img src="{src}" alt="">' if src else ''}
<div class="fbody"><div class="ftop"><b>{_e(f['id'])} · {_e(KIND_LABEL.get(f['kind'], f['kind']))}</b>
<span class="pill">{_e(STATUS_LABEL.get(f.get('status'), f.get('status')))}</span></div>
<div>{_e(f.get('surface', '').title())} · {_e(f.get('group'))} · {_e(dims or 'size not measured')}</div>
<div class="muted">OpenCV score {f.get('score', 0):.2f}{' · severity ' + _e(f['severity']) if f.get('severity') else ''}
{' · ' + _e(f.get('evidence') or f.get('reason')) if (f.get('evidence') or f.get('reason')) else ''}</div></div></div>""")
    rejected = r.get("rejected", [])
    why: dict[str, int] = {}
    for x in rejected:
        key = x["reason"].split(" (")[0]
        key = "on furniture or objects" if "in front" in key or "stands out" in key or "looks like" in key else \
              "on doors" if "door" in key else "seen through an opening" if "beyond" in key else key
        why[key] = why.get(key, 0) + 1
    rej = ", ".join(f"{n} {k}" for k, n in sorted(why.items(), key=lambda kv: -kv[1]))
    flags = "".join(f"<li class='{_e(f['level'])}'><b>{_e(f['level'].upper())}</b> {_e(f['message'])}</li>"
                    for f in integ.get("flags", [])) or "<li class='low'>No integrity concerns found.</li>"
    plan = _img(folder, (r.get("visuals") or {}).get("floor_plan"), 900)
    walls = "".join(f"<figure><img src='{s}' alt=''><figcaption>{_e(wl['id'])} · {wl['length_m']} m · seen "
                    f"{wl['seen_pct']}%</figcaption></figure>"
                    for wl in (r.get("visuals") or {}).get("walls", [])[:6] if (s := _img(folder, wl.get("image"), 700)))
    steps = [s for s in a.get("trace", []) if s.get("kind") in ("tool", "human")]
    trace = "".join(f"<li><code>{_e(s.get('tool'))}</code> {_e(json.dumps(s.get('input', {}))[:140])}</li>" for s in steps[:40])
    reasons = "".join(f"<li>{_e(x)}</li>" for x in rk["reasons"]) or "<li>No rule fired.</li>"
    val = rk["valuation"]
    labels = "".join(f"<li>frame {lab['frame']}: {_e(' / '.join(ln['text'] for ln in lab.get('lines', [])[:4]))}</li>"
                     for lab in r.get("labels", [])[:6])
    o = cov["overall"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inspection report · {_e(r['name'])}</title><style>
:root{{--ink:#17201d;--muted:#5d6b66;--line:#d9e0dc;--bg:#f6f8f7;--ok:#1f7a55;--warn:#b26a00;--bad:#b3261e;--accent:#0f6e5c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:28px 20px 60px}}h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 10px;
border-bottom:2px solid var(--ink);padding-bottom:4px}}.muted{{color:var(--muted);font-size:12.5px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.card{{background:#fff;border:1px solid var(--line);
border-radius:10px;padding:12px 14px}}.big{{font-size:26px;font-weight:700;line-height:1.1}}.lbl{{font-size:11.5px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)}}.dec{{border-left:6px solid var(--accent);background:#fff;border-radius:10px;padding:14px 16px;
margin:14px 0}}.dec .big{{color:var(--accent)}}table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line)}}
th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);font-size:13px}}th{{background:#eef2f0}}
.findings{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}}.finding{{display:flex;gap:10px;background:#fff;
border:1px solid var(--line);border-radius:10px;padding:8px}}.finding img{{width:110px;height:110px;object-fit:cover;border-radius:6px}}
.finding.confirmed{{border-left:5px solid var(--bad)}}.finding.needs_review{{border-left:5px solid var(--warn)}}.fbody{{font-size:12.5px}}
.ftop{{display:flex;justify-content:space-between;gap:6px}}.pill{{font-size:11px;background:#eef2f0;border-radius:99px;padding:1px 8px}}
ul{{margin:6px 0;padding-left:20px}}li.high{{color:var(--bad)}}li.medium{{color:var(--warn)}}figure{{margin:0}}
.plan{{text-align:center}}.plan img{{max-width:640px}}.walls{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}}.walls img,.plan img{{width:100%;border-radius:8px;
border:1px solid var(--line);background:#fff}}figcaption{{font-size:12px;color:var(--muted)}}code{{font-size:12px}}
.print{{position:fixed;right:16px;top:16px;background:var(--ink);color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer}}
@media print{{.print{{display:none}}body{{background:#fff}}h2{{break-after:avoid}}.finding,figure,tr{{break-inside:avoid}}}}
</style></head><body><button class="print" onclick="print()">Print / save as PDF</button><main>
<div class="lbl">Property inspection report · Team XCore · OpenCV {_e(r.get('opencv'))}</div>
<h1>{_e(r.get('property') or r['name'])}</h1>
<div class="muted">Walkthrough {_e(r['name'])} · {r['frames']} frames · report generated {time.strftime('%Y-%m-%d %H:%M')}</div>
<div class="dec"><div class="lbl">Decision</div><div class="big">{_e(decision)}</div><div>{_e(who)}</div>
<div class="muted">{_e(note)}</div><div class="muted">Underwriting rules: {_e(rk['decision'])}</div></div>
<div class="grid">
<div class="card"><div class="lbl">Condition grade</div><div class="big">{_e(rk['grade'])}</div><div>{_e(rk['grade_label'])} · score {rk['condition_score']}</div></div>
<div class="card"><div class="lbl">Risk level</div><div class="big">{_e(rk['risk_level'])}</div><div>premium loading {rk['insurance']['premium_loading_pct']}%</div></div>
<div class="card"><div class="lbl">Repair cost (indicative)</div><div class="big">{_money(rk['repair_cost']['low'], '')}–{_money(rk['repair_cost']['high'], '')}</div><div>{_e(cur)}</div></div>
<div class="card"><div class="lbl">Surface inspected</div><div class="big">{o['inspected_pct']}%</div><div>inspectable {o['inspectable_pct']}% · seen {o['seen_pct']}%</div></div>
<div class="card"><div class="lbl">Evidence integrity</div><div class="big">{_e(integ.get('level', 'low').title())}</div><div>{len(integ.get('flags', []))} flag(s)</div></div>
</div>
<h2>Why</h2><ul>{reasons}</ul>
<h2>Risk and valuation</h2><table>{row('Category', 'Insurance peril', 'Findings', 'Risk points', 'Repair cost', tag='th')}{perils}</table>
<p>Valuation impact {val['impact_pct'][0]}% to {val['impact_pct'][1]}% ({_e(val['basis'])}). Mortgage collateral risk
<b>{_e(rk['mortgage']['collateral_risk'])}</b>{', repair retention ' + _money(rk['mortgage']['repair_retention'], cur) if rk['mortgage']['repair_retention'] else ''}.</p>
<h2>Coverage: what was actually inspected</h2>
<table>{row('Surface', 'Seen', 'Inspectable', 'Inspected', 'Area', tag='th')}{cov_rows}</table>
<p class="muted">Inspectable: filmed within 3 m, at most 70° off straight-on, in a sharp frame. Inspected: inspectable in a
frame the analysis used. Below {60}% inspected the rules refer the case for more evidence.</p>
<div class="grid"><div><b>Largest gaps</b><ul>{gaps or '<li>None</li>'}</ul></div><div><b>Re-film requests</b><ul>{refilm or '<li>None</li>'}</ul></div></div>
{f'<div class="plan"><img src="{plan}" alt="Floor plan with findings and the camera path"></div>' if plan else ''}
<h2>Findings</h2>
<p class="muted">{len(findings)} on the building's surfaces ({sum(f.get('status') == 'confirmed' for f in findings)} confirmed,
{sum(f.get('status') == 'needs_review' for f in findings)} for review), {len(dismissed)} dismissed by the agent.
{len(rejected)} candidates were ruled out with 3D geometry and object recognition{': ' + _e(rej) if rej else ''}.</p>
<div class="findings">{''.join(cards) or '<p>No defects found on the inspected surfaces.</p>'}</div>
<h2>Walls, rectified onto their planes</h2><div class="walls">{walls or '<p class="muted">Not rendered.</p>'}</div>
<h2>Evidence integrity</h2><ul>{flags}</ul>
<p class="muted">Checked with OpenCV: perceptual hashes against earlier walkthroughs, hard cuts, frozen or looped frames,
screen moire and bezels. Privacy: {r.get('privacy', {}).get('faces', 0)} face(s) and
{r.get('privacy', {}).get('text_regions', 0)} text region(s) blurred before analysis.</p>
{f'<h2>Labels read</h2><ul>{labels}</ul>' if labels else ''}
<h2>How the agent worked</h2><p class="muted">Engine: {_e(a.get('engine', '–'))} · {len(steps)} tool calls</p><ol>{trace}</ol>
<h2>Disclaimer</h2><p class="muted">{_e(rk['disclaimer'])} Classical computer vision can mistake objects, shadows or
patterns for damage; findings marked "Needs review" must be checked by a person.</p>
</main></body></html>"""


def to_markdown(folder: Path) -> str:
    d = load(folder)
    r, a = d["result"], d["agent"]
    rk = a["risk"]
    cur = rk["repair_cost"]["currency"]
    decision, who, note = _decision(a)
    o = r["coverage"]["overall"]
    lines = [f"# Inspection report: {r.get('property') or r['name']}", "",
             f"Walkthrough `{r['name']}` · {r['frames']} frames · OpenCV {r.get('opencv')} · {time.strftime('%Y-%m-%d %H:%M')}",
             "", f"**Decision: {decision}** ({who}). {note}", "",
             f"- Condition grade **{rk['grade']}** ({rk['grade_label']}, score {rk['condition_score']})",
             f"- Risk level **{rk['risk_level']}**, premium loading {rk['insurance']['premium_loading_pct']}%",
             f"- Repair cost {cur} {rk['repair_cost']['low']:,} – {rk['repair_cost']['high']:,} (indicative)",
             f"- Underwriting rules: {rk['decision']}",
             f"- Inspected {o['inspected_pct']}% of surfaces (inspectable {o['inspectable_pct']}%, seen {o['seen_pct']}%)",
             f"- Evidence integrity: {(r.get('integrity') or {}).get('level', 'low')}", "", "## Why", ""]
    lines += [f"- {x}" for x in rk["reasons"]] or ["- No rule fired."]
    lines += ["", "## Coverage", "", "| Surface | Seen | Inspectable | Inspected |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {v['seen_pct']}% | {v['inspectable_pct']}% | {v['inspected_pct']}% |" for k, v in r["coverage"]["by_kind"].items()]
    if a.get("refilm"):
        lines += ["", "## Re-film requests", ""] + [f"- **{x['surface']}**: {x['instruction']}" for x in a["refilm"]]
    lines += ["", "## Findings", "", "| ID | Defect | Status | Surface | Size | Score |", "| --- | --- | --- | --- | --- | --- |"]
    for f in a.get("findings") or r.get("findings", []):
        if f.get("status") == "dismissed":
            continue
        s = f.get("size") or {}
        size = f"{s['area_m2']:.2f} m²" if s.get("area_m2") else f"{s['length_m']:.2f} m" if s.get("length_m") else "–"
        lines.append(f"| {f['id']} | {KIND_LABEL.get(f['kind'], f['kind'])} | {STATUS_LABEL.get(f.get('status'), '')} | "
                     f"{f.get('surface', '')} {f.get('group', '')} | {size} | {f.get('score', 0):.2f} |")
    flags = (r.get("integrity") or {}).get("flags", [])
    lines += ["", "## Evidence integrity", ""] + ([f"- {f['level'].upper()}: {f['message']}" for f in flags]
                                                 or ["- No integrity concerns found."])
    lines += ["", "---", rk["disclaimer"], ""]
    return "\n".join(lines)
