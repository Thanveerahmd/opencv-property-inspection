"""Condition grade, risk, repair cost, valuation and underwriting decision from the inspection evidence.

Ported from the inspection platform's risk engine (v1/v2). Deterministic and auditable: every number comes from
RULES below, never from model probabilities, and every decision lists the rules that fired. Inputs are what the
OpenCV pipeline and the agent produced: findings with a status (confirmed / needs review / candidate / dismissed),
their kind, surface, size and severity; surface coverage; and the evidence-integrity flags.
"""

from __future__ import annotations

RULES: dict = {
    "version": 2,
    # how much a finding counts, by its status after review
    "status_weight": {"confirmed": 1.0, "needs_review": 0.5, "candidate": 0.2, "dismissed": 0.0},
    "severity_factor": {"minor": 0.5, "moderate": 1.0, "severe": 2.0, "unknown": 0.8},
    # category -> label, insurance peril, whether it drives collateral (mortgage) risk
    "categories": {
        "structural": {"label": "Structural", "peril": "Structural movement / subsidence", "collateral": True},
        "water": {"label": "Water & moisture", "peril": "Escape of water / damp", "collateral": True},
        "safety": {"label": "Safety", "peril": "Accidental damage / injury liability", "collateral": False},
        "fire": {"label": "Fire", "peril": "Fire / electrical", "collateral": True},
        "cosmetic": {"label": "Cosmetic / wear", "peril": "Wear and tear (usually excluded)", "collateral": False},
    },
    # defect -> category, risk points at severity factor 1.0, repair cost range (USD) at moderate severity
    "defects": {
        "crack": {"label": "Crack", "category": "structural", "points": 12, "cost": [250, 1500]},
        "water_damage": {"label": "Water damage", "category": "water", "points": 14, "cost": [400, 3000]},
        "mould": {"label": "Mould", "category": "water", "points": 14, "cost": [500, 3500]},
        "rust": {"label": "Rust", "category": "water", "points": 5, "cost": [80, 400]},
        "stain": {"label": "Stain", "category": "water", "points": 4, "cost": [60, 300]},
        "peeling": {"label": "Peeling paint", "category": "cosmetic", "points": 3, "cost": [80, 500]},
        "default": {"label": "Surface damage", "category": "cosmetic", "points": 3, "cost": [50, 300]},
    },
    # a stain this large (m²) is treated as water damage
    "water_damage_min_m2": 0.25,
    # condition score = 100 - sum(points x severity x status weight) / sqrt(views); grade by score
    "grades": [{"min": 85, "grade": "A", "label": "Good"}, {"min": 70, "grade": "B", "label": "Fair"},
               {"min": 50, "grade": "C", "label": "Below average"}, {"min": 30, "grade": "D", "label": "Poor"},
               {"min": 0, "grade": "E", "label": "Very poor"}],
    # indicative effect on market / collateral value by grade [low %, high %]
    "valuation_impact_pct": {"A": [0, 0], "B": [-1, -3], "C": [-3, -7], "D": [-7, -15], "E": [-15, -25]},
    "premium_loading_pct": {"Low": 0, "Moderate": 5, "High": 15, "Critical": 30},
    # evidence completeness: surface inspected below this -> refer for more evidence
    "min_inspected_pct": 60,
}

DISCLAIMER = ("Decision support only: indicative figures from configurable rules. A qualified surveyor or underwriter "
              "makes the final decision.")


def severity_of(f: dict) -> str:
    """Severity the agent gave, else one from the measured size."""
    sev = (f.get("severity") or "").lower()
    if sev in ("major", "severe"):
        return "severe"
    if sev in ("minor", "moderate"):
        return sev
    size = f.get("size") or {}
    if f.get("kind") == "crack":
        width = size.get("width_mm") or 0
        return "severe" if width >= 5 else "moderate" if width >= 1 else "unknown"
    area = size.get("area_m2") or 0
    return "severe" if area >= 1.0 else "moderate" if area >= 0.1 else "minor" if area > 0 else "unknown"


def defect_of(f: dict) -> str:
    kind = (f.get("defect") or f.get("kind") or "default").lower()
    kind = {"mold": "mould", "peeling_paint": "peeling"}.get(kind, kind)
    if kind == "stain" and ((f.get("size") or {}).get("area_m2") or 0) >= RULES["water_damage_min_m2"]:
        return "water_damage"
    return kind if kind in RULES["defects"] else "default"


def assess_result(findings: list[dict], result: dict, **kw) -> dict:
    """assess() with the evidence taken from a pipeline result: coverage, integrity, views and the room model."""
    layout = result.get("layout") or {}
    return assess(findings, result["coverage"]["overall"]["inspected_pct"], result.get("integrity"),
                  views=max(1, len(result.get("keyframes", []))),
                  model_issue=(f"the room model is uncertain (walls not found: {', '.join(layout['walls_not_found'])})"
                               if layout.get("confidence") == "low" else None), **kw)


def assess(findings: list[dict], inspected_pct: float | None = None, integrity: dict | None = None,
           views: int = 1, property_value: float | None = None, currency: str = "USD",
           cost_multiplier: float = 1.0, model_issue: str | None = None) -> dict:
    """Grade, risk level, perils, repair cost, valuation and the underwriting decision, with reasons.

    `model_issue` says why the 3D room model cannot be trusted; coverage measured on it is then not evidence."""
    R = RULES
    cats = {k: {**v, "points": 0.0, "findings": 0, "cost_low": 0.0, "cost_high": 0.0} for k, v in R["categories"].items()}
    items, deduction, cost_lo, cost_hi = [], 0.0, 0.0, 0.0
    for f in findings:
        status = f.get("status") or "candidate"
        w = float(R["status_weight"].get(status, 0.0))
        if w <= 0:
            continue
        defect = defect_of(f)
        rule = R["defects"][defect]
        sev = severity_of(f)
        k = float(R["severity_factor"][sev])
        pts = rule["points"] * k * w
        lo, hi = (c * k * w * cost_multiplier for c in rule["cost"])
        deduction += pts
        cost_lo += lo
        cost_hi += hi
        c = cats[rule["category"]]
        c["points"] += pts
        c["findings"] += 1
        c["cost_low"] += lo
        c["cost_high"] += hi
        items.append({"id": f.get("id"), "defect": defect, "label": rule["label"], "surface": f.get("surface"),
                      "group": f.get("group"), "severity": sev, "status": status, "category": rule["category"],
                      "risk_points": round(pts, 1), "repair_cost": [round(lo), round(hi)]})

    # a walkthrough is judged over all its views, so a room is not penalised for being filmed from many angles
    score = max(0.0, min(100.0, 100.0 - deduction / max(1.0, views ** 0.5)))
    grade = next(g for g in R["grades"] if score >= g["min"])
    severe_collateral = any(i["status"] == "confirmed" and i["severity"] == "severe"
                            and R["categories"][i["category"]]["collateral"] for i in items)
    level = "Low" if score >= 85 else "Moderate" if score >= 70 else "High" if score >= 50 else "Critical"
    if severe_collateral and level in ("Low", "Moderate"):
        level = "High"

    reasons = [f"{c['label']}: {c['findings']} finding(s), {c['points']:.1f} risk points"
               for c in cats.values() if c["findings"]]
    if severe_collateral:
        reasons.append("Severe structural, water or fire finding confirmed (collateral risk)")
    flags = list((integrity or {}).get("flags", []))
    high_flags = [f for f in flags if f.get("level") == "high"]
    evidence_ok = (inspected_pct is None or inspected_pct >= R["min_inspected_pct"]) and not model_issue
    if model_issue:
        reasons.append(f"Evidence incomplete: {model_issue}, so coverage cannot be confirmed")
    elif not evidence_ok:
        reasons.append(f"Evidence incomplete: only {inspected_pct:.0f}% of surfaces inspected "
                       f"(minimum {R['min_inspected_pct']}%)")
    if high_flags:
        reasons.append("Evidence integrity: " + "; ".join(f["message"] for f in high_flags))
    review = sum(i["status"] in ("needs_review", "candidate") for i in items)
    if review:
        reasons.append(f"{review} finding(s) still need a person's review")

    if high_flags:
        decision, action = "REFER - evidence integrity", "REFER"
    elif level == "Critical" or severe_collateral:
        decision, action = "REFER - professional survey required", "REPAIR"
    elif not evidence_ok:
        decision, action = "REFER - additional evidence required", "REFER"
    elif level == "High":
        decision, action = "PROCEED WITH CONDITIONS - repairs / retention", "REPAIR"
    elif level == "Moderate":
        decision, action = "PROCEED WITH CONDITIONS - note defects", "ACCEPT"
    else:
        decision, action = "PROCEED", "ACCEPT"

    vpct = R["valuation_impact_pct"][grade["grade"]]
    valuation = {"impact_pct": vpct, "basis": f"Grade {grade['grade']} ({grade['label']})"}
    if property_value:
        valuation["impact_amount"] = [round(property_value * p / 100) for p in vpct]
        valuation["adjusted_value"] = [round(property_value * (1 + vpct[1] / 100) - cost_hi),
                                       round(property_value * (1 + vpct[0] / 100) - cost_lo)]
    perils = sorted(({"category": k, "label": c["label"], "peril": c["peril"], "findings": c["findings"],
                      "risk_points": round(c["points"], 1), "repair_cost": [round(c["cost_low"]), round(c["cost_high"])]}
                     for k, c in cats.items() if c["findings"]), key=lambda p: -p["risk_points"])
    return {
        "condition_score": round(score, 1),
        "grade": grade["grade"], "grade_label": grade["label"],
        "risk_level": level,
        "decision": decision, "action": action,
        "reasons": reasons,
        "repair_cost": {"low": round(cost_lo), "high": round(cost_hi), "currency": currency},
        "valuation": valuation,
        "insurance": {"perils": perils, "premium_loading_pct": R["premium_loading_pct"][level]},
        "mortgage": {"collateral_risk": "High" if severe_collateral or level == "Critical"
                     else "Medium" if level == "High" else "Low",
                     "repair_retention": round(cost_hi * 1.2) if level in ("High", "Critical") else 0},
        "evidence": {"inspected_pct": inspected_pct, "complete": evidence_ok, "integrity_flags": flags},
        "findings": items,
        "rules_version": R["version"],
        "disclaimer": DISCLAIMER,
    }
