from spatial_inspection import risk


def crack(status="confirmed", severity="moderate", **kw):
    return {"id": "F1", "kind": "crack", "status": status, "severity": severity, "surface": "wall", **kw}


def test_clean_well_filmed_room_proceeds_with_grade_a():
    a = risk.assess([], inspected_pct=75)
    assert a["grade"] == "A" and a["risk_level"] == "Low" and a["decision"] == "PROCEED" and a["action"] == "ACCEPT"
    assert a["repair_cost"] == {"low": 0, "high": 0, "currency": "USD"}


def test_low_coverage_refers_for_more_evidence():
    a = risk.assess([], inspected_pct=42)
    assert a["decision"] == "REFER - additional evidence required" and not a["evidence"]["complete"]


def test_severe_confirmed_crack_needs_a_survey_and_raises_collateral_risk():
    a = risk.assess([crack(severity="major")], inspected_pct=80)
    assert a["decision"] == "REFER - professional survey required" and a["mortgage"]["collateral_risk"] == "High"
    assert a["insurance"]["perils"][0]["category"] == "structural"


def test_high_integrity_flag_wins():
    flags = {"flags": [{"code": "footage_other_property", "level": "high", "message": "reused"}]}
    a = risk.assess([], inspected_pct=90, integrity=flags)
    assert a["decision"] == "REFER - evidence integrity" and a["action"] == "REFER"


def test_status_weights_and_dismissed_findings():
    full = risk.assess([crack()], inspected_pct=80)["condition_score"]
    half = risk.assess([crack(status="needs_review")], inspected_pct=80)["condition_score"]
    none = risk.assess([crack(status="dismissed")], inspected_pct=80)["condition_score"]
    assert none == 100 and 100 - half == (100 - full) / 2


def test_large_stain_counts_as_water_damage_and_property_value_is_used():
    stain = {"id": "F2", "kind": "stain", "status": "confirmed", "severity": "moderate", "size": {"area_m2": 0.6}}
    a = risk.assess([stain], inspected_pct=80, property_value=200_000)
    assert a["findings"][0]["defect"] == "water_damage"
    assert a["valuation"]["adjusted_value"][1] <= 200_000
