import numpy as np

from spatial_inspection.pipeline import _inside, _xywh
from spatial_inspection.walkthrough import annotate, place_pin_tips, summary_text


def test_pin_tips_leave_the_surface_and_do_not_collide():
    fs = [{"xyz": [1.0, 1.2, 0.0], "normal": [0, 0, 1]}, {"xyz": [1.05, 1.2, 0.0], "normal": [0, 0, 1]},
          {"xyz": None}]
    place_pin_tips(fs)
    a, b = np.array(fs[0]["tip"]), np.array(fs[1]["tip"])
    assert a[2] > 0.3 and b[2] > 0.3                 # out of the wall along its normal
    assert np.linalg.norm(a - b) >= 0.28             # heads fanned apart
    assert fs[2]["tip"] is None


def test_inside_share():
    assert _inside((10, 10, 10, 10), [0, 0, 100, 100]) == 1.0
    assert _inside((90, 90, 20, 20), [0, 0, 100, 100]) == 0.25
    assert _inside((200, 200, 5, 5), [0, 0, 100, 100]) == 0.0
    assert _xywh([1, 2, 4, 8]) == (1, 2, 3, 6)


def test_annotate_draws_boxes():
    img = np.full((120, 160, 3), 200, np.uint8)
    out = annotate(img, [{"bbox": [20, 30, 40, 40], "state": "DETECTED", "n": 1, "defect": "Crack"}])
    assert out.shape == img.shape and not np.array_equal(out, img)
    assert np.array_equal(img, np.full((120, 160, 3), 200, np.uint8))   # input untouched


def test_summary_mentions_grade_findings_and_gaps():
    risk = {"grade": "B", "grade_label": "Fair", "condition_score": 81.0, "risk_level": "Moderate",
            "decision": "PROCEED WITH CONDITIONS"}
    stats = {"overall": {"seen_pct": 70.0, "inspectable_pct": 55.0, "inspected_pct": 48.0}}
    findings = [{"n": 1, "defect": "Crack", "surface_label": "Wall 2", "component": "Wall", "severity": "moderate",
                 "state": "DETECTED"}]
    text = summary_text("x", risk, stats, findings, [{"label": "Ceiling", "uncovered_m2": 3.1}], {"level": "low", "flags": []})
    assert "Grade B" in text and "#1 Crack on the Wall 2" in text and "Ceiling" in text
