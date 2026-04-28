"""Smoke tests for the per-venue review-field normalizer."""
from paper_reviewer.normalize import normalize_decision, normalize_review


def _v(x):
    return {"value": x}


def test_iclr_style():
    note = {
        "id": "rev1",
        "content": {
            "summary": _v("A summary."),
            "strengths": _v("Strong baselines."),
            "weaknesses": _v("Missing ablations on dataset X."),
            "questions": _v("How does this scale?"),
            "rating": _v(7),
            "confidence": _v(4),
            "soundness": _v(3),
            "presentation": _v(3),
            "contribution": _v(3),
        },
    }
    nr = normalize_review("ICLR.cc/2025/Conference", "abc", note)
    assert nr.rating == 7.0
    assert nr.confidence == 4.0
    assert nr.weaknesses.startswith("Missing")
    assert nr.soundness == 3.0


def test_neurips_recommendation():
    note = {
        "id": "rev2",
        "content": {
            "paper_summary": _v("Different field name."),
            "weaknesses": _v("Too few seeds."),
            "recommendation": _v("6: marginally above the acceptance threshold"),
            "confidence": _v(3),
        },
    }
    nr = normalize_review("NeurIPS.cc/2024/Conference", "abc", note)
    assert nr.rating == 6.0
    assert nr.summary == "Different field name."


def test_old_5_point_rescale():
    # If neither `rating` nor `recommendation` is present and we got a 1-5
    # `overall_rating`, we rescale to 1-10.
    note = {"id": "x", "content": {"overall_rating": _v(4)}}
    nr = normalize_review("Workshop/2023", "x", note)
    assert nr.rating == 8.0


def test_decision_accept():
    d = normalize_decision({"content": {"decision": _v("Accept (poster)")}})
    assert d.accepted is True


def test_decision_reject():
    d = normalize_decision({"content": {"decision": _v("Reject")}})
    assert d.accepted is False


def test_decision_missing():
    assert normalize_decision(None).accepted is None
