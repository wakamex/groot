import json

import pytest

from groot1 import data


def fake_load(rows_by_split):
    return lambda hf_id, config=None, split=None, streaming=False: iter(rows_by_split.get(split, []))


def test_typed_synth_uses_soft_teacher_and_maps_train_to_test(monkeypatch):
    case = {"state_id": "s1", "state": "plain text state",
            "questions": json.dumps({"q": {"type": "choice", "instructions": "pick", "criteria": {"a": "A.", "b": "B."}},
                                     "n": {"type": "noul", "instructions": "true?"}}),
            "teacher": json.dumps({"q": {"probabilities": {"a": 0.25, "b": 0.75}}, "n": {"noul": 0.9}}),
            "gold": json.dumps({"q": "b", "n": True})}
    monkeypatch.setattr(data, "_load", fake_load({"train": [case], "validation": []}))
    rows = list(data.conv_typed_synth())
    assert [r["split"] for r in rows] == ["test", "test"]
    assert rows[0]["state"] == "plain text state" and rows[0]["target"] == [0.25, 0.75]
    assert rows[1]["options"] == ["false", "true"] and rows[1]["target"] == pytest.approx([0.1, 0.9])


def test_synth_deepseek_ignores_jev(monkeypatch):
    item = {"id": "w/1", "state": "note",
            "questions": json.dumps({"c": {"type": "choice", "instructions": "next?", "criteria": {"x": "X", "y": "Y"}},
                                     "s": {"type": "score", "instructions": "how bad?", "criteria": ["low", "mid", "high"]}}),
            "answers": json.dumps({"c": {"probabilities": {"x": 0.5, "y": 0.5}}}),
            "teachers": json.dumps({"c": {"typesafe/jev-1.13": [1.0, 0.0], "deepseek-v4-flash": [0.2, 0.8]},
                                    "s": {"typesafe/jev-1.13": [0, 0, 1], "deepseek-v4-flash": [0.6, 0.3, 0.1]}})}
    monkeypatch.setattr(data, "_load", fake_load({"test": [item]}))
    rows = list(data.conv_synth_deepseek())
    assert [r["target"] for r in rows] == [pytest.approx([0.2, 0.8]), pytest.approx([0.6, 0.3, 0.1])]


def test_calibrated_td_skips_trained_sources_and_keeps_heldout_hard(monkeypatch):
    def row(i, source, soft):
        return {"id": f"r{i}", "source": source, "type": "choice", "instructions": "which?", "state": f"text {i}",
                "options": ["a", "b"], "option_descriptions": ["first", "second"], "gold_index": 1, "soft_label": soft}
    monkeypatch.setattr(data, "_load", fake_load({
        "validation": [row(0, "clinc/clinc_oos", [0.5, 0.5]), row(1, "allenai/scicite", [0.3, 0.7])],
        "test": [],
        "heldout": [row(2, "pubmedqa", None)]}))
    rows = list(data.conv_calibrated_td())
    assert [(r["source"], r["split"]) for r in rows] == [("calibrated_td", "val"), ("pubmedqa_heldout", "test")]
    assert rows[0]["options"] == ["a: first", "b: second"] and rows[0]["target"] == [0.3, 0.7]
    assert rows[1]["target"] == [0.0, 1.0]
