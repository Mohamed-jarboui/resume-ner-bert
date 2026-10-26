"""
Fast tests for the parts of the pipeline that do not need the BERT weights:
data generation, label alignment, metrics, post-processing and section detection.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_generation.generator import ResumeNERDataGenerator
from src.evaluation.metrics import NERMetrics
from src.inference.post_processing import SkillFilter
from src.preprocessing.sections import identify_section, segment_resume
from src.preprocessing.tokenizer import NERTokenizer

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    LABEL2ID = yaml.safe_load(f)["labels"]["label2id"]
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ---------- Data generation ----------

@pytest.fixture(scope="module")
def samples():
    random.seed(0)
    return ResumeNERDataGenerator().generate_samples(500)


def test_tokens_and_tags_have_same_length(samples):
    for s in samples:
        assert len(s["tokens"]) == len(s["tags"]) > 0


def test_all_tags_are_known_labels(samples):
    for s in samples:
        assert set(s["tags"]) <= set(LABEL2ID)


def test_tags_follow_bio_scheme(samples):
    for s in samples:
        previous = "O"
        for tag in s["tags"]:
            if tag.startswith("I-"):
                assert previous[2:] == tag[2:], f"{tag} after {previous} in {s['tokens']}"
            previous = tag


def test_generation_is_reproducible_with_seed():
    random.seed(42)
    first = ResumeNERDataGenerator().generate_samples(20)
    random.seed(42)
    second = ResumeNERDataGenerator().generate_samples(20)
    assert first == second


def test_label_mappings_match_config():
    with open(ROOT / "models/checkpoints/bert_crf/label_mappings.json", encoding="utf-8") as f:
        mappings = json.load(f)
    assert mappings["label2id"] == LABEL2ID


# ---------- Subword label alignment ----------

def test_only_first_subword_gets_the_label():
    tokenizer = NERTokenizer.__new__(NERTokenizer)  # skip loading the HF tokenizer
    # [CLS] Py ##thon dev [SEP]  ->  word ids: None 0 0 1 None
    aligned = tokenizer.align_labels_with_tokens([11, 0], [None, 0, 0, 1, None])
    assert aligned == [-100, 11, -100, 0, -100]


# ---------- Metrics ----------

def _ids(tags):
    return [LABEL2ID[t] for t in tags]


def test_perfect_prediction_scores_one():
    labels = np.array([_ids(["B-SKILL", "O", "B-DEGREE", "I-DEGREE"])])
    result = NERMetrics(ID2LABEL).compute_metrics(labels.copy(), labels)
    assert result["f1"] == pytest.approx(1.0)


def test_missed_entity_lowers_recall_only():
    labels = np.array([_ids(["B-SKILL", "O", "B-SKILL"])])
    preds = np.array([_ids(["B-SKILL", "O", "O"])])
    result = NERMetrics(ID2LABEL).compute_metrics(preds, labels)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)


def test_ignored_positions_are_skipped():
    labels = np.array([[-100, LABEL2ID["B-NAME"], -100, LABEL2ID["I-NAME"], -100]])
    preds = np.array([[0, LABEL2ID["B-NAME"], 0, LABEL2ID["I-NAME"], 0]])
    result = NERMetrics(ID2LABEL).compute_metrics(preds, labels)
    assert result["f1"] == pytest.approx(1.0)


# ---------- Post-processing ----------

def _ent(etype, text):
    return {"type": etype, "entity": text}


def test_skill_filter_removes_false_skills():
    entities = [
        _ent("SKILL", "Python"),
        _ent("SKILL", "john@mail.com"),
        _ent("SKILL", "2024"),
        _ent("SKILL", "excellent"),
        _ent("SKILL", "x"),
        _ent("SKILL", "C"),
        _ent("NAME", "Sarah Johnson"),
    ]
    kept = [e["entity"] for e in SkillFilter().filter_entities(entities)]
    assert kept == ["Python", "C", "Sarah Johnson"]


def test_normalize_results_groups_and_deduplicates():
    entities = [
        _ent("NAME", "Sarah Johnson"),
        _ent("EMAIL", "sarah@tech.com"),
        _ent("SKILL", "Docker"),
        _ent("SKILL", "docker"),
        _ent("DEGREE", "Bachelor of Computer Science"),
        _ent("EXPERIENCE", "Software Engineer"),
        _ent("DATE", "2020"),
    ]
    out = SkillFilter().normalize_results(entities)
    assert out["name"] == "Sarah Johnson"
    assert out["email"] == "sarah@tech.com"
    assert out["skills"] == ["Docker"]
    assert out["degrees"] == ["Bachelor of Computer Science"]
    assert out["experience"] == ["Software Engineer", "2020"]


# ---------- Section detection ----------

@pytest.mark.parametrize(
    "line, expected",
    [
        ("EDUCATION", "EDUCATION"),
        ("Formation:", "EDUCATION"),
        ("Compétences", "SKILLS"),
        ("Experience", "EXPERIENCE"),
        ("I have experience with Python and Docker", None),
    ],
)
def test_identify_section(line, expected):
    assert identify_section(line) == expected


def test_segment_resume_splits_on_headers():
    text = "Sarah Johnson\nSKILLS\nPython, Docker\nEDUCATION\nBSc Computer Science"
    sections = segment_resume(text)
    assert sections["HEADER"].strip() == "Sarah Johnson"
    assert sections["SKILLS"].strip() == "Python, Docker"
    assert sections["EDUCATION"].strip() == "BSc Computer Science"
