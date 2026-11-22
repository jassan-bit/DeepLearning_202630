"""Structural editorial checks only; no scientific execution or calculations."""
import copy
import importlib.util
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("academic_annotation_fixture", HERE / "annotate_academic_note.py")
note=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(note)


def fixture():
    return dict(metadata=dict(e3_source_pins=dict(source_commit=note.SOURCE_P1,
        manifest_sha256=note.MANIFEST_P1,bootstrap_sha256=note.BOOTSTRAP_P1),
        e3_monte_carlo_default=False),cells=[dict(cell_type="markdown",metadata={},source="V1"),
        dict(cell_type="code",metadata={},source="RUN_STATISTICAL_VALIDATION = False\n",execution_count=None,outputs=[])])


def test_annotation_preserves_code_pins_original_and_repeated_application():
    original=fixture();before=copy.deepcopy(original)
    result=note.annotate(original,"https://github.com/jassan-bit/DeepLearning_202630/blob/RESULTS/academic.json")
    assert original==before
    assert [c for c in result["cells"] if c["cell_type"]=="code"]==[c for c in original["cells"] if c["cell_type"]=="code"]
    assert result["metadata"]["e3_source_pins"]==original["metadata"]["e3_source_pins"]
    assert "accepted=false" in result["cells"][1]["source"]
    assert "94,8366 %" in result["cells"][1]["source"] and "18 límites" in result["cells"][1]["source"]
    assert note.annotate(result,"https://github.com/jassan-bit/DeepLearning_202630/blob/RESULTS/academic.json")==result


def test_other_scientific_source_is_rejected():
    book=fixture();book["metadata"]["e3_source_pins"]["source_commit"]="a"*40
    with pytest.raises(ValueError,match="V1/P1"):
        note.annotate(book)


def test_annotation_does_not_activate_disabled_execution():
    book=fixture();book["cells"][1]["source"]="RUN_STATISTICAL_VALIDATION = True\n"
    with pytest.raises(ValueError,match="RUN_STATISTICAL_VALIDATION"):
        note.annotate(book)
