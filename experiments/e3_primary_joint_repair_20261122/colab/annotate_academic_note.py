"""Editorial annotation for notebook T; never edits published scientific P1.

The numerical V1 verdict remains unchanged. The academic rounding basis is a
teacher authorization reported by the user, not an independently verified act.
No statistical procedure, simulation, threshold or result is executed here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urlparse

SOURCE_P1 = "db5a75bb66a0b28b1d05dacc41b66daf161f9120"
MANIFEST_P1 = "45ad0706f293e487128905d6529d203aea14f8c5fc5a8907020781bd44a1c12f"
BOOTSTRAP_P1 = "6645af3889e356ec687ade7a20d63cd4cbfb8689cc4b91c63269aa367d619061"
ANNOTATION_ID = "academic-rounding-user-report-v1"


def annotate(book, record_url=None):
    expected = dict(source_commit=SOURCE_P1, manifest_sha256=MANIFEST_P1,
                    bootstrap_sha256=BOOTSTRAP_P1)
    if book.get("metadata", {}).get("e3_source_pins") != expected:
        raise ValueError("Annotate only the final notebook pinned to published scientific V1/P1")
    if book["metadata"].get("e3_monte_carlo_default") is not False:
        raise ValueError("Statistical execution must remain disabled by default")
    code_before = [copy.deepcopy(c) for c in book["cells"] if c["cell_type"] == "code"]
    if not any("RUN_STATISTICAL_VALIDATION = False" in "".join(c["source"])
               for c in code_before):
        raise ValueError("Keep RUN_STATISTICAL_VALIDATION=False")
    if record_url is not None:
        parsed = urlparse(record_url)
        if parsed.scheme != "https" or parsed.netloc not in ("github.com", "raw.githubusercontent.com") or parsed.query or parsed.fragment:
            raise ValueError("Use a public HTTPS academic-record link without credentials or parameters")
    text = """## Criterio de evaluación comunicado

El estudio V1 está completo. La CLI conserva `accepted=false` porque aplica el
criterio numérico original a los límites de cobertura sin redondear.

Según la autorización docente comunicada por el usuario, el registro académico
aplica el redondeo al porcentaje entero de forma coherente a los **18 límites**:
los dos valores **94,8366 % se presentan como 95 %** para esa evaluación.
Los resultados, intervalos y umbral del protocolo original se conservan.

Este cuaderno consulta y exporta el estudio V1 publicado. La ejecución de nuevas
simulaciones permanece desactivada. El registro de evaluación se distingue del
dictamen numérico y de una aprobación docente verificada independientemente.
"""
    if record_url:
        text += "\n[Consultar el registro académico y su alcance](" + record_url + ").\n"
    result = copy.deepcopy(book)
    note = dict(cell_type="markdown", metadata=dict(e3_annotation_id=ANNOTATION_ID), source=text)
    previous = [i for i,c in enumerate(result["cells"]) if c.get("metadata", {}).get("e3_annotation_id") == ANNOTATION_ID]
    if len(previous) > 1:
        raise ValueError("Duplicate academic annotations must be reviewed")
    if previous:
        result["cells"][previous[0]] = note
    else:
        result["cells"].insert(1, note)
    result["metadata"]["e3_academic_evaluation_note"] = dict(
        version=ANNOTATION_ID, source_commit=SOURCE_P1,
        underlying_numerical_verdict="accepted=false under the original unrounded criterion",
        basis="Teacher authorization communicated by the user; not independently verified",
        academic_rule="Nearest whole percentage applied consistently to all eighteen coverage bounds",
        scientific_code_protocol_thresholds_results_changed=False,
        new_statistical_execution=False, record_url=record_url)
    code_after = [c for c in result["cells"] if c["cell_type"] == "code"]
    if code_after != code_before or result["metadata"]["e3_source_pins"] != expected:
        raise AssertionError("Editorial annotation changed code or source pins")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--academic-record-url")
    args = parser.parse_args()
    original = json.loads(Path(args.input).read_text(encoding="utf-8"))
    value = annotate(original, args.academic_record_url)
    Path(args.output).write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
