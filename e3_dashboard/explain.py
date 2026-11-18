"""Exact black-box occlusion contrasts on physically coherent raw windows.

Each intervention replaces one complete daily OHLCV bar with its fit-only
reference bar. It preserves OHLC consistency, but is a sensitivity experiment,
not a causal attribution or proof of explanation fidelity.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np


def occlusion_explanation(raw, reference, predict, original_prediction):
    raw = np.asarray(raw, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64).copy()
    reference[:, :4] *= raw[-1, 3] / reference[-1, 3]
    started = perf_counter()
    records = []
    for i in range(7):
        changed = raw.copy()
        changed[i] = reference[i]
        counterfactual = float(predict(changed))
        records.append({"lag": i - 7, "label": f"t−{7-i}",
                        "prediction_with_reference_bar": counterfactual,
                        "effect": float(original_prediction - counterfactual)})
    return {"method": "occlusion_raw_window", "unit": "sigma7",
            "reference": "Ventana real de ajuste fijada por fecha; precios reescalados al último cierre de esta entrada, sin usar prueba.",
            "attributions": records, "latency_ms": (perf_counter() - started) * 1000,
            "interpretation": "Cambio al reemplazar una barra OHLCV completa por la referencia de ajuste. Se conserva coherencia de cada barra, pero puede alterarse la dinámica temporal; los contrastes no suman necesariamente la predicción ni identifican causas.",
            "fidelity_status": "contrastes exactos del predictor; fidelidad explicativa y estabilidad entre semillas pendientes"}
