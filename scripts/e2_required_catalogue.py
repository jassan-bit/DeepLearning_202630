"""Current mandatory E2 scope: ten candidates, three E1 references, three controls.

This prospective adapter rejects optional systems before any fit. It leaves the
historical registries and sealed Colab campaigns unchanged. The CLI exports
metadata only; it neither starts training nor migrates a campaign.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_PATH = ROOT / "experiments/e2_required_catalogue_20261112/catalogue.json"
VERSION = "e2-required-catalogue-16-v1"
TOP10 = (
    "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
)
E1_REFERENCES = ("MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
NEURAL = TOP10 + E1_REFERENCES
REQUIRED = NEURAL + CLASSICAL
OPTIONAL_HISTORICAL = ("CNN-1D + LSTM + atención temporal", "Ridge")
TOP10_FAMILIES = (
    "convolutional", "convolutional", "convolutional", "recurrent_autoencoder",
    "convolutional", "mlp_mixing", "mlp_mixing", "mlp_mixing",
    "transformer_attention", "transformer_attention",
)


def catalogue():
    """Return the versioned scope only after checking roles, IDs and families."""
    value = json.loads(CATALOGUE_PATH.read_text(encoding="utf-8"))
    rows = value.get("systems", [])
    expected_roles = ("Top10",) * 10 + ("E1_reference",) * 3 + ("classical_comparator",) * 3
    if (value.get("version") != VERSION
            or tuple(row.get("model_id") for row in rows) != REQUIRED
            or tuple(row.get("role") for row in rows) != expected_roles
            or tuple(row.get("family") for row in rows[:10]) != TOP10_FAMILIES
            or len(set(REQUIRED)) != 16 or len(set(TOP10_FAMILIES)) < 4
            or value.get("counts") != {"top10": 10, "top10_families": 4,
                "e1_references": 3, "neural": 13, "classical": 3, "total": 16}
            or tuple(row.get("model_id") for row in value.get("excluded_from_required_scope", []))
                != OPTIONAL_HISTORICAL):
        raise ValueError("The mandatory E2 catalogue differs from the registered sixteen systems")
    return deepcopy(value)


def require_models(names, *, neural_only=False):
    """Validate an explicit work list before constructing or training models."""
    catalogue()
    if isinstance(names, str):
        raise ValueError("Supply a sequence of model IDs")
    names = tuple(names)
    allowed = NEURAL if neural_only else REQUIRED
    if not names or len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError("Only unique mandatory E2 model IDs are allowed")
    return names


def build_required_neural_model(name, trial, **kwargs):
    require_models((name,), neural_only=True)
    if __package__:
        from . import e2_training_models as registry
    else:
        import e2_training_models as registry
    return registry.build_train_model(name, trial, **kwargs)


def run_required_neural_selection(prepared, name, adapter, **kwargs):
    """Keep the registered recipe/domains; exclude the extra hybrid before fits."""
    require_models((name,), neural_only=True)
    if __package__:
        from . import e2_training_bindings as bindings
    else:
        import e2_training_bindings as bindings
    return bindings.run_neural_selection(prepared, name, adapter, **kwargs)


def run_required_neural_refits(prepared, selection, adapter, seeds, **kwargs):
    """Check the winning system's scope before any fresh final refit."""
    require_models((selection.winner.payload["model"],), neural_only=True)
    if __package__:
        from . import e2_training_bindings as bindings
    else:
        import e2_training_bindings as bindings
    return bindings.run_selected_refits(prepared, selection, adapter, seeds, **kwargs)


def select_required_classical(panel, prepared, **kwargs):
    """Select ARIMA and the two naive rules, without evaluating Ridge."""
    catalogue()
    if "include_ridge" in kwargs:
        raise ValueError("Ridge is excluded from the mandatory catalogue")
    if __package__:
        from . import e2_required_classical_training as classical
    else:
        import e2_required_classical_training as classical
    return classical.select_classical(panel, prepared, include_ridge=False, **kwargs)


def refit_required_classical(panel, prepared, selection, **kwargs):
    """Reject historical four-control selections before any refit begins."""
    catalogue()
    if selection.ridge_alpha is not None or set(selection.selected_scores) != set(CLASSICAL):
        raise ValueError("Use a three-comparator selection without Ridge")
    if __package__:
        from . import e2_required_classical_training as classical
    else:
        import e2_required_classical_training as classical
    return classical.refit_classical(panel, prepared, selection, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new JSON metadata file; never overwrite")
    args = parser.parse_args(argv)
    value = catalogue()
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
