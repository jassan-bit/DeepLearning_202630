"""Read-only name compatibility for the frozen E ARIMA prediction receipt.

E already uses ``Prediction.value`` for the forecast. Its receipt accidentally
asks for ``raw_value`` although the frozen dataclass calls that field ``raw``.
This adapter changes no number, truncation rule, state update, or fitted model.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import importlib
from pathlib import Path

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
VERSION = 'e2-arima-prediction-name-compat-v1'
ENGINE_FILES = {
    'scripts/e2_temporal_controls.py': '36e5a8b15e51ddf7d36397e2b6e71734afb36c6894cac0a71c9364b277fcc139',
    'scripts/e2_reduced_final.py': '033e4bf7db0d4f69624063830d116a1b5cf159a401c310a79979aa46307dad19',
}


def _raw_value(prediction):
    return prediction.raw


_RAW_VALUE = property(_raw_value, doc='Compatibility name for the original, unclipped raw field.')


def install_prediction_raw_alias(engine_root, controls=None):
    """Validate the exact E source and install one read-only class property."""
    root = Path(engine_root).absolute()
    for relative, expected in ENGINE_FILES.items():
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('The original E ARIMA source differs: ' + relative)
    controls = importlib.import_module('e2_temporal_controls') if controls is None else controls
    if Path(controls.__file__).resolve() != (root / 'scripts/e2_temporal_controls.py').resolve():
        raise ValueError('The loaded prediction contract is not the verified E module')
    prediction = controls.Prediction
    if (not isinstance(prediction, type) or not is_dataclass(prediction)
            or prediction.__module__ != controls.__name__
            or prediction.__qualname__ != 'Prediction'
            or [f.name for f in fields(prediction)] != ['date', 'raw', 'value', 'truncated']
            or prediction.__dataclass_params__.frozen is not True):
        raise ValueError('The frozen E prediction dataclass contract changed')
    previous = prediction.__dict__.get('raw_value')
    if previous is not None and previous is not _RAW_VALUE:
        raise ValueError('Another raw prediction attribute implementation is already active')
    if previous is None:
        setattr(prediction, 'raw_value', _RAW_VALUE)
    return {
        'version': VERSION, 'engine_commit': ENGINE_COMMIT,
        'original_source_files_sha256': dict(ENGINE_FILES),
        'prediction_class': 'e2_temporal_controls.Prediction',
        'compatibility_attribute': 'raw_value', 'original_attribute': 'raw',
        'alias_is_read_only': True, 'numeric_prediction_attribute': 'value',
        'numeric_predictions_changed': False, 'truncation_changed': False,
        'forecast_before_observation_changed': False, 'scientific_fit_repeated': False,
    }
