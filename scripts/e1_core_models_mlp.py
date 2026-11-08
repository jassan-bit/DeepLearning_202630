"""New core registry; historical model implementations remain unchanged.

The MLP is a project adaptation of shallow feedforward volatility baselines,
with ReLU and a positive Softplus output matching the common target scaling.
It is not an exact replication of any cited paper's data or architecture.
"""
from dl_models import SmallMLP, SingleLayerLSTM, count_parameters
from dl_models import architecture_spec as base_spec
from dl_models_gru_lstm_tcn import SingleLayerGRU
from dl_models_gru_lstm_tcn import architecture_spec as recurrent_spec

FAMILIES = ("GRU", "LSTM", "MLP")


def build_model(family, history_days, width, dropout=.1, input_features=5):
    builders = {"GRU": SingleLayerGRU, "LSTM": SingleLayerLSTM, "MLP": SmallMLP}
    if family not in builders:
        raise ValueError("The new core admits exactly GRU, LSTM and MLP")
    return builders[family](history_days, width, dropout, input_features)


def architecture_spec(model):
    if isinstance(model, (SmallMLP, SingleLayerLSTM)):
        return base_spec(model)
    return recurrent_spec(model)
