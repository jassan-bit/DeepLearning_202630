"""Independent §22 bodies; no historical models replaced, market fits or NAS claims.

Snapshots inspected 2026-11-09, architecture only (no author code imported):
 cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline,
 3ecc8971fa768bd01ada7f3a613688255e6256c2, FCN.py and ResNet.py;
 hfawaz/InceptionTime, 952d115e83fb3a66d75c5858c702f8dd5eeb18b7,
 classifiers/inception.py; 2M-kotb/LSTM-based-Stacked-Autoencoder,
 8668132ef6f8855dbab289c5a2b2478f871c4c4b, 2layer_evaluate.py.

Input is fold-scaled relative_window D7; output is y/a (caller fits positive a
on training labels and rescales once). No moments or downloaded weights.
BN uses population variance in both normalization and moving statistics,
epsilon .001 and retention .99: a declared non-fused Keras-style convention,
not bitwise parity with historical TensorFlow fused BN. Source square Conv2D
on a singleton spatial axis is translated to temporal Conv1D: unused weights,
fan-in and initialization distributions differ. Checkpoints are not portable.
LSTM uses i,f,g,o sigmoid/tanh gates and effective forget bias1; legacy CuDNN
serialization/fused rounding/random draws are not reproduced. Synthetic scope:
PyTorch 2.15.0+cpu float32/64, not GPU or mixed precision. Independent code.
"""
from __future__ import annotations
from copy import deepcopy
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_NAMES = ('FCN-1D', 'ResNet-1D', 'Inception-1D', 'LSTM-autoencoder con predictor')
RECIPE_IDS = dict(zip(MODEL_NAMES, ('fcn1d_wang_d7_v1_proposed',
    'resnet1d_wang_d7_v1_proposed', 'inception1d_single_d7_v1_proposed',
    'lstm_sae_25_26_d7_v1_proposed')))


def check_window(x: Tensor, channels: int = 5) -> None:
    if x.ndim != 3 or tuple(x.shape[1:]) != (7, channels) or x.shape[0] < 1:
        raise ValueError(f'Expected nonempty [B,7,{channels}] closed windows')
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError('Expected finite floating-point windows')


class PopulationBatchNorm1d(nn.Module):
    """Channel BN for [B,C,T], reducing B and T, including B=1."""
    def __init__(self, channels: int):
        super().__init__()
        self.eps, self.retention = .001, .99
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            var, mean = torch.var_mean(x, dim=(0, 2), correction=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 1 - self.retention)
                self.running_var.lerp_(var.detach(), 1 - self.retention)
                self.num_batches_tracked.add_(1)
        else:
            mean, var = self.running_mean, self.running_var
        return ((x - mean[None, :, None]) * torch.rsqrt(var[None, :, None] + self.eps)
                * self.weight[None, :, None] + self.bias[None, :, None])


class SameConv1d(nn.Conv1d):
    """Stride1 SAME cross-correlation, odd extra zero on the right."""
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, bias=bias)
        self.same_padding = ((kernel_size - 1) // 2, kernel_size // 2)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return super().forward(F.pad(x, self.same_padding))


def dense(inputs):
    layer = nn.Linear(inputs, 1)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def conv_bn(c_in, c_out, kernel, *, bias=True, relu=True):
    layers = [SameConv1d(c_in, c_out, kernel, bias=bias), PopulationBatchNorm1d(c_out)]
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FCNRecipe(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(conv_bn(5, 128, 8), conv_bn(128, 256, 5), conv_bn(256, 128, 3))
        self.head = dense(128)

    def forward(self, x):
        check_window(x)
        return F.softplus(self.head(self.blocks(x.transpose(1, 2)).mean(-1))).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, initial_bn=False):
        super().__init__()
        self.initial_bn = PopulationBatchNorm1d(c_in) if initial_bn else nn.Identity()
        self.main = nn.Sequential(conv_bn(c_in, c_out, 8), conv_bn(c_out, c_out, 5),
                                  conv_bn(c_out, c_out, 3, relu=False))
        # Pinned source compares each block with ORIGINAL input channels.
        self.shortcut = conv_bn(c_in, c_out, 1, relu=False)

    def forward(self, x):
        return F.relu(self.main(self.initial_bn(x)) + self.shortcut(x))


class ResNetRecipe(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(ResidualBlock(5, 64, True), ResidualBlock(64, 128),
                                    ResidualBlock(128, 128))
        self.head = dense(128)

    def forward(self, x):
        check_window(x)
        return F.softplus(self.head(self.blocks(x.transpose(1, 2)).mean(-1))).squeeze(-1)


def same_max_pool3(x):
    return F.max_pool1d(F.pad(x, (1, 1), value=-math.inf), 3, stride=1)


class InceptionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bottleneck = SameConv1d(channels, 32, 1, bias=False)
        self.branches = nn.ModuleList(SameConv1d(32, 32, k, bias=False) for k in (40, 20, 10))
        self.pool_projection = SameConv1d(channels, 32, 1, bias=False)
        self.bn = PopulationBatchNorm1d(128)

    def forward(self, x):
        shared = self.bottleneck(x)
        branches = [branch(shared) for branch in self.branches]
        branches.append(self.pool_projection(same_max_pool3(x)))
        return F.relu(self.bn(torch.cat(branches, dim=1)))


class InceptionRecipe(nn.Module):
    """One network, never a hidden five-member ensemble."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(InceptionModule(c) for c in (5, 128, 128, 128, 128, 128))
        self.shortcuts = nn.ModuleList((conv_bn(5, 128, 1, bias=False, relu=False),
                                       conv_bn(128, 128, 1, bias=False, relu=False)))
        self.head = dense(128)

    def forward(self, x):
        check_window(x)
        x = x.transpose(1, 2)
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                x = F.relu(x + self.shortcuts[i // 3](residual))
                residual = x
        return F.softplus(self.head(x.mean(-1))).squeeze(-1)


def recipe_lstm(inputs, hidden):
    layer = nn.LSTM(inputs, hidden, batch_first=True)
    # Transposes of Keras [inputs,4H] and [H,4H] kernels.
    nn.init.xavier_uniform_(layer.weight_ih_l0)
    nn.init.orthogonal_(layer.weight_hh_l0)
    nn.init.zeros_(layer.bias_ih_l0)
    nn.init.zeros_(layer.bias_hh_l0)
    with torch.no_grad():
        layer.bias_ih_l0[hidden:2 * hidden].fill_(1)
    return layer


def last_state(encoder, x):
    return encoder(x)[1][0][-1]  # Fresh zero h/c, never carried across windows.


def repeat7(x):
    return x.unsqueeze(1).expand(-1, 7, -1)


class SAERecipe(nn.Module):
    """Supervised/inference predictor with no decoders or reconstruction loss."""
    def __init__(self):
        super().__init__()
        self.encoder1, self.encoder2 = recipe_lstm(5, 25), recipe_lstm(25, 26)
        self.dropout1, self.dropout2 = nn.Dropout(.3), nn.Dropout(.3)
        self.head = dense(26)

    def forward(self, x):
        check_window(x)
        h1 = self.dropout1(last_state(self.encoder1, x))
        h2 = self.dropout2(last_state(self.encoder2, repeat7(h1)))
        return F.softplus(self.head(h2)).squeeze(-1)


class ReconstructionAE(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.encoder, self.decoder = recipe_lstm(inputs, hidden), recipe_lstm(hidden, 5)

    def forward(self, x):
        check_window(x, self.encoder.input_size)
        return self.decoder(repeat7(last_state(self.encoder, x)))[0]


class SAEPretraining:
    """Prepare fold-local AE1/AE2 explicitly; NEVER fit or choose phase budgets.

    Caller supplies TRAIN ONLY windows and provenance, accepts proposed X/s,
    fits ae1, calls prepare_phase2, fits ae2, then transfer_predictor. Both
    decoders reconstruct original train_x/s; encoder inputs remain unscaled.
    new_optimizer creates fresh optimizer state for each pretraining phase.
    Caller creates a separate optimizer for supervised fitting. Provenance is
    caller-supplied, not date-based certification of the train split.
    """
    def __init__(self, train_x, fold_id, *, accept_auxiliary_scale=False):
        if not accept_auxiliary_scale:
            raise ValueError('Auxiliary X/s requires explicit acceptance of the adaptation')
        check_window(train_x)
        if not isinstance(fold_id, str) or not fold_id.strip():
            raise ValueError('Nonempty training-fold provenance identifier required')
        self.fold_id = fold_id
        self._train_x = train_x.detach().clone()
        self.scale = max(1., float(self._train_x.abs().max()))
        self.ae1 = ReconstructionAE(5, 25).to(device=train_x.device, dtype=train_x.dtype)
        self.ae2 = ReconstructionAE(25, 26).to(device=train_x.device, dtype=train_x.dtype)
        self.phase, self._phase2_x = 'ae1', None

    def reconstruction_target(self, x):
        check_window(x)
        return x / self.scale  # Never refit or clip validation/test values.

    def phase1_data(self):
        return self._train_x.clone(), self.reconstruction_target(self._train_x).clone()

    def prepare_phase2(self):
        if self.phase != 'ae1':
            raise RuntimeError('Compute phase2 representations only once per fold')
        self.ae1.eval()
        self.ae1.requires_grad_(False)
        with torch.no_grad():
            self._phase2_x = repeat7(last_state(self.ae1.encoder, self._train_x)).clone()
        self.phase = 'ae2'
        return self.phase2_data()

    def phase2_data(self):
        if self.phase != 'ae2':
            raise RuntimeError('Prepare phase2 after fitting AE1 first')
        return self._phase2_x.clone(), self.reconstruction_target(self._train_x).clone()

    def new_optimizer(self, *, lr):
        module = self.ae1 if self.phase == 'ae1' else self.ae2
        return torch.optim.Adam(module.parameters(), lr=lr)

    def transfer_predictor(self):
        if self.phase != 'ae2':
            raise RuntimeError('Prepare/finalize reconstruction phases first')
        model = SAERecipe().to(device=self._train_x.device, dtype=self._train_x.dtype)
        model.encoder1, model.encoder2 = deepcopy(self.ae1.encoder), deepcopy(self.ae2.encoder)
        model.requires_grad_(True)
        model.train()
        return model


def build_model(name):
    factories = dict(zip(MODEL_NAMES, (FCNRecipe, ResNetRecipe, InceptionRecipe, SAERecipe)))
    if name not in factories:
        raise ValueError(f'Unknown reconstructed §22 model: {name}')
    return factories[name]()
