"""UNREGISTERED synthetic prototypes. Never imported by the registered runner.

Original compact regression implementations, not reproductions of cited papers.
Inputs are pre-scaled train-only tensors (batch, seven closed days, five OHLCV).
No loader, optimizer, HPO, pretraining scheduler, or test scoring lives here.
"""
import torch
from torch import nn
from torch.nn import functional as F


def check_x(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (7, 5):
        raise ValueError("Expected D7x5, no extra target-lag channels")
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite input")


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(incoming, outgoing, kernel,
                                         padding=kernel // 2), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(incoming, width, 3),
                                  nn.Conv1d(width, width, 3, padding=1))
        self.skip = nn.Conv1d(incoming, width, 1) if incoming != width else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class InceptionBlock(nn.Module):
    def __init__(self, incoming, width):
        super().__init__()
        self.bottleneck = nn.Conv1d(incoming, width, 1)
        self.branches = nn.ModuleList([ConvBlock(width, width, k) for k in (1, 3, 5)])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  ConvBlock(incoming, width, 1))

    def forward(self, x):
        z = self.bottleneck(x)
        return torch.cat([branch(z) for branch in self.branches] + [self.pool(x)], dim=1)


class ConvRegressor(nn.Module):
    def __init__(self, family, width=16):
        super().__init__()
        if family == 'FCN1D':
            self.features = nn.Sequential(ConvBlock(5, width, 3),
                                           ConvBlock(width, width, 3), ConvBlock(width, width, 1))
            out = width
        elif family == 'ResNet1D':
            self.features = nn.Sequential(ResidualBlock(5, width), ResidualBlock(width, width))
            out = width
        elif family == 'Inception1D':
            self.features = InceptionBlock(5, width)
            out = 4 * width
        else:
            raise ValueError(family)
        self.head = nn.Linear(out, 1)

    def forward(self, x):
        check_x(x)
        return F.softplus(self.head(self.features(x.transpose(1, 2)).mean(-1))).squeeze(-1)


class PredictiveLSTMAE(nn.Module):
    """Separate reconstruction and prediction primitives; no joint-loss claim.

    Decoder receives only the repeated final latent, never future observations.
    Pretrain reconstruction exclusively within each training fold, then finetune
    encoder/head on its supervised labels. Freeze policy remains unregistered.
    """
    def __init__(self, width=16):
        super().__init__()
        self.encoder = nn.LSTM(5, width, batch_first=True)
        self.decoder = nn.LSTM(width, width, batch_first=True)
        self.reconstruction = nn.Linear(width, 5)
        self.head = nn.Linear(width, 1)

    def latent(self, x):
        check_x(x)
        _, (h, _) = self.encoder(x)
        return h[-1]

    def reconstruct(self, x):
        z = self.latent(x)
        decoded, _ = self.decoder(z[:, None, :].expand(-1, 7, -1))
        return self.reconstruction(decoded)

    def forward(self, x):
        return F.softplus(self.head(self.latent(x))).squeeze(-1)


class HARResidual(nn.Module):
    """Signed correction of an externally supplied, temporally OOF HAR value.

    HAR has extended target-history support, not D7x5. Feed training corrections
    only the OOF records validated by e2_proposed_temporal. No base fit here.
    NOT READY FOR REAL-DATA TRAINING: clamp_min has zero gradient below zero.
    The current forward is a provisional inference link, not a training loss.
    A future protocol must specify fitting the signed correction to OOF raw
    residuals before applying the inference link (or justify another link).
    """
    def __init__(self, width=16):
        super().__init__()
        self.correction = nn.Sequential(nn.Flatten(), nn.Linear(35, width),
                                        nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x, har_prediction):
        check_x(x)
        if har_prediction.shape != (len(x),) or not torch.isfinite(har_prediction).all():
            raise ValueError("Expected one finite raw HAR forecast per origin")
        raw = har_prediction + self.correction(x).squeeze(-1)
        return raw.clamp_min(0)


def build_prototype(name, width=16):
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError("Positive integer width required")
    if name in ('FCN1D', 'ResNet1D', 'Inception1D'):
        return ConvRegressor(name, width)
    if name == 'LSTM-AE':
        return PredictiveLSTMAE(width)
    if name == 'HAR-residual':
        return HARResidual(width)
    raise ValueError("Unknown proposed architecture")
