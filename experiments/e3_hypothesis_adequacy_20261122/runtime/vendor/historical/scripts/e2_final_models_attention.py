"""Published attention/SSM backbones adapted to the E2 D7 volatility target.

This own integration imports hash-pinned author sources without redistributing
those trees. Their licenses remain attached to the retrieved source trees.
The CPU Mamba recurrence follows state-spaces/mamba v1.2.0 (Apache-2.0),
Copyright (c) 2023 Tri Dao and Albert Gu. It implements its real, input-dependent
selective state-space equations, not an RNN substituted for Mamba.

Only synthetic and source-integrity checks belong here. No market file is read,
no weights are downloaded, and no training starts when this module is imported.
"""
from __future__ import annotations

from hashlib import sha256
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

MODELS = ("Informer", "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
# Populated from retrieved immutable source files; never populated from results.
SOURCE_SPECS = {'Informer': {'repository': 'zhouhaoyi/Informer2020',
              'commit': '29f2a739226a509202a092b464163da81fa74960',
              'preset_paths': ['main_informer.py', 'scripts/ETTh1.sh'],
              'files': {'models/model.py': 'bc8e062fbbb4f4ae970ca0d02d2bf43b2e92c3251c3627d887c01d35cbc6891a',
                        'models/encoder.py': 'bfd72a16e8bf7b2b36bcab47d198b885863e4b2ebfa5707f619c31078bc87737',
                        'models/decoder.py': 'ec2ebb81ba8d5400c06cb7db5ae2d93bb95dd04b2afbc33b5cefa284afb9a9df',
                        'models/attn.py': 'd9a294d829371759f1984f9f8fee12408774dda1a3de662313151d86e8513528',
                        'models/embed.py': 'd7f4f06370c640b56acab9df4092988f7711afb33cc64e868dc080286a0c4b9d',
                        'utils/masking.py': '33862345a8e9297b4214614e16c86829279953506319da4ffe33a476ef047c28',
                        'scripts/ETTh1.sh': '3550722a983d09f98d02d4f04bee2947ab7c7fb550b9697fb1d73398dc1391e1',
                        'main_informer.py': 'b35cf6c96a5c2b57600995a14865ddc0124b9c89b8e3f5184fa3c3e1a38e6736',
                        'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'}},
 'Autoformer': {'repository': 'thuml/Autoformer',
                'commit': '51c7d416ae120b805fd5beef2f4ccf7de496a6ff',
                'preset_paths': ['run.py', 'scripts/ETT_script/Autoformer_ETTh1.sh'],
                'files': {'models/Autoformer.py': 'f18e9e82844ed8c4dc26e00c3c2f0b7c278ac8ea8427c4fab7331939bf69829d',
                          'layers/Embed.py': '4133ae41fc64f5531cd001d2e883bd3ab89283ddea9ffdf5659960cc8bd37918',
                          'layers/AutoCorrelation.py': 'dddbc3cc533f9640bbee0e624a49532efe4ee99735101f51a4b03e105772af01',
                          'layers/Autoformer_EncDec.py': '512a01144804da2a64ade5c1e687b624a0257238da18ea02af0720479ea216e2',
                          'utils/masking.py': '02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef',
                          'scripts/ETT_script/Autoformer_ETTh1.sh': '94dc59bccbe97567633030fe0beca466d973849755992d239fc0c29d6f7a58fa',
                          'run.py': '60ec12b07d018830675d186f462f310eb3da831e2ec3d9d3160b319e4e6b3421',
                          'LICENSE': '8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2'}},
 'iTransformer': {'repository': 'thuml/iTransformer',
                  'commit': 'c2426e68ca13f74aaec08045c5c724d8ad328124',
                  'preset_paths': ['run.py', 'scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh'],
                  'files': {'model/iTransformer.py': '6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db',
                            'layers/Embed.py': '2de7e5049a696d320a23a8bdb6e08e823cf7fa3f246a0a49555fcd729c5c8398',
                            'layers/Transformer_EncDec.py': '985c31f4ae187b08afaa9663dc960a40544b2ec33ab55389ceab3930edc1ce02',
                            'layers/SelfAttention_Family.py': '38ac67428d528b8e145e4944ca059b40629a8446c891142602d5aaadc38dfd9e',
                            'utils/masking.py': '02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef',
                            'scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh': '81f8409cff60b754b45ed2e76133c7e3b6e585a50c8361f04fa3daf3b6997cc0',
                            'run.py': '650873cfd6fee1f0415d4e8fd69943ad736fd1bab6495b640aa27b1a769f59f9',
                            'LICENSE': '29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849'}},
 'Crossformer': {'repository': 'Thinklab-SJTU/Crossformer',
                 'commit': 'c10c8eadb153d1dd9798250967747ca3ebb81383',
                 'preset_paths': ['main_crossformer.py', 'scripts/ETTh1.sh'],
                 'files': {'cross_models/cross_former.py': 'deee243b2ae877084320dce49e691c8860c56f0cbda269c6f0cdb50848f6c708',
                           'cross_models/cross_encoder.py': 'be2fbf13317c27186563b68ff03326719a5a5db900b896889f46d570605587c2',
                           'cross_models/cross_decoder.py': '3c1d476df29d7a520d2ed3e8a06ec8e7186e96edda0ba74d6b943c8253fbdd40',
                           'cross_models/cross_embed.py': '6b78484268eb67f8fcf135537c3d61982c903992cc3f8f44fb47a46ec47c83fa',
                           'cross_models/attn.py': '39256d8f37e1acb3e52d645887316af89136d5910d4224c697c0cc38896f429b',
                           'main_crossformer.py': '3e2eec7ac04df696eb67968c981ecf99a694dcc92ce62e2a9b410a7e9722c5d7',
                           'scripts/ETTh1.sh': '3285202cae57c35bab6635054d1a3487423d1d365280fdbb1dff7767730850b4',
                           'LICENSE': '96c19fcecfadd5fc02b73d95dec507288f8d5b1724f1c8e6229fb21a987d8352'}},
 'S-Mamba': {'repository': 'wzhwzhwzh0921/S-D-Mamba',
             'commit': 'e7e8bf04066135afa43d85b0a87afa97cda16e3f',
             'preset_paths': ['run.py', 'scripts/multivariate_forecasting/ETT/S_Mamba_ETTh1.sh'],
             'files': {'model/S_Mamba.py': '48952b3906962917bc76345b5247ce350cae3243915142896b1952b4813b5fc9',
                       'layers/Embed.py': 'fa23f3680014bcc63674fba0eb3594cfa4c4f81882c5d2452936fda081ad49e0',
                       'layers/Mamba_EncDec.py': '8f89aefe9a44c59856e1743e66e3df7ec1d6bef110c3f6247cfdf6644e1af33d',
                       'layers/SelfAttention_Family.py': '38ac67428d528b8e145e4944ca059b40629a8446c891142602d5aaadc38dfd9e',
                       'utils/masking.py': '02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef',
                       'scripts/multivariate_forecasting/ETT/S_Mamba_ETTh1.sh': '72dc76618e76f9ff145b39ae3900aae30b301b64a4e07a65ecc6af9247dfae68',
                       'run.py': 'bb15489d1ec32560df328affac05467ef532291d9f76386bd162d5c0be6e874a',
                       'requirements.txt': '58fd1abde485724c5cc883907b3d476cca90ed3aba308badd60d75326a42608b'}},
 'MambaReference': {'repository': 'state-spaces/mamba',
                    'commit': '2845255a95f7e4a76fe649ec0532178fe3015dd0',
                    'preset_paths': ['mamba_ssm/modules/mamba_simple.py',
                                     'mamba_ssm/ops/selective_scan_interface.py'],
                    'files': {'mamba_ssm/modules/mamba_simple.py': 'dc035a4dcfbc9f9809e2bb60764f7bfe38259b6397968399fc33cad159300a43',
                              'mamba_ssm/ops/selective_scan_interface.py': 'fe606b4c7e81b47bb091cf59dc474aece1112a6ca01eb6f22309090e56030bc4',
                              'LICENSE': '760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5'},
                    'version_tag': 'v1.2.0'}}

STRUCTURES = {
    "Informer": {"d_model": 512, "d_ff": 2048, "encoder_stacks": [3, 1],
                 "encoder_heads": 16, "decoder_heads": 8, "decoder_cross_heads": 8,
                 "second_stack_context_divisor": 4, "d_layers": 2, "factor": 5,
                 "distil": True, "activation": "gelu", "decoder_label_len": 3},
    "Autoformer": {"d_model": 512, "d_ff": 2048, "e_layers": 2,
                   "d_layers": 1, "n_heads": 8, "factor": 3,
                   "moving_avg": 25, "activation": "gelu", "decoder_label_len": 3},
    "iTransformer": {"d_model": 256, "d_ff": 256, "e_layers": 2,
                     "n_heads": 8, "factor": 1, "activation": "gelu", "use_norm": True},
    "Crossformer": {"d_model": 256, "d_ff": 512, "e_layers": 3,
                    "decoder_stages": 4, "n_heads": 4, "seg_len": 6,
                    "merge_window": 2, "router_tokens": 10, "baseline": False},
    "S-Mamba": {"d_model": 256, "d_ff": 256, "e_layers": 2,
                "d_state": 2, "d_conv": 2, "expand": 1,
                "dt_rank": 16, "activation": "gelu", "use_norm": True,
                "forward_and_reverse_scans_per_block": 2},
}


def _dropout(candidate):
    if not isinstance(candidate, dict):
        raise TypeError("An explicit candidate dictionary is required")
    parameters = candidate.get("parameters", candidate)
    if not isinstance(parameters, dict):
        raise TypeError("Candidate parameters must be a dictionary")
    value = float(parameters.get("dropout", .1))
    if not math.isfinite(value) or not 0 <= value <= .3:
        raise ValueError("Dropout is outside the registered E2 search interval")
    return value


def _source_root(name, source_roots):
    if not isinstance(source_roots, dict) or name not in source_roots:
        raise ValueError("Explicit hash-pinned author source root required: " + name)
    root = Path(source_roots[name]).resolve()
    for relative, expected in SOURCE_SPECS[name]["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Author source missing or changed: " + name + "/" + relative)
    return root


def _load(name, source_roots, module):
    root = _source_root(name, source_roots)
    # Each worker builds one model. Constructor-time imports are isolated from
    # other author trees that reuse these generic top-level package names.
    for key in list(sys.modules):
        if any(key == p or key.startswith(p + ".") for p in
               ("model", "models", "layers", "utils", "cross_models")):
            del sys.modules[key]
    importlib.invalidate_caches()
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(root))


def _check(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (7, 5) or not len(x):
        raise ValueError("Expected a nonempty [B,7,5] D7 OHLCV window")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("D7 inputs must be finite floating-point values")


def _moments(x):
    return torch.cat((x.mean(dim=1), x.std(dim=1, unbiased=False)), dim=1)


def _normalize(x):
    mean = x.mean(dim=1, keepdim=True).detach()
    centered = x - mean
    return centered / torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5)


def _without_calendar(embedding, x, positional=True):
    value = embedding.value_embedding(x)
    if positional:
        value = value + embedding.position_embedding(x)
    return embedding.dropout(value)


class _ShortProbAttention(nn.Module):
    """Exact small-context limit of Informer's ProbSparse attention.

    With factor5 and D7, every query is retained. Computing the exact attention
    is algebraically identical to updating all queries, avoids unnecessary RNG,
    and defines the one-token log(1)=0 boundary. Longer contexts retain the
    original implementation. Its unused attention-dropout is not introduced.
    """
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, queries, keys, values, attn_mask=None):
        length, source_length = queries.shape[1], keys.shape[1]
        selected = self.original.factor * math.ceil(math.log(max(length, 1)))
        if length > 1 and source_length > 1 and selected < length:
            return self.original(queries, keys, values, attn_mask)
        scale = self.original.scale or queries.shape[-1] ** -.5
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        if self.original.mask_flag:
            if length != source_length:
                raise ValueError("Informer masked attention requires equal query/key lengths")
            mask = torch.ones(length, length, device=queries.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        probabilities = scores.softmax(dim=-1)
        result = torch.einsum("bhls,bshd->blhd", probabilities, values).contiguous()
        return result, probabilities if self.original.output_attention else None


def _patch_prob_attention(backbone):
    for layer in list(backbone.modules()):
        attn = getattr(layer, "inner_attention", None)
        if attn is not None and attn.__class__.__name__ == "ProbAttention":
            layer.inner_attention = _ShortProbAttention(attn)


def _base_metadata(name, dropout):
    spec = SOURCE_SPECS[name]
    return {"model": name, "history_days": 7, "input_channels": 5,
            "forecast_horizon_days": 1, "target": "sigma7 / training-only y_scale",
            "structure": STRUCTURES[name], "dropout": dropout,
            "author_repository": spec["repository"], "author_commit": spec["commit"],
            "author_source_files": spec["files"], "source_presets_inspected": spec["preset_paths"],
            "calendar_inputs": False, "target_lags_added": False,
            "same_window_moments": "five means and five population standard deviations of the same scaled D7",
            "output_link": "Softplus", "paper_replication_claimed": False,
            "original_OHLCV_denormalization_applied": False,
            "pretrained_weights": "none", "backbone_family_replaced": False,
            "configuration_choice_source": {
                "Informer": "paper main Table7 recipe; inspected ETTh1.sh is a different recipe, not claimed as the winner replicated here",
                "Autoformer": "official ETTh1 script factor3 plus original run.py defaults d512/ff2048/heads8",
                "iTransformer": "official ETTh1_96_96 script d256/ff256/enc2 plus run.py heads8",
                "Crossformer": "official ETTh1_24 script seg6 plus main_crossformer.py defaults d256/ff512/merge2/heads4/enc3",
                "S-Mamba": "official ETTh1_96_96 script d256/ff256/state2/enc2, original d_conv2/expand1"
            }[name]}


class AttentionVolatilityAdapter(nn.Module):
    def __init__(self, name, candidate, source_roots):
        super().__init__()
        self.family = name
        p = _dropout(candidate)
        metadata = _base_metadata(name, p)
        if name == "Informer":
            original = _load(name, source_roots, "models.model")
            self.backbone = original.InformerStack(5, 5, 5, 7, 3, 1, factor=5,
                d_model=512, n_heads=16, e_layers=[3, 1], d_layers=2, d_ff=2048,
                dropout=p, attn="prob", embed="timeF", freq="d", activation="gelu",
                output_attention=False, distil=True, mix=True, device=torch.device("cpu"))
            # Paper stack3+1 uses a quarter-context second branch. The author
            # constructor labels [0,1] as customizable; here use divisors1,4.
            self.backbone.encoder.inp_lens = [0, 2]
            self.backbone.decoder = original.Decoder([
                original.DecoderLayer(
                    original.AttentionLayer(original.ProbAttention(True, 5,
                        attention_dropout=p, output_attention=False), 512, 8, mix=True),
                    original.AttentionLayer(original.FullAttention(False, 5,
                        attention_dropout=p, output_attention=False), 512, 8, mix=False),
                    512, 2048, dropout=p, activation="gelu") for _ in range(2)],
                norm_layer=nn.LayerNorm(512))
            for emb, maximum in [(self.backbone.enc_embedding, 7), (self.backbone.dec_embedding, 4)]:
                del emb.temporal_embedding
                # Retain exactly the sinusoidal values used by this D7 task,
                # avoiding unused 5000-token checkpoint buffers.
                emb.position_embedding.pe = emb.position_embedding.pe[:, :maximum].clone()
            _patch_prob_attention(self.backbone)
            features = 5
            metadata["D7_changes"] = ["seq_len7,label_len3,pred_len1; observed context only",
                "paper quarter-context branch has one observed token",
                "exact ProbSparse all-query limit and defined single-token boundary",
                "remove calendar embedding; preserve both encoder stacks, distillation and full two-layer decoder",
                "trim unused sinusoidal-buffer tail after encoder7/decoder4; used positions unchanged",
                "keep decoder projection as five learned task features, without OHLCV interpretation"]
        elif name == "Autoformer":
            original = _load(name, source_roots, "models.Autoformer")
            args = SimpleNamespace(seq_len=7, label_len=3, pred_len=1, enc_in=5,
                dec_in=5, c_out=5, d_model=512, d_ff=2048, e_layers=2, d_layers=1,
                n_heads=8, factor=3, moving_avg=25, dropout=p, activation="gelu",
                embed="timeF", freq="d", output_attention=False)
            self.backbone = original.Model(args)
            for emb in (self.backbone.enc_embedding, self.backbone.dec_embedding):
                del emb.temporal_embedding
                # This source embedding class explicitly omits position values.
                del emb.position_embedding
            features = 5
            metadata["D7_changes"] = ["seq_len7,label_len3,pred_len1; no future observations",
                "remove calendar embedding and unused positional buffer; retain endpoint padding for moving_average25",
                "preserve encoder2, decoder1, all trend/seasonal decomposition and projections",
                "decoder output supplies five learned task features; scalar bridge is explicit"]
        elif name == "iTransformer":
            original = _load(name, source_roots, "model.iTransformer")
            args = SimpleNamespace(seq_len=7, pred_len=1, d_model=256, d_ff=256,
                e_layers=2, n_heads=8, factor=1, dropout=p, activation="gelu",
                embed="timeF", freq="d", output_attention=False, use_norm=True,
                class_strategy="projection")
            self.backbone = original.Model(args)
            del self.backbone.projector
            features = 5 * 256
            metadata["D7_changes"] = ["seq_len7 and five OHLCV variable tokens",
                "preserve original normalization and full256-wide two-layer inverted encoder",
                "replace per-covariate projector and its OHLCV denormalization with scalar task head"]
        elif name == "Crossformer":
            original = _load(name, source_roots, "cross_models.cross_former")
            self.backbone = original.Crossformer(data_dim=5, in_len=7, out_len=1,
                seg_len=6, win_size=2, factor=10, d_model=256, d_ff=512,
                n_heads=4, e_layers=3, dropout=p, baseline=False, device=torch.device("cpu"))
            features = 5
            metadata["D7_changes"] = ["preserve official ETTh1 seg_len6, CLIff512 and merge_window2",
                "author left-repeats first observed value: D7 padded to12 (two segments), no new observations",
                "retain full hierarchical encoder3 and additive four-stage decoder, not TSL encoder-only variant",
                "crop original out_len1 and use five learned task features in scalar bridge"]
        else:
            raise ValueError("Unsupported attention backbone: " + name)
        self.head = nn.Linear(features + 10, 1)
        metadata["readout_input_dim"] = features + 10
        metadata["readout"] = f"Linear({features+10},1) then Softplus"
        self.architecture_metadata = metadata

    def forward(self, x):
        _check(x)
        if self.family == "Informer":
            encoded = _without_calendar(self.backbone.enc_embedding, x)
            encoded, _ = self.backbone.encoder(encoded, attn_mask=None)
            known_decoder = torch.cat((x[:, -3:], x.new_zeros(len(x), 1, 5)), dim=1)
            decoded = _without_calendar(self.backbone.dec_embedding, known_decoder)
            decoded = self.backbone.decoder(decoded, encoded, x_mask=None, cross_mask=None)
            latent = self.backbone.projection(decoded[:, -1])
        elif self.family == "Autoformer":
            seasonal, trend = self.backbone.decomp(x)
            trend = torch.cat((trend[:, -3:], x.mean(dim=1, keepdim=True)), dim=1)
            seasonal = torch.cat((seasonal[:, -3:], x.new_zeros(len(x), 1, 5)), dim=1)
            encoded = _without_calendar(self.backbone.enc_embedding, x, positional=False)
            encoded, _ = self.backbone.encoder(encoded, attn_mask=None)
            decoded = _without_calendar(self.backbone.dec_embedding, seasonal, positional=False)
            seasonal_part, trend_part = self.backbone.decoder(decoded, encoded,
                x_mask=None, cross_mask=None, trend=trend)
            latent = (seasonal_part + trend_part)[:, -1]
        elif self.family == "iTransformer":
            embedded = self.backbone.enc_embedding(_normalize(x), None)
            encoded, _ = self.backbone.encoder(embedded, attn_mask=None)
            latent = encoded.reshape(len(x), -1)
        else:
            latent = self.backbone(x)[:, 0]
        joined = torch.cat((latent, _moments(x)), dim=1)
        return F.softplus(self.head(joined)).squeeze(-1)


def selective_scan_cpu(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=True, return_last_state=False):
    """Real variable-B/C selective scan, equivalent to Mamba's reference case.

    u/delta:[B,D,L], A:[D,N], B/C:[B,N,L]. The full scan is differentiable.
    This explicit CPU recurrence preserves learned input-dependent time steps,
    transitions, state input/output maps, skip D and SiLU gating.
    """
    if u.ndim != 3 or delta.shape != u.shape or A.ndim != 2:
        raise ValueError("Invalid selective-scan input shapes")
    batch, width, length = u.shape
    state_width = A.shape[1]
    if A.shape[0] != width or B.shape != (batch, state_width, length) or C.shape != B.shape:
        raise ValueError("Selective scan requires the registered real variable-B/C shapes")
    if A.is_complex() or not length:
        raise ValueError("Only the real nonempty S-Mamba recurrence is supported")
    dtype = u.dtype
    u, delta = u.float(), delta.float()
    A, B, C = A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = u.new_zeros(batch, width, state_width)
    output = []
    for index in range(length):
        step = delta[:, :, index]
        state = torch.exp(step[:, :, None] * A[None]) * state + (
            step[:, :, None] * B[:, None, :, index] * u[:, :, index, None])
        observed = (state * C[:, None, :, index]).sum(dim=-1)
        if D is not None:
            observed = observed + D.float()[None] * u[:, :, index]
        output.append(observed)
    result = torch.stack(output, dim=-1)
    if z is not None:
        result = result * F.silu(z.float())
    result = result.to(dtype)
    return (result, state) if return_last_state else result


class SelectiveStateSpaceCPU(nn.Module):
    """Mamba v1 real selective block, d_conv2 and expand1 as S-Mamba source."""
    def __init__(self, d_model=256, d_state=2, d_conv=2, expand=1):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv-1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state*2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(.1) - math.log(.001)) + math.log(.001)).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True
        self.A_log = nn.Parameter(torch.arange(1, d_state+1, dtype=torch.float32).log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        length = x.shape[1]
        signal, gate = self.in_proj(x).transpose(1, 2).chunk(2, dim=1)
        signal = F.silu(self.conv1d(signal)[..., :length])
        projected = self.x_proj(signal.transpose(1, 2))
        delta, coefficient_b, coefficient_c = torch.split(projected,
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.linear(delta, self.dt_proj.weight).transpose(1, 2)
        scanned = selective_scan_cpu(signal, delta, -self.A_log.exp(),
            coefficient_b.transpose(1, 2), coefficient_c.transpose(1, 2), self.D,
            gate, self.dt_proj.bias, delta_softplus=True)
        return self.out_proj(scanned.transpose(1, 2))


class _BidirectionalStateSpaceLayer(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.attention = SelectiveStateSpaceCPU()
        self.attention_r = SelectiveStateSpaceCPU()
        self.conv1 = nn.Conv1d(256, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.norm1, self.norm2 = nn.LayerNorm(256), nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        updated = self.attention(x) + self.attention_r(x.flip(1)).flip(1)
        x = self.norm1(x + updated)
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class SMambaVolatilityAdapter(nn.Module):
    def __init__(self, candidate, source_roots):
        super().__init__()
        _source_root("S-Mamba", source_roots)
        p = _dropout(candidate)
        self.family = "S-Mamba"
        self.embedding = nn.Linear(7, 256)
        self.embedding_dropout = nn.Dropout(p)
        self.layers = nn.ModuleList([_BidirectionalStateSpaceLayer(p) for _ in range(2)])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(5*256+10, 1)
        metadata = _base_metadata(self.family, p)
        metadata.update({"readout_input_dim": 1290, "readout": "Linear(1290,1) then Softplus",
            "SSM_reference": SOURCE_SPECS["MambaReference"],
            "SSM_execution": "differentiable FP32 CPU selective-scan reference recurrence; no fused CUDA kernel",
            "D7_changes": ["five OHLCV variable tokens, each with seven observed days",
                "preserve bidirectional variable scans, two residual blocks and FF256",
                "replace covariate projection/denormalization with scalar latent task head",
                "omit source registered man/man2/a modules never used by its forward method"],
            "license_of_original_S_Mamba_repository": "not verified; own math integration does not redistribute that code",
            "Mamba_reference_license": "Apache-2.0; Copyright2023 TriDao/AlbertGu"})
        self.architecture_metadata = metadata

    def forward(self, x):
        _check(x)
        hidden = self.embedding_dropout(self.embedding(_normalize(x).transpose(1, 2)))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden).reshape(len(x), -1)
        return F.softplus(self.head(torch.cat((hidden, _moments(x)), dim=1))).squeeze(-1)


def build_model(name, *, candidate, source_roots=None):
    if name not in MODELS:
        raise ValueError("Unsupported attention/SSM model: " + str(name))
    if isinstance(candidate, dict):
        if candidate.get("model", name) != name:
            raise ValueError("Candidate belongs to a different model")
        parameters = candidate.get("parameters", candidate)
        for field in ("d_model", "d_ff", "e_layers", "n_heads", "d_state", "d_conv", "seg_len"):
            if field in parameters and parameters[field] != STRUCTURES[name].get(field):
                raise ValueError("Candidate cannot silently replace the published structure: " + field)
    return SMambaVolatilityAdapter(candidate, source_roots) if name == "S-Mamba" else AttentionVolatilityAdapter(name, candidate, source_roots)
