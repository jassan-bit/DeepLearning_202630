"""Device portability checks for the actual fourteen-model pilot registry.

CPU tests exercise finite synthetic windows only. CUDA tests run the real CUDA
forward and backward, or are explicitly skipped when no CUDA device exists.
Nothing in this module loads market data or launches an HPO/NAS search.
"""
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest

# cuBLAS requires this before a CUDA context is initialized when deterministic
# matrix operations are requested. It is a documented test execution setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from torch import nn

from e2_development_models import build_pilot_model, reference_pattern
from e2_training_models import FIXED_COUNTS, NEURAL_NAMES
from e2_nas_factory import MODEL_IDS

SOURCE = Path(os.environ.get("E2_ITRANSFORMER_SOURCE_ROOT",
                            str(ROOT / "work/author_sources/iTransformer")))


def synthetic_windows():
    return torch.linspace(-.7, .8, 70, dtype=torch.float32).reshape(2, 7, 5)


def construct(name):
    return build_pilot_model(name, reference_pattern() if name in MODEL_IDS else None,
                             seed=274801, source_roots={"iTransformer": str(SOURCE)})


class _ModelAssertions:
    def require_source(self, name):
        if name == "iTransformer" and not SOURCE.is_dir():
            self.skipTest("The eight pinned iTransformer source files were not supplied")

    def assert_contract(self, name, model, metadata, keys, output, device_type):
        self.assertEqual(model.architecture_metadata, metadata)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.device.type, device_type)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output > 0).all()))
        self.assertTrue(all(value.device.type == device_type for value in model.parameters()))
        self.assertTrue(all(value.device.type == device_type for value in model.buffers()))
        if name in FIXED_COUNTS:
            self.assertEqual(sum(value.numel() for value in model.parameters()), FIXED_COUNTS[name])


class CPUModelPortabilityTests(_ModelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def check_model(self, name):
        self.require_source(name)
        model = construct(name)
        metadata, keys = deepcopy(model.architecture_metadata), tuple(model.state_dict())
        model.eval()
        with torch.no_grad():
            output = model(synthetic_windows())
        self.assert_contract(name, model, metadata, keys, output, "cpu")
        model.train()
        prediction = model(synthetic_windows())
        nn.functional.mse_loss(prediction, torch.tensor([.3, .4])).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None and bool(torch.isfinite(gradient).all())
                            for gradient in gradients))


@unittest.skipUnless(torch.cuda.is_available(), "A real CUDA device is required; CPU is not a CUDA validation")
class CUDAModelPortabilityTests(_ModelAssertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        state = cls.previous
        torch.use_deterministic_algorithms(state["deterministic"], warn_only=state["warn_only"])
        torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = state["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = state["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = state["cudnn_tf32"]
        torch.cuda.empty_cache()

    def check_model(self, name):
        self.require_source(name)
        model = construct(name)
        metadata, keys = deepcopy(model.architecture_metadata), tuple(model.state_dict())
        model.eval()
        cpu_x = synthetic_windows()
        with torch.no_grad():
            expected = model(cpu_x)
        model.to("cuda")
        cuda_x = cpu_x.to("cuda")
        model.eval()
        with torch.no_grad():
            output = model(cuda_x)
        torch.cuda.synchronize()
        self.assert_contract(name, model, metadata, keys, output, "cuda")
        # FP32 backends can round differently; this is numerical portability,
        # not a claim of bitwise CPU/CUDA identity or trained-score equality.
        torch.testing.assert_close(output.cpu(), expected, rtol=5e-4, atol=5e-5)
        model.train()  # cuDNN recurrent backward requires the training path.
        prediction = model(cuda_x)
        loss = nn.functional.mse_loss(prediction, torch.tensor([.3, .4], device="cuda"))
        loss.backward()
        torch.cuda.synchronize()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None and gradient.device.type == "cuda"
                            and bool(torch.isfinite(gradient).all()) for gradient in gradients))


def _case(name):
    def test(self):
        self.check_model(name)
    return test


for _index, _name in enumerate(NEURAL_NAMES):
    setattr(CPUModelPortabilityTests, "test_model_%02d" % _index, _case(_name))
    setattr(CUDAModelPortabilityTests, "test_model_%02d" % _index, _case(_name))


if __name__ == "__main__":
    unittest.main()
