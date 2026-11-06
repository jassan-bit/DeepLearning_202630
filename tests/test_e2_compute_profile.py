"""Synthetic arithmetic and timing-policy checks; no market files or model fits."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from torch import nn
import torch.nn.functional as F

from e2_compute_profile import (profile_model, measure_throughput,
                               measure_throughput_block, summarize_throughput_blocks)


class Function(nn.Module):
    def __init__(self, function):
        super().__init__(); self.function = function

    def forward(self, x):
        return self.function(x)


class ComputeProfileTests(unittest.TestCase):
    def test_linear_mac2_bias_and_per_sample(self):
        layer = nn.Linear(3, 2, bias=True)
        report = profile_model(layer, torch.ones(4, 3))
        self.assertEqual(report["parameters"], 8)
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertEqual(report["estimated_flops_total"], 56)
        self.assertEqual(report["estimated_flops_per_sample"], 14)

    def test_convolution_groups_and_bias(self):
        # 4 input channels,2groups =>2channels/kernel,3taps; output[1,6,5].
        layer = nn.Conv1d(4, 6, 3, groups=2, bias=True)
        report = profile_model(layer, torch.ones(1, 4, 7))
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertEqual(report["estimated_flops_total"], 30*(2*2*3+1))
        self.assertEqual(report["parameters"], 42)

    def test_complex_linear_uses_real_scalar_complex_mac_convention(self):
        layer = nn.Linear(3, 2, bias=True, dtype=torch.complex64)
        report = profile_model(layer, torch.ones(1, 3, dtype=torch.complex64))
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertEqual(report["estimated_flops_total"], 8*2*3+2*2)

    def test_reduction_and_normalization_explicit_convention(self):
        mean = profile_model(Function(lambda x: x.mean(dim=-1)), torch.ones(2, 3))
        self.assertEqual(mean["estimated_flops_total"], 6)
        norm = profile_model(nn.LayerNorm(3), torch.ones(2, 3))
        self.assertEqual(norm["coverage"], "complete_under_convention")
        self.assertEqual(norm["estimated_flops_total"], 7*6+2*2)
        softmax = profile_model(nn.Softmax(dim=-1), torch.ones(2, 3))
        self.assertEqual(softmax["estimated_flops_total"], 5*6-2*2)

    def test_eval_batchnorm_and_training_flags_restored(self):
        layer = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
        layer.train(); layer[0].eval()
        old_mean = layer[1].running_mean.clone()
        report = profile_model(layer, torch.ones(2, 3))
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertTrue(layer.training); self.assertFalse(layer[0].training)
        self.assertTrue(layer[1].training)
        torch.testing.assert_close(old_mean, layer[1].running_mean)

    def test_binary_einsum_and_bmm(self):
        weights = torch.ones(1, 3, 4)
        model = Function(lambda x: torch.einsum("bik,bkj->bij", x, weights))
        report = profile_model(model, torch.ones(1, 2, 3))
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertEqual(report["estimated_flops_total"], 2*1*2*3*4)
        self.assertTrue(any("bmm" in name or "einsum" in name for name in report["operation_counts"]))

    def test_selective_scan_three_operand_einsum_honors_contraction_path(self):
        b = torch.ones(1, 4, 3); u = torch.ones(1, 2, 3)
        model = Function(lambda delta: torch.ops.aten.einsum.default("bdl,bnl,bdl->bdln", [delta, b, u], path=[0, 2, 0, 1]))
        report = profile_model(model, torch.ones(1, 2, 3))
        self.assertEqual(report["coverage"], "complete_under_convention")
        # delta*u multiplies6numbers, then outer product withB produces24numbers.
        self.assertEqual(report["estimated_flops_total"], 6+24)
        a = torch.ones(2, 3); middle = torch.ones(3, 4)
        chain = Function(lambda c: torch.ops.aten.einsum.default("ab,bc,cd->ad", [a, middle, c], path=[0, 1, 0, 1]))
        result = profile_model(chain, torch.ones(4, 5))
        self.assertEqual(result["estimated_flops_total"], 48+80)
        a = torch.ones(1, 2); b = torch.ones(1, 3)
        broadcast = Function(lambda c: torch.ops.aten.einsum.default("ij,ik,ik->ij", [a, b, c], path=[0, 1, 0, 1]))
        result = profile_model(broadcast, torch.ones(4, 3))
        self.assertEqual(result["estimated_flops_total"], 6+48)

    def test_ssm_primitives_are_counted_not_silently_ignored(self):
        # Four real nonlinear primitives, multiplication,addition,cumulative sum.
        model = Function(lambda x: torch.cumsum(torch.exp(x)*F.silu(x)+F.softplus(torch.sigmoid(x)), dim=1))
        report = profile_model(model, torch.ones(1, 3, 2))
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertEqual(report["estimated_flops_total"], 6*6+(6-2))
        for name in ["exp", "silu", "softplus", "sigmoid"]:
            self.assertTrue(any(name in op for op in report["operation_counts"]))

    def test_fused_gru_and_lstm_match_explicit_gate_equations(self):
        # One layer,7steps,B1,I5,H4; bias in both affine branches.
        gru = profile_model(nn.GRU(5, 4, batch_first=True), torch.ones(1, 7, 5))
        lstm = profile_model(nn.LSTM(5, 4, batch_first=True), torch.ones(1, 7, 5))
        self.assertEqual(gru["coverage"], "complete_under_convention")
        self.assertEqual(lstm["coverage"], "complete_under_convention")
        self.assertEqual(gru["estimated_flops_total"], 7*(2*3*4*(5+4)+17*4))
        self.assertEqual(lstm["estimated_flops_total"], 7*(2*4*4*(5+4)+21*4))

    def test_multilayer_bidirectional_recurrence_and_projection_guard(self):
        model = nn.LSTM(3, 2, num_layers=2, bidirectional=True, batch_first=True, bias=False)
        report = profile_model(model, torch.ones(1, 2, 3))
        self.assertEqual(report["estimated_flops_total"], 912)
        projected = profile_model(nn.LSTM(3, 4, proj_size=2, batch_first=True), torch.ones(1, 2, 3))
        self.assertEqual(projected["coverage"], "partial")
        self.assertIsNone(projected["estimated_flops_total"])

    def test_dense_attention_reference_and_real_weight_normalization(self):
        q = torch.ones(1, 2, 3, 4)
        model = Function(lambda x: F.scaled_dot_product_attention(x, x, x, dropout_p=0.))
        report = profile_model(model, q)
        self.assertEqual(report["coverage"], "complete_under_convention")
        self.assertGreater(report["estimated_flops_total"], 0)
        self.assertTrue(any("attention" in op for op in report["operation_counts"]))
        layer = nn.utils.weight_norm(nn.Conv1d(2, 3, 3))
        normalized = profile_model(layer, torch.ones(1, 2, 7))
        self.assertEqual(normalized["coverage"], "complete_under_convention")
        # Weight norm 3*18+3; convolution15*(2*2*3+1).
        self.assertEqual(normalized["estimated_flops_total"], 57+195)

    def test_native_fft_seven_marks_partial_and_no_radix2_count(self):
        model = Function(lambda x: torch.fft.rfft(x, dim=-1).abs())
        report = profile_model(model, torch.ones(1, 7))
        self.assertEqual(report["coverage"], "partial")
        self.assertIsNone(report["estimated_flops_per_sample"])
        self.assertEqual(report["counted_flops_total"], 4*4)
        self.assertIsNone(report["FFT_references"][0]["radix2_complex_reference_scalar_ops"])
        self.assertFalse(report["FFT_references"][0]["all_dimensions_power_of_two"])
        self.assertFalse(report["partial_count_is_hardware_lower_bound"])

    def test_native_fft_power2_reference_is_not_claimed_native_count(self):
        report = profile_model(Function(lambda x: torch.fft.fft(x)), torch.ones(1, 8))
        self.assertEqual(report["coverage"], "partial")
        self.assertEqual(report["FFT_references"][0]["radix2_complex_reference_scalar_ops"], 5*8*3)
        self.assertFalse(report["FFT_references"][0]["reference_is_native_FLOP_count_or_hardware_bound"])

    def test_unknown_floating_factorization_is_partial(self):
        report = profile_model(Function(lambda x: torch.linalg.svdvals(x)), torch.eye(3)[None])
        self.assertEqual(report["coverage"], "partial")
        self.assertIsNone(report["estimated_flops_total"])
        self.assertTrue(any("svd" in row["operator"] for row in report["unsupported_floating_ops"]))

    def test_throughput_uses_actual_batch_calls_and_restores_flags(self):
        class Predictor(nn.Module):
            def __init__(self): super().__init__(); self.calls = 0
            def forward(self, x): self.calls += 1; return x*2
        model = Predictor(); model.train(); threads = torch.get_num_threads()
        report = measure_throughput(model, torch.ones(32, 2), warmup=2, blocks=3, calls_per_block=2)
        self.assertEqual(model.calls, 2+3*2)
        self.assertEqual(report["timed_samples"], 3*2*32)
        self.assertEqual(report["timed_batches"], 6)
        self.assertEqual(report["blocks"], 3)
        self.assertTrue(math.isfinite(report["samples_per_second"]))
        self.assertGreater(report["samples_per_second"], 0)
        self.assertAlmostEqual(report["samples_per_second"], report["timed_samples"]/sum(report["block_seconds"]))
        self.assertTrue(model.training)
        self.assertEqual(torch.get_num_threads(), threads)

    def test_interleavable_block_and_pooled_not_mean_rate(self):
        report = summarize_throughput_blocks([1, 3], batch_size=32, calls_per_block=2)
        self.assertEqual(report["samples_per_second"], 32)
        self.assertNotEqual(report["samples_per_second"], report["block_rate_mean_samples_sec"])
        block = measure_throughput_block(lambda x: x, torch.ones(32, 1), calls_per_block=2)
        self.assertEqual(block["samples"], 64)
        self.assertGreater(block["block_seconds"], 0)

    def test_incorrect_output_batch_and_invalid_schedule_rejected(self):
        with self.assertRaises(ValueError):
            measure_throughput(lambda x: x[:1], torch.ones(32, 1), warmup=0, blocks=1, calls_per_block=1)
        calls = []
        with self.assertRaises(ValueError):
            measure_throughput(lambda x: calls.append(1), torch.ones(32, 1), calls_per_block=0)
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            summarize_throughput_blocks([0], batch_size=32, calls_per_block=2)


if __name__ == "__main__":
    unittest.main()
