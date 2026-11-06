"""Synthetic final-backbone and real selective-scan checks; no market files.

Set E2_FINAL_ATTENTION_SOURCES to the retrieved immutable source root containing
Informer/,Autoformer/,iTransformer/,Crossformer/,S-Mamba/,MambaReference/.
Source integration tests explicitly skip when those external trees are absent.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_final_models_attention import (MODELS, SOURCE_SPECS, STRUCTURES,
    SelectiveStateSpaceCPU, _ShortProbAttention, build_model, selective_scan_cpu)


class SelectiveStateSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(7901)
        return (torch.randn(2, 3, 5, requires_grad=True), torch.randn(2, 3, 5, requires_grad=True),
                -torch.rand(3, 2), torch.randn(2, 2, 5, requires_grad=True),
                torch.randn(2, 2, 5, requires_grad=True), torch.randn(3),
                torch.randn(2, 3, 5), torch.randn(3))

    def test_selective_scan_matches_explicit_scalar_state_recurrence(self):
        u, delta, A, B, C, D, z, bias = self.fixture()
        actual = selective_scan_cpu(u, delta, A, B, C, D, z, bias)
        expected = torch.zeros_like(u)
        for batch in range(2):
            for channel in range(3):
                state = torch.zeros(2)
                for t in range(5):
                    dt = F.softplus(delta[batch, channel, t] + bias[channel])
                    state = torch.exp(dt*A[channel])*state + dt*B[batch, :, t]*u[batch, channel, t]
                    expected[batch, channel, t] = ((state*C[batch, :, t]).sum() + D[channel]*u[batch, channel, t])*F.silu(z[batch, channel, t])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-6)

    def test_scan_backward_reaches_input_delta_and_state_maps(self):
        inputs = self.fixture()
        actual = selective_scan_cpu(*inputs)
        actual.square().sum().backward()
        for value in [inputs[0], inputs[1], inputs[3], inputs[4]]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_scan_is_causal_and_returns_actual_final_state(self):
        u, delta, A, B, C, D, z, bias = self.fixture()
        full, state = selective_scan_cpu(u, delta, A, B, C, D, z, bias, return_last_state=True)
        prefix, prefix_state = selective_scan_cpu(u[..., :3], delta[..., :3], A,
            B[..., :3], C[..., :3], D, z[..., :3], bias, return_last_state=True)
        torch.testing.assert_close(full[..., :3], prefix)
        self.assertEqual(tuple(state.shape), (2, 3, 2))
        self.assertFalse(torch.equal(state, prefix_state))

    def test_block_uses_input_dependent_scan_and_all_parameters(self):
        torch.manual_seed(72)
        model = SelectiveStateSpaceCPU(d_model=16, d_state=2)
        x = torch.randn(2, 5, 16)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(model.A_log._no_weight_decay)
        self.assertTrue(model.D._no_weight_decay)

    def test_single_token_block_is_finite_and_source_state_width_is_retained(self):
        model = SelectiveStateSpaceCPU()
        self.assertEqual(model.d_model, 256)
        self.assertEqual(model.d_state, 2)
        self.assertEqual(model.d_conv, 2)
        self.assertEqual(model.dt_rank, 16)
        self.assertTrue(torch.isfinite(model(torch.zeros(1, 1, 256))).all())

    def test_scan_rejects_wrong_state_layout(self):
        u, delta, A, B, C, D, z, bias = self.fixture()
        with self.assertRaises(ValueError):
            selective_scan_cpu(u, delta, A, B[:1], C, D, z, bias)

    def test_structure_override_and_cross_model_candidate_rejected(self):
        for candidate in ({"dropout": .1, "d_model": 16},
                          {"model": "Informer", "parameters": {"dropout": .1}}):
            with self.assertRaises(ValueError):
                build_model("S-Mamba", candidate=candidate)

    def test_unknown_model_and_missing_source_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_model("MLP", candidate={"dropout": .1})
        with self.assertRaises(ValueError):
            build_model("Informer", candidate={"dropout": .1})


class SourceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variable = os.environ.get("E2_FINAL_ATTENTION_SOURCES")
        if not variable:
            raise unittest.SkipTest("Explicit pinned author sources not supplied")
        cls.base = Path(variable)
        cls.roots = {name: cls.base/name for name in MODELS}
        torch.set_num_threads(1)

    def test_all_five_published_backbones_forward_backward_and_parameters(self):
        sizes = {}
        for i, name in enumerate(MODELS):
            with self.subTest(model=name):
                torch.manual_seed(301+i)
                model = build_model(name, candidate={"dropout": .13}, source_roots=self.roots)
                x = torch.randn(2, 7, 5)
                y = model(x)
                self.assertEqual(tuple(y.shape), (2,))
                self.assertTrue(torch.isfinite(y).all() and (y>=0).all())
                y.square().mean().backward()
                missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(missing, [])
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                sizes[name] = sum(p.numel() for p in model.parameters())
                self.assertEqual(model.architecture_metadata["structure"], STRUCTURES[name])
                self.assertFalse(model.architecture_metadata["calendar_inputs"])
                self.assertFalse(model.architecture_metadata["original_OHLCV_denormalization_applied"])
        # In particular these are not the d16/f32 P0 prototypes.
        self.assertGreater(sizes["Informer"], 20_000_000)
        self.assertGreater(sizes["Autoformer"], 10_000_000)
        self.assertGreater(sizes["Crossformer"], 10_000_000)
        self.assertGreater(sizes["iTransformer"], 700_000)
        self.assertGreater(sizes["S-Mamba"], 1_000_000)

    def test_eval_prediction_independent_of_batch_companions_and_rng(self):
        for name in MODELS:
            with self.subTest(model=name):
                torch.manual_seed(500)
                model = build_model(name, candidate={"dropout": .19}, source_roots=self.roots).eval()
                sample = torch.randn(1,7,5)
                with torch.no_grad():
                    first = model(sample)
                    joined = model(torch.cat((sample, torch.randn(2,7,5)*5), dim=0))[:1]
                    torch.manual_seed(9999)
                    repeated = model(sample)
                torch.testing.assert_close(first, joined, atol=3e-6, rtol=3e-5)
                torch.testing.assert_close(first, repeated, atol=1e-7, rtol=1e-6)

    def test_nested_candidate_dropout_and_complete_decoders(self):
        for name in MODELS:
            with self.subTest(model=name):
                model = build_model(name, candidate={"model":name,"parameters":{"dropout": .27}}, source_roots=self.roots)
                self.assertEqual(model.architecture_metadata["dropout"], .27)
                self.assertTrue(any(isinstance(m,torch.nn.Dropout) and m.p==.27 for m in model.modules()))
                if name=="Informer":
                    self.assertEqual([len(e.attn_layers) for e in model.backbone.encoder.encoders], [3,1])
                    self.assertEqual(model.backbone.encoder.inp_lens, [0,2])
                    self.assertEqual(len(model.backbone.decoder.layers),2)
                if name=="Autoformer": self.assertEqual(len(model.backbone.decoder.layers),1)
                if name=="Crossformer": self.assertEqual(len(model.backbone.decoder.decode_layers),4)

    def test_instances_remain_valid_after_another_author_tree_is_imported(self):
        first=build_model("iTransformer",candidate={"dropout": .1},source_roots=self.roots).eval()
        x=torch.randn(1,7,5)
        with torch.no_grad():a=first(x)
        second=build_model("Autoformer",candidate={"dropout": .1},source_roots=self.roots).eval()
        with torch.no_grad():
            second(x)
            b=first(x)
        torch.testing.assert_close(a,b)

    def test_full_cpu_mamba_block_matches_pinned_official_forward_and_gradient(self):
        from einops import rearrange,repeat
        import math
        from typing import Optional
        path=self.base/'MambaReference/mamba_ssm/modules/mamba_simple.py'
        expected=SOURCE_SPECS['MambaReference']['files']['mamba_ssm/modules/mamba_simple.py']
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),expected)
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Mamba')
        namespace={'torch':torch,'nn':torch.nn,'F':F,'math':math,'rearrange':rearrange,
                   'repeat':repeat,'Tensor':torch.Tensor,'Optional':Optional,
                   'selective_scan_fn':selective_scan_cpu,'mamba_inner_fn':None,
                   'causal_conv1d_fn':None,'causal_conv1d_update':None,
                   'selective_state_update':None,'RMSNorm':None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
        reference=namespace['Mamba'](d_model=16,d_state=2,d_conv=2,expand=1,use_fast_path=False)
        actual=SelectiveStateSpaceCPU(d_model=16,d_state=2,d_conv=2,expand=1)
        reference.load_state_dict(actual.state_dict(),strict=True)
        torch.manual_seed(910)
        x1=torch.randn(2,5,16,requires_grad=True);x2=x1.detach().clone().requires_grad_()
        a,b=actual(x1),reference(x2)
        torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-6)
        a.square().mean().backward();b.square().mean().backward()
        torch.testing.assert_close(x1.grad,x2.grad,atol=2e-7,rtol=1e-5)
        for (name,p),(_,q) in zip(actual.named_parameters(),reference.named_parameters()):
            self.assertIsNotNone(p.grad,name)
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=1e-5)

    def test_invalid_windows_and_nonfinite_dropout_are_rejected(self):
        model=build_model("S-Mamba", candidate={"dropout": .1}, source_roots=self.roots)
        for x in (torch.zeros(1,6,5),torch.full((1,7,5),float('nan')),torch.empty(0,7,5)):
            with self.assertRaises(ValueError):model(x)
        for p in [-.1,.31,float('nan')]:
            with self.assertRaises(ValueError):build_model("Informer", candidate={"dropout": p}, source_roots=self.roots)

    def test_tampered_source_is_rejected_before_import(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):build_model("iTransformer",candidate={"dropout": .1},source_roots={"iTransformer":d})

    def test_cpu_scan_matches_pinned_official_reference_value_and_gradient(self):
        from einops import rearrange, repeat
        path=self.base/'MambaReference/mamba_ssm/ops/selective_scan_interface.py'
        expected=SOURCE_SPECS['MambaReference']['files']['mamba_ssm/ops/selective_scan_interface.py']
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),expected)
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref')
        namespace={'torch':torch,'F':F,'rearrange':rearrange,'repeat':repeat}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
        reference=namespace['selective_scan_ref']
        torch.manual_seed(602)
        shapes=[(2,3,5),(2,3,5),(3,2),(2,2,5),(2,2,5),(3,),(2,3,5),(3,)]
        originals=[torch.randn(shape) for shape in shapes]
        originals[2]=-originals[2].abs()
        actual_inputs=[x.clone().requires_grad_() for x in originals]
        reference_inputs=[x.clone().requires_grad_() for x in originals]
        actual=selective_scan_cpu(*actual_inputs)
        expected=reference(*reference_inputs,delta_softplus=True)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=1e-6)
        actual.square().sum().backward();expected.square().sum().backward()
        for a,b in zip(actual_inputs,reference_inputs):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-5,rtol=1e-5)


if __name__=="__main__":unittest.main()
