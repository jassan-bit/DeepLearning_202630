"""Registered fourteen-model construction and immutable-source checks, no fitting."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import torch
from torch import nn
import e2_training_models as registry
from fetch_e2_itransformer_sources import MANIFEST, fetch_sources

SOURCE=Path(os.environ.get("E2_ITRANSFORMER_SOURCE_ROOT",str(ROOT/"work/author_sources/iTransformer")))


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.sources={"iTransformer":str(SOURCE)}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self,name,*,trial=None,seed=29):
        trial=trial or registry.registered_trials(name)[0]
        architecture=registry.reference_architecture(name) if name in registry.MODEL_IDS else None
        return registry.build_train_model(name,trial,architecture=architecture,
                                          source_roots=self.sources,seed=seed)

    def test_thirteen_local_models_construct_without_source_download_or_shape_changes(self):
        x=torch.randn(1,7,5)
        for name in registry.NEURAL_NAMES:
            if name=="iTransformer":
                continue
            with self.subTest(name=name):
                model=self.model(name).eval()
                with torch.no_grad(): y=model(x)
                self.assertEqual(tuple(y.shape),(1,))
                self.assertTrue(bool(torch.isfinite(y).all()))
                self.assertTrue(bool((y>0).all()))
                if name in registry.FIXED_COUNTS:
                    self.assertEqual(sum(p.numel() for p in model.parameters()),registry.FIXED_COUNTS[name])
                self.assertFalse(model.architecture_metadata["weights_pretrained"])
                self.assertFalse(model.architecture_metadata["training_axes_applied"])

    @unittest.skipUnless(SOURCE.is_dir(),"Explicit verified iTransformer source tree not supplied")
    def test_fourteenth_model_uses_original_pinned_backbone_and_all_requested_dropout_rates(self):
        self.assertEqual(registry.itransformer_source_manifest()["files"],MANIFEST["files"])
        for relative,expected in MANIFEST["files"].items():
            self.assertEqual(hashlib.sha256((SOURCE/relative).read_bytes()).hexdigest(),expected)
        trial=next(t for t in registry.registered_trials("iTransformer") if t["dropout"]==.5)
        model=self.model("iTransformer",trial=trial).eval()
        sites=[m for m in model.modules() if isinstance(m,nn.Dropout)]
        self.assertTrue(sites)
        self.assertTrue(all(m.p==.5 for m in sites))
        self.assertEqual(model.architecture_metadata["dropout"],.5)
        self.assertTrue(model.architecture_metadata["iTransformer_existing_dropout_rate_bridge"])
        self.assertEqual(sum(p.numel() for p in model.parameters()),795403)
        x=torch.randn(2,7,5)
        with torch.no_grad():
            together=model(x)
            alone=model(x[:1])
        self.assertEqual(tuple(together.shape),(2,))
        torch.testing.assert_close(together[:1],alone,rtol=2e-5,atol=2e-6)

    def test_unregistered_candidates_implicit_seeds_and_architecture_resizing_are_rejected(self):
        trial=registry.registered_trials("MLP16")[0]
        with self.assertRaises(ValueError):registry.build_train_model("MLP16",trial)
        forged=deepcopy(trial);forged["dropout"]=.123
        with self.assertRaises(ValueError):registry.build_train_model("MLP16",forged,seed=1)
        with self.assertRaises(ValueError):
            registry.build_train_model("MLP16",trial,architecture={"widths":[32]*3},seed=1)
        nas_trial=registry.registered_trials("TimesNet")[0]
        with self.assertRaises(ValueError):registry.build_train_model("TimesNet",nas_trial,seed=1)
        wrong=deepcopy(trial);wrong["recipe_id"]="other"
        with self.assertRaises(ValueError):registry.build_train_model("GRU32",wrong,seed=1)

    def test_fresh_initialization_is_reproducible_without_mutating_caller_rng(self):
        torch.manual_seed(331);before=torch.random.get_rng_state().clone()
        a=self.model("GRU32",seed=45)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b=self.model("GRU32",seed=45)
        c=self.model("GRU32",seed=46)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values())))
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.parameters(),c.parameters())))
        self.assertTrue(all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters())))
        self.assertEqual(a.architecture_metadata["role"],"reinitialized_E1_reference")
        x=torch.randn(2,7,5);a.eval()
        with torch.no_grad():
            first=a(x);a(200*x+5);last=a(x)
        torch.testing.assert_close(first,last)

    @unittest.skipUnless(SOURCE.is_dir(),"Explicit verified iTransformer source tree not supplied")
    def test_pinned_source_tampering_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            target=Path(temporary)/"original";shutil.copytree(SOURCE,target)
            path=target/"model/iTransformer.py";path.write_bytes(path.read_bytes()+b"\n")
            trial=registry.registered_trials("iTransformer")[0]
            with self.assertRaises(ValueError):
                registry.build_train_model("iTransformer",trial,
                    source_roots={"iTransformer":str(target)},seed=1)


class SourceRetrievalTests(unittest.TestCase):
    def test_existing_destination_is_refused_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                fetch_sources(Path(temporary),downloader=lambda _:self.fail("Unexpected network"))

    def test_invalid_response_is_rejected_and_temporary_tree_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"new"
            with self.assertRaises(ValueError):
                fetch_sources(output,downloader=lambda _:b"incorrect source")
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temporary).iterdir()),[])

    @unittest.skipUnless(SOURCE.is_dir(),"Explicit verified iTransformer source tree not supplied")
    def test_complete_new_tree_preserves_exact_pinned_bytes(self):
        prefix="https://raw.githubusercontent.com/"+MANIFEST["repository"]+"/"+MANIFEST["commit"]+"/"
        requested=[]
        def retrieve(url):
            self.assertTrue(url.startswith(prefix));relative=url[len(prefix):]
            self.assertIn(relative,MANIFEST["files"]);requested.append(relative)
            return (SOURCE/relative).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/"new"
            receipt=fetch_sources(output,downloader=retrieve)
            self.assertEqual(receipt["status"],"complete")
            self.assertEqual(len(requested),8)
            for relative,expected in MANIFEST["files"].items():
                self.assertEqual(hashlib.sha256((output/relative).read_bytes()).hexdigest(),expected)

