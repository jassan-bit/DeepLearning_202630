"""Causal preprocessing and mean-seed selection/refit tests on artificial calendars."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_temporal_training import (
    DatedPanel, FoldScaler, relative_window, prepare_b, CandidateSpec, TrialRun,
    RefitRun, select_b, refit_b, SelectionAborted,
    top_k_with_reference,
)
from dl_data_v2 import relative_features

DAY = pd.Timedelta(days=1)


def fixture():
    dates = pd.date_range("2001-01-01", periods=400, tz="UTC")
    i = np.arange(len(dates), dtype=float)
    close = 100+.1*i+np.sin(i/20)
    open_ = close*.999
    high = np.maximum(close,open_)*1.002
    low = np.minimum(close,open_)*.998
    x = np.column_stack([open_,high,low,close,1000+i])
    y = .01+.00001*i
    return dates,x,y


def callback(candidate, seed, train, validation):
    offset = candidate.payload["errors"][seed-10]
    return TrialRun(
        "completed", candidate.candidate_id+":"+str(seed),
        {"optimizer_steps": 2, "seconds": .01},
        tuple(v+offset for v in validation.y_original), validation.dates,
        {"supervised": 2 if seed==10 else 5},
    )


class TemporalPreparationTests(unittest.TestCase):
    def test_relative_window_matches_preserved_e1_function(self):
        dates,x,y = fixture()
        raw = np.stack([x[20:27],x[80:87]])
        actual = relative_window(raw)
        np.testing.assert_array_equal(actual, relative_features(raw))
        np.testing.assert_array_equal(actual[:, -1, 3], np.zeros(2))
        with self.assertRaises(ValueError):
            relative_window(np.zeros((1,7,5)))

    def test_exact_boundaries_train_only_scales_and_separate_refit(self):
        dates,x,y = fixture()
        panel = DatedPanel(dates,x,y)
        prepared = prepare_b(panel,dates[200],dates[209])
        self.assertEqual(prepared.inner_train.dates[-1],dates[102])
        self.assertEqual(prepared.inner_plan.origin,dates[110])
        self.assertEqual(prepared.inner_plan.evaluation_dates[-1],dates[192])
        self.assertEqual(len(prepared.inner_plan.evaluation_dates),83)
        self.assertEqual(prepared.refit_train.dates[-1],dates[192])
        self.assertIs(prepared.inner_validation.scaler,prepared.inner_train.scaler)
        self.assertIsNot(prepared.refit_train.scaler,prepared.inner_train.scaler)
        raw = panel.raw_windows(prepared.inner_train.dates)
        transformed = relative_features(raw)
        np.testing.assert_allclose(prepared.inner_train.scaler.mean, transformed.mean((0,1)),rtol=0,atol=0)
        np.testing.assert_allclose(prepared.inner_train.scaler.std, transformed.std((0,1),ddof=0),rtol=0,atol=0)
        expected = y[dates.get_indexer(prepared.inner_train.dates)].mean()
        self.assertEqual(prepared.inner_train.scaler.target_scale, expected)

    def test_future_changes_cannot_change_inner_training_or_any_refit_from_outer(self):
        dates,x,y = fixture()
        first = prepare_b(DatedPanel(dates,x,y),dates[200],dates[209])
        changed=x.copy();targets=y.copy()
        changed[103:,:4]*=10;changed[103:,4]*=100;targets[103:]*=5
        second=prepare_b(DatedPanel(dates,changed,targets),dates[200],dates[209])
        self.assertEqual(first.inner_train.signature,second.inner_train.signature)
        self.assertNotEqual(first.refit_train.signature,second.refit_train.signature)
        changed=x.copy();targets=y.copy()
        changed[200:,:4]*=10;changed[200:,4]*=100;targets[200:]*=5
        third=prepare_b(DatedPanel(dates,changed,targets),dates[200],dates[209])
        self.assertEqual(first.inner_train.signature,third.inner_train.signature)
        self.assertEqual(first.inner_validation.signature,third.inner_validation.signature)
        self.assertEqual(first.refit_train.signature,third.refit_train.signature)

    def test_quality_gaps_remove_examples_without_compressing_calendar(self):
        dates,x,y=fixture()
        quality=np.ones(len(dates),dtype=bool);quality[60]=False;quality[203]=False
        panel=DatedPanel(dates,x,y,quality)
        prepared=prepare_b(panel,dates[200],dates[209])
        self.assertNotIn(dates[60],prepared.inner_train.dates)
        self.assertNotIn(dates[67],prepared.inner_train.dates)
        self.assertIn(dates[68],prepared.inner_train.dates)
        self.assertEqual(len(prepared.outer_plan.evaluation_dates),10)
        self.assertEqual(prepared.outer_plan.evaluation_dates[3],dates[203])
        self.assertFalse(prepared.outer_plan.eligible[3])
        with self.assertRaises(ValueError):
            DatedPanel(dates.delete(60),np.delete(x,60,0),np.delete(y,60))
        with self.assertRaises(ValueError):
            DatedPanel(dates[::-1],x[::-1],y[::-1])

    def test_source_arrays_and_prepared_callbacks_are_isolated(self):
        dates,x,y=fixture()
        panel=DatedPanel(dates,x,y)
        x[:]=0;y[:]=999
        prepared=prepare_b(panel,dates[200],dates[209])
        self.assertTrue(bool((prepared.inner_train.x!=0).any()))
        original=prepared.inner_train.x.clone()
        seen=[]
        def mutate(candidate,seed,train,validation):
            torch.testing.assert_close(train.x,original)
            train.x.fill_(999);validation.y.fill_(999)
            seen.append((candidate.candidate_id,seed))
            return callback(candidate,seed,train,validation)
        result=select_b(prepared,[CandidateSpec("A",{"errors":[1,1]})],[10,11],mutate)
        self.assertEqual(result.attempts,2)
        self.assertEqual(len(seen),2)
        prepared.validate()
        torch.testing.assert_close(prepared.inner_train.x,original)


class SelectionRefitTests(unittest.TestCase):
    def setUp(self):
        self.dates,x,y=fixture()
        self.prepared=prepare_b(DatedPanel(self.dates,x,y),self.dates[200],self.dates[209])

    def test_primary_is_mean_of_seed_rmse_not_pooled_seed_mse(self):
        candidates=[CandidateSpec("A",{"errors":[1.,3.]}),
                    CandidateSpec("B",{"errors":[2.1,2.1]})]
        result=select_b(self.prepared,candidates,[10,11],callback)
        self.assertEqual(result.winner.candidate_id,"A")
        self.assertAlmostEqual(result.candidate_ledger["A"]["mean_seed_RMSE"],2.)
        self.assertEqual(result.fixed_epochs,{"supervised":4})
        self.assertEqual(result.attempts,4)

    def test_failed_seed_keeps_attempts_and_excludes_whole_candidate(self):
        seen=[]
        def fit(candidate,seed,train,val):
            seen.append((candidate.candidate_id,seed))
            if candidate.candidate_id=="A" and seed==10:
                return TrialRun.failed("timeout",{"seconds":1.},"A:10")
            return callback(candidate,seed,train,val)
        result=select_b(self.prepared,[CandidateSpec("A",{"errors":[0.,0.]}),
                                      CandidateSpec("B",{"errors":[2.,2.]})],[10,11],fit)
        self.assertEqual(result.winner.candidate_id,"B")
        self.assertEqual(len(seen),4)
        self.assertEqual(result.candidate_ledger["A"]["status"],"failed")
        self.assertEqual(len(result.candidate_ledger["A"]["runs"]),2)
        def fail(candidate,seed,train,val):
            return TrialRun.failed("NaN",{"optimizer_steps":0},candidate.candidate_id+str(seed))
        with self.assertRaises(SelectionAborted) as caught:
            select_b(self.prepared,[CandidateSpec("A",{})],[10,11],fail)
        self.assertEqual(len(caught.exception.candidate_ledger["A"]["runs"]),2)

    def test_validation_dates_cannot_be_changed_and_fatal_error_aborts(self):
        def bad(candidate,seed,train,val):
            result=callback(candidate,seed,train,val)
            result.prediction_dates=result.prediction_dates[1:]
            return result
        with self.assertRaises(ValueError):
            select_b(self.prepared,[CandidateSpec("A",{"errors":[1,1]})],[10,11],bad)
        def fatal(candidate,seed,train,val):
            return TrialRun.failed("implementation failure",{"seconds":0.},str(seed),fatal=True)
        with self.assertRaises(SelectionAborted):
            select_b(self.prepared,[CandidateSpec("A",{})],[10,11],fatal)

    def test_refit_has_fresh_runs_fixed_median_epochs_and_no_validation_argument(self):
        selection=select_b(self.prepared,[CandidateSpec("A",{"errors":[1,1]})],[10,11],callback)
        models=[]
        def fit(candidate,seed,train,fixed_epochs):
            self.assertEqual(train.dates[-1],self.dates[192])
            self.assertEqual(fixed_epochs,{"supervised":4})
            model=torch.nn.Linear(35,1);models.append(model)
            return RefitRun("completed","refit:"+str(seed),{"optimizer_steps":4},
                            deepcopy(fixed_epochs),True,artifact=model)
        results=refit_b(self.prepared,selection,[20,21],fit)
        self.assertEqual(len(results),2)
        self.assertIsNot(results[0].artifact,results[1].artifact)
        with self.assertRaises(ValueError):
            refit_b(self.prepared,selection,[10],fit)
        def warm(candidate,seed,train,epochs):
            return RefitRun("completed","new",{"optimizer_steps":1},epochs,False)
        with self.assertRaises(ValueError):
            refit_b(self.prepared,selection,[20],warm)

    def test_mutated_selections_and_reused_run_ids_are_rejected(self):
        selection=select_b(self.prepared,[CandidateSpec("A",{"errors":[1,1]})],[10,11],callback)
        selection.fixed_epochs["supervised"]=99
        with self.assertRaises(ValueError):
            refit_b(self.prepared,selection,[20],lambda *args:None)
        def reused(candidate,seed,train,val):
            result=callback(candidate,seed,train,val);result.run_id="same";return result
        with self.assertRaises(ValueError):
            select_b(self.prepared,[CandidateSpec("A",{"errors":[1,1]})],[10,11],reused)

    def test_nas_parsimony_uses_delta_or_mean_score_interval_overlap(self):
        def fitted(candidate,seed,train,val):
            error=candidate.payload["errors"][seed-10]
            return TrialRun("completed",candidate.candidate_id+str(seed),
                            {"model_parameters":candidate.payload["parameters"],"seconds":.01},
                            tuple(y+error for y in val.y_original),val.dates,{"supervised":3})
        candidates=[CandidateSpec("A",{"errors":[1.]*5,"parameters":100}),
                    CandidateSpec("B",{"errors":[1.05]*5,"parameters":50}),
                    CandidateSpec("C",{"errors":[1.2]*5,"parameters":1})]
        result=select_b(self.prepared,candidates,range(10,15),fitted,
                        selection_rule="nas_parsimony",delta=.1)
        self.assertEqual(result.winner.candidate_id,"B")
        self.assertEqual(result.tie_candidates,("A","B"))
        self.assertEqual(result.selection_rule,"nas_parsimony")
        self.assertIn("mean_RMSE_CI95",result.candidate_ledger["A"])
        candidates=[CandidateSpec("A",{"errors":[.8,1.,1.2,1.1,.9],"parameters":100}),
                    CandidateSpec("B",{"errors":[1.15]*5,"parameters":50})]
        result=select_b(self.prepared,candidates,range(10,15),fitted,
                        selection_rule="nas_parsimony",delta=.01)
        self.assertEqual(result.winner.candidate_id,"B")
        self.assertTrue(result.candidate_ledger["B"]["technical_tie_to_best"])

    def test_nas_rejects_insufficient_seeds_or_inconsistent_parameter_counts(self):
        candidates=[CandidateSpec("A",{"errors":[1.]*5})]
        with self.assertRaises(ValueError):
            select_b(self.prepared,candidates,[10,11],callback,
                     selection_rule="nas_parsimony",delta=.1)
        def mismatched(candidate,seed,train,val):
            return TrialRun("completed",str(seed),{"model_parameters":seed},
                            tuple(y+1 for y in val.y_original),val.dates,{"supervised":3})
        with self.assertRaises(ValueError):
            select_b(self.prepared,candidates,range(10,15),mismatched,
                     selection_rule="nas_parsimony",delta=.1)

    def test_phase1_finalists_add_reference_once_without_replacing_failures(self):
        candidates=[CandidateSpec("A",{"errors":[.1]}),
                    CandidateSpec("B",{"errors":[.2]}),
                    CandidateSpec("REF",{"errors":[.3]})]
        selection=select_b(self.prepared,candidates,[10],callback)
        finalists=top_k_with_reference(self.prepared,selection,candidates,2,"REF")
        self.assertEqual(tuple(c.candidate_id for c in finalists),("A","B","REF"))
        all_rows=top_k_with_reference(self.prepared,selection,candidates,3,"REF")
        self.assertEqual(len(all_rows),3)
        with self.assertRaises(ValueError):
            top_k_with_reference(self.prepared,selection,candidates,4,"REF")


if __name__=="__main__":
    unittest.main()

