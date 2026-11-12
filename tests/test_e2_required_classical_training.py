"""Classical chronology/selection tests on synthetic panels and an explicit ARIMA double."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import pandas as pd
import torch
import e2_required_classical_training as classic
import e2_temporal_controls as controls
import e2_real_executor as dated
from e2_temporal_training import DatedPanel, prepare_b


def fixture(*, validation_gap=False, outer_gap=False):
    rng=np.random.default_rng(291103)
    calendar=pd.date_range("2001-01-01",periods=260,tz="UTC")
    close=100*np.exp(np.cumsum(rng.normal(0,.008,len(calendar))))
    open_=np.r_[close[0],close[:-1]]
    high=np.maximum(open_,close)*1.005
    low=np.minimum(open_,close)*.995
    features=np.column_stack((open_,high,low,close,100+rng.uniform(0,50,len(close))))
    returns=np.r_[np.nan,np.diff(np.log(close))]
    targets=np.full(len(close),np.nan)
    for i in range(7,len(close)):
        targets[i]=np.std(returns[i-6:i+1],ddof=0)
    quality=np.ones(len(close),bool)
    quality[60]=False
    if validation_gap:quality[160]=False
    if outer_gap:quality[228]=False
    panel=DatedPanel(calendar,features,targets,quality)
    return panel,prepare_b(panel,calendar[220],calendar[237])


class ARIMADouble(controls.FrozenARIMA):
    fits=[]
    failures=set()

    @classmethod
    def fit(cls,history,origin,order,*,source_commit,**unused):
        origin=controls.day(origin);cutoff=origin-8*controls.DAY
        cls.fits.append({"order":order,"last_supplied":controls.day(history.index[-1]),
                         "origin":origin,"cutoff":cutoff,"source_commit":source_commit})
        if order in cls.failures:
            raise controls.ControlFailure("injected convergence failure",{"order":list(order),"retries":0})
        obj=cls();obj.origin=origin;obj.current=cutoff;obj.scale=1.
        obj._result=SimpleNamespace(params=np.array([1.,.2]))
        obj.receipt={"status":"fitted","order":list(order),"fit_cutoff":cutoff.isoformat()}
        obj.predictions=[];obj.observations=[];obj._pending=None;obj._failure=None
        y=controls.labels(history).loc[:cutoff];obj.last=float(y.dropna().iloc[-1])
        obj.missing_observations=0
        return obj

    def forecast(self,target):
        target=controls.day(target)
        if target!=self.current+controls.DAY or self._pending is not None:
            raise ValueError("double chronology")
        value=controls.output(target,self.last)
        self.predictions.append(vars(value));self._pending=target
        return value

    def observe(self,target,value):
        target=controls.day(target)
        if target!=self.current+controls.DAY:
            raise ValueError("double chronology")
        if target>=self.origin and self._pending!=target:
            raise ValueError("forecast must precede observation")
        if value is None or not np.isfinite(value):
            self.missing_observations+=1
        else:self.last=float(value)
        self.current=target;self._pending=None
        self.observations.append((target,value))


class ClassicalTrainingTests(unittest.TestCase):
    def setUp(self):
        ARIMADouble.fits=[];ARIMADouble.failures=set()
        self.panel,self.prepared=fixture()
        self.backend=patch.object(controls.FrozenARIMA,"fit",side_effect=ARIMADouble.fit)
        self.backend.start()
        self.addCleanup(self.backend.stop)

    def test_ridge_augmented_fit_matches_centered_closed_form_without_intercept_penalty(self):
        split=self.prepared.inner_train
        model=classic.fit_ridge(split,.7)
        x=split.x.numpy().astype(np.float64).reshape(len(split.dates),-1)
        y=split.y.numpy().astype(np.float64)
        xm,ym=x.mean(0),y.mean();center=x-xm
        expected=np.linalg.solve(center.T@center+.7*np.eye(x.shape[1]),center.T@(y-ym))
        np.testing.assert_allclose(model.coefficients,expected,rtol=1e-9,atol=1e-9)
        self.assertAlmostEqual(model.intercept,float(ym-xm@expected),places=10)
        self.assertFalse(model.coefficients.flags.writeable)

    def test_all_18_orders_and_18_alphas_are_distinct_and_failed_trials_remain(self):
        ARIMADouble.failures={(2,1,2)}
        result=classic.select_classical(self.panel,self.prepared,source_commit="synthetic-revision")
        ridge=[r for r in result.ledger if r["system"]=="Ridge"]
        arima=[r for r in result.ledger if r["system"]=="ARIMA"]
        self.assertEqual(len(ridge),18);self.assertEqual(len({r["alpha"] for r in ridge}),18)
        self.assertEqual(len(arima),18);self.assertEqual(sum(r["fit_count"] for r in arima),18)
        failed=[r for r in arima if r["status"]=="failed"]
        self.assertEqual(len(failed),1)
        self.assertEqual(failed[0]["order"],[2,1,2])
        self.assertIsNone(failed[0]["rmse"])
        self.assertEqual(failed[0]["failure_receipt"]["retries"],0)
        self.assertEqual(len(ARIMADouble.fits),18)
        for fit in ARIMADouble.fits:
            self.assertEqual(fit["last_supplied"],controls.day(self.prepared.inner_plan.cutoff))
            self.assertEqual(fit["cutoff"],controls.day(self.prepared.origin)-98*controls.DAY)
        self.assertTrue(result.record()["full_18_order_domain"])

    def test_required_comparators_skip_every_ridge_fit_and_refit(self):
        expected={"ARIMA","SeasonalNaive","Persistence"}
        with patch.object(classic,"fit_ridge",side_effect=AssertionError("Ridge must not run")) as ridge:
            selected=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                               include_ridge=False)
            self.assertIsNone(selected.ridge_alpha)
            self.assertEqual(set(selected.selected_scores),expected)
            self.assertEqual({row["system"] for row in selected.ledger},expected)
            self.assertEqual(len([row for row in selected.ledger if row["system"]=="ARIMA"]),18)
            self.assertTrue(selected.record()["full_18_order_domain"])
            self.assertFalse(selected.record()["include_ridge"])
            self.assertEqual(selected.record()["excluded_systems"],["Ridge"])
            result=classic.refit_classical(self.panel,self.prepared,selected,source_commit="synthetic")
            ridge.assert_not_called()
        self.assertEqual({system.name for system in result.systems},expected)
        self.assertEqual(result.receipts["new_parameter_fits"],1)
        self.assertEqual(set(result.fitted_models),{"ARIMA"})
        self.assertNotIn("Ridge",result.receipts)
        self.assertIsNone(result.receipts["selection_ledger"]["ridge_alpha"])
        self.assertEqual(len(ARIMADouble.fits),19)
        self.assertEqual(ARIMADouble.fits[-1]["last_supplied"],controls.day(self.prepared.outer_plan.cutoff))

    def test_ridge_exclusion_is_explicit_and_cannot_be_reintroduced_by_alpha_only(self):
        selected=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                           orders=((0,0,0),),include_ridge=False)
        contradictory=replace(selected,ridge_alpha=classic.RIDGE_ALPHAS[0])
        before=len(ARIMADouble.fits)
        with patch.object(classic,"fit_ridge",side_effect=AssertionError("No fit before validation")) as ridge:
            with self.assertRaisesRegex(ValueError,"Ridge inclusion"):
                classic.refit_classical(self.panel,self.prepared,contradictory,source_commit="synthetic")
            ridge.assert_not_called()
        self.assertEqual(len(ARIMADouble.fits),before)
        for value in (None,0,1,"False"):
            with self.subTest(include_ridge=value):
                with self.assertRaisesRegex(ValueError,"explicit boolean"):
                    classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                               include_ridge=value)
        self.assertEqual(len(ARIMADouble.fits),before)

    def test_historical_default_and_explicit_ridge_inclusion_keep_identical_receipts(self):
        with patch.object(classic.time,"perf_counter",return_value=10.):
            default=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                              orders=((0,0,0),))
            explicit=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                               orders=((0,0,0),),include_ridge=True)
        self.assertEqual(default.record(),explicit.record())
        self.assertEqual(default._seal,explicit._seal)
        self.assertEqual(set(default.record()),{"version","status","ridge_alpha","arima_order",
            "seasonal_period","selected_scores","prepared_signature","arima_domain",
            "full_18_order_domain","ledger"})
        self.assertNotIn("include_ridge",default.record())
        self.assertIsNotNone(default.ridge_alpha)

    def test_arima_forecasts_every_calendar_day_and_common_mask_stays_fixed(self):
        panel,prepared=fixture(validation_gap=True)
        events=[]
        result=classic.select_classical(panel,prepared,source_commit="synthetic",
            orders=((0,0,0),),prediction_sink=events.append)
        arima=next(r for r in result.ledger if r["system"]=="ARIMA")
        self.assertEqual(len(arima["forecast_events"]),83)
        self.assertEqual(arima["bridge_days"],7)
        expected=sum(prepared.inner_plan.eligible)
        self.assertEqual(len(arima["predictions_original"]),expected)
        self.assertLess(expected,83)
        dates=[e["date"] for e in events if e["system"]=="ARIMA"]
        self.assertEqual(dates,[day.isoformat() for day in prepared.inner_plan.evaluation_dates])
        self.assertFalse(result.record()["full_18_order_domain"])

    def test_all_arima_failures_block_selection_without_substituting_an_order(self):
        ARIMADouble.failures=set(controls.ORDERS)
        with self.assertRaises(classic.ClassicalSelectionFailure) as caught:
            classic.select_classical(self.panel,self.prepared,source_commit="synthetic")
        rows=[r for r in caught.exception.receipt["ledger"] if r["system"]=="ARIMA"]
        self.assertEqual(len(rows),18)
        self.assertTrue(all(r["status"]=="failed" for r in rows))
        self.assertFalse(caught.exception.receipt["retry_or_fallback"])

    def test_failed_prediction_sink_prevents_current_day_arima_observation(self):
        observed=[]
        original=ARIMADouble.observe
        def observe(model,target,value):
            observed.append((controls.day(target),model.origin))
            return original(model,target,value)
        def fail(event):
            if event["system"]=="ARIMA":raise OSError("injected sink failure")
        with patch.object(ARIMADouble,"observe",new=observe):
            with self.assertRaises(classic.ClassicalSelectionFailure):
                classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                    orders=((0,0,0),),prediction_sink=fail)
        self.assertEqual(len(observed),7)
        self.assertTrue(all(target<origin for target,origin in observed))

    def test_refit_is_fresh_at_outer_cutoff_and_ridge_fingerprint_freezes_scaler(self):
        selected=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                          orders=((0,0,0),))
        result=classic.refit_classical(self.panel,self.prepared,selected,source_commit="synthetic")
        self.assertEqual({s.name for s in result.systems},set(classic.SYSTEM_NAMES))
        self.assertEqual(result.receipts["new_parameter_fits"],2)
        self.assertEqual(set(result.fitted_models),{"Ridge","ARIMA"})
        np.testing.assert_allclose(result.receipts["Ridge"]["coefficients"],
                                   result.fitted_models["Ridge"].coefficients)
        self.assertEqual(ARIMADouble.fits[-1]["last_supplied"],controls.day(self.prepared.outer_plan.cutoff))
        self.assertTrue(result.receipts["ARIMA"]["bridge_not_filtered_yet"])
        ridge=next(s for s in result.systems if s.name=="Ridge")
        fingerprint=deepcopy(ridge.fingerprint())
        window=self.panel.raw_windows([self.prepared.origin])[0]
        self.assertTrue(np.isfinite(ridge.predict(self.prepared.origin,window)))
        self.assertEqual(fingerprint,ridge.fingerprint())
        self.assertNotEqual(result.receipts["Ridge"]["scaler"]["training_sha256"],
                            self.prepared.inner_train.scaler.training_sha256)

    def test_refitted_systems_complete_dated_replay_with_missing_outer_rows(self):
        panel,prepared=fixture(outer_gap=True)
        selection=classic.select_classical(panel,prepared,source_commit="synthetic",
                                           orders=((0,0,0),))
        refits=classic.refit_classical(panel,prepared,selection,source_commit="synthetic")
        replay=dated.DatedReplay(panel.calendar,panel.features,panel.targets,quality=panel.quality)
        with tempfile.TemporaryDirectory() as parent:
            journal=dated.DatedJournal(Path(parent)/"journal")
            result=dated.run_frozen_block(prepared.outer_plan,replay,refits.systems,journal)
            journal.close()
        self.assertEqual(result["calendar_days"],18)
        self.assertLess(result["scored_days"],18)
        self.assertEqual(set(result["metrics"]),set(classic.SYSTEM_NAMES))
        self.assertEqual(result["model_fits"],0)

    def test_future_outer_changes_do_not_change_selection_or_supplied_fit_prefix(self):
        original=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                          orders=((0,0,0),))
        x,y,q=self.panel.features,self.panel.targets,self.panel.quality
        positions=self.panel.calendar>self.prepared.outer_plan.cutoff
        x[positions,:4]*=5;x[positions,4]*=9;y[positions]=.3
        changed=DatedPanel(self.panel.calendar,x,y,q)
        future=prepare_b(changed,self.prepared.origin,self.prepared.outer_plan.evaluation_dates[-1])
        again=classic.select_classical(changed,future,source_commit="synthetic",orders=((0,0,0),))
        self.assertEqual(original.selected_scores,again.selected_scores)
        self.assertEqual(original.ridge_alpha,again.ridge_alpha)
        self.assertEqual(original.arima_order,again.arima_order)
        self.assertEqual(original.seasonal_period,again.seasonal_period)

    def test_mutated_selection_ledger_is_rejected_before_refit(self):
        selected=classic.select_classical(self.panel,self.prepared,source_commit="synthetic",
                                          orders=((0,0,0),))
        selected.ledger[0]["rmse"]=-99
        before=len(ARIMADouble.fits)
        with self.assertRaises(ValueError):
            classic.refit_classical(self.panel,self.prepared,selected,source_commit="synthetic")
        self.assertEqual(len(ARIMADouble.fits),before)


if __name__=="__main__":
    unittest.main()
