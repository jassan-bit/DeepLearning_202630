"""Synthetic calendar/physical-freeze tests; no actual E1 evaluation values."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_reduced_final_data as data


def freeze_fixture(root):
    root = Path(root)
    raw = {"scaler.json": b'{"train_end":"2024-10-31"}', "calibrator.json": b'{"calibration_end":"2024-12-30"}'}
    rows = []
    for model in data.SYSTEMS:
        for seed in (data.SEEDS if model in data.NEURAL else ("deterministic",)):
            name = f"weights/{len(rows):03d}.pt"
            raw[name] = (model + str(seed)).encode()
            rows.append({"model": model, "seed": seed, "checkpoint": name,
                "checkpoint_sha256": sha256(raw[name]).hexdigest(),
                "scaler_sha256": sha256(raw['scaler.json']).hexdigest(),
                "calibrator_sha256": sha256(raw['calibrator.json']).hexdigest()})
    for p, value in raw.items():
        target = root / p; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
    return {"protocol_id":data.PROTOCOL_ID,"protocol_sha256":data.PROTOCOL_SHA256,
        "systems":list(data.SYSTEMS),"final_seed_ids":list(data.SEEDS),
        "pilot_power_MDE_closed":True,"analysis_frozen":True,
        "models":rows,"frozen_files_sha256":{p:sha256(v).hexdigest() for p,v in raw.items()}}


def synthetic_frames():
    import numpy as np
    import pandas as pd
    dates = pd.date_range('2023-01-01','2026-08-31',tz='UTC')
    t = np.arange(len(dates))
    close = np.exp(7 + .0003*t + .03*np.sin(t/13))
    frame = pd.DataFrame({'open':close*.999,'high':close*1.01,'low':close*.99,'close':close,'volume':100+t},index=dates)
    return frame.loc[:'2024-12-31'],frame.loc['2025-01-01':]


class PhysicalFreezeTests(unittest.TestCase):
    def test_all_81_predictors_and_physical_artifacts_verified(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = freeze_fixture(root)
            result = data.validate_final_freeze(manifest,root)
            self.assertEqual(result['neural_seed_predictors'],78)
            self.assertEqual(result['classical_predictors'],3)
            self.assertEqual(result['physical_artifacts_verified'],83)

    def test_partial_duplicate_wrong_seed_and_open_analysis_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            original=freeze_fixture(root)
            bad=[]
            m=deepcopy(original);m['models'].pop();bad.append(m)
            m=deepcopy(original);m['models'][-1]=deepcopy(m['models'][0]);bad.append(m)
            m=deepcopy(original);m['final_seed_ids'][0]=271201;bad.append(m)
            m=deepcopy(original);m['analysis_frozen']=False;bad.append(m)
            m=deepcopy(original);m['pilot_power_MDE_closed']=False;bad.append(m)
            m=deepcopy(original);m['models'][0]['scaler_sha256']='a'*64;bad.append(m)
            for value in bad:
                with self.subTest(value=value),self.assertRaises(ValueError):
                    data.validate_final_freeze(value,root)

    def test_checkpoint_tamper_fails_before_any_market_loader(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=freeze_fixture(root)
            Path(root,manifest['models'][0]['checkpoint']).write_bytes(b'tampered')
            with patch.object(data,'reserve_known_e1_evaluation') as reserved:
                with self.assertRaisesRegex(ValueError,'bytes changed'):
                    data.load_known_e1_evaluation(root,{},manifest,artifact_root=root)
                reserved.assert_not_called()

    def test_path_escape_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=freeze_fixture(root)
            value=manifest['frozen_files_sha256'].pop('scaler.json')
            manifest['frozen_files_sha256']['../scaler.json']=value
            with self.assertRaisesRegex(ValueError,'normalized and relative'):
                data.validate_final_freeze(manifest,root)


class AdmissionTests(unittest.TestCase):
    def test_reservation_opens_metadata_only(self):
        root=Path(__file__).resolve().parents[1]
        original=data._read
        seen=[]
        def observed(r,p,h):
            seen.append(p)
            self.assertTrue(p.endswith('.json'))
            return original(r,p,h)
        with patch.object(data,'_read',side_effect=observed):
            reservation=data.reserve_known_e1_evaluation(root)
        self.assertEqual(seen,[data.RESERVE,data.E1_AUDIT])
        self.assertEqual(reservation['market_payloads_opened'],0)
        self.assertFalse(reservation['blindness_recovered'])
        self.assertEqual(len(reservation['archives']),20)

    def test_dec31_bridges_first_forecast_and_target_is_population_seven_returns(self):
        import numpy as np
        import pandas as pd
        dev,test=synthetic_frames()
        panel=data._panel_from_frames(dev,test)
        target_date=pd.Timestamp('2025-01-01',tz='UTC')
        window=panel.raw_windows([target_date])[0]
        np.testing.assert_array_equal(window[-1],dev.loc['2024-12-31'].to_numpy())
        full=pd.concat([dev,test])
        closes=full.loc['2024-12-25':'2025-01-01','close'].to_numpy()
        expected=np.std(np.diff(np.log(closes)),ddof=0)
        self.assertAlmostEqual(panel.targets_at([target_date])[0],expected,places=13)
        self.assertEqual(len(panel.support.loc['2025-01-01':'2026-08-31']),608)

    def test_missing_day_not_compressed_and_invalid_geometry_rejected(self):
        import numpy as np
        dev,test=synthetic_frames()
        with self.assertRaisesRegex(ValueError,'608 dates'):
            data._panel_from_frames(dev,test.iloc[1:])
        test.iloc[10,0]=np.inf
        with self.assertRaisesRegex(ValueError,'Infinite'):
            data._panel_from_frames(dev,test)

    def test_development_bridge_uses_registered_sources_and_preserves_Dec31(self):
        import numpy as np
        import pandas as pd
        import e2_development_data as old
        dev,_=synthetic_frames()
        dev.loc['2023-03-24',:]=np.nan
        targets=np.log(dev.close).diff().rolling(7,min_periods=7).std(ddof=0)
        market=dev.copy();market['trades']=1;market.index.name='market_day_utc'
        labels=pd.DataFrame({k:targets for k in ('sigma_7','sigma_14','sigma_21','sigma_28')},index=dev.index)
        labels.index.name='market_day_utc'
        raw={old.PROTOCOL_PATH:json.dumps({'sources':old.EXPECTED_SOURCES}).encode(),
            old.MARKET_PATH:market.to_csv().encode(),old.TARGET_PATH:labels.to_csv().encode()}
        def read_registered(root,relative,expected):
            self.assertIn(relative,raw)
            return raw[relative],expected
        with patch.object(old,'_read_registered',side_effect=read_registered) as reader:
            panel,receipt=data.load_development_bridge('/synthetic-only')
        self.assertEqual(reader.call_count,3)
        self.assertEqual(str(panel.calendar[-1].date()),'2024-12-31')
        self.assertEqual(receipt['bridge_dates'],['2024-12-31'])
        self.assertEqual(receipt['external_2025_2026_files_opened'],0)
        self.assertEqual(receipt['training_label_end'],'2024-10-31')

    def test_closed_cache_reloads_without_original_raws_or_daily_file(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=freeze_fixture(root)
            dev,test=synthetic_frames()
            test.index.name='market_day_utc'
            daily=Path(root,'synthetic-daily.csv')
            daily.write_text(test.to_csv())
            devpanel=data._panel_from_frames(dev)
            reservation={'sha256':'a'*64,'archives':[]}
            output=Path(root,'private')
            with patch.object(data,'reserve_known_e1_evaluation',return_value=reservation), \
                 patch.object(data,'E1_DAILY_SHA256',sha256(daily.read_bytes()).hexdigest()), \
                 patch.object(data,'load_development_bridge',return_value=(devpanel,{})) as bridge:
                panel,receipt=data.load_known_e1_evaluation(root,reservation,manifest,
                    artifact_root=root,daily_csv=daily,output_dir=output,development_root='/synthetic-dev-cache')
                daily.unlink()
                cached,cached_receipt=data.load_known_e1_evaluation(root,reservation,manifest,
                    artifact_root=root,daily_csv=daily,output_dir=output,development_root='/synthetic-dev-cache')
                self.assertEqual(bridge.call_args.args,('/synthetic-dev-cache',))
            self.assertEqual(receipt,cached_receipt)
            self.assertEqual(receipt['common_eligible_count'],608)
            self.assertFalse(receipt['blindness_recovered'])
            self.assertEqual(list(panel.calendar),list(cached.calendar))

    def test_missing_input_fails_before_creating_first_access_marker(self):
        with tempfile.TemporaryDirectory() as root:
            manifest=freeze_fixture(root)
            reservation={'sha256':'a'*64,'archives':[]}
            output=Path(root,'private')
            with patch.object(data,'reserve_known_e1_evaluation',return_value=reservation):
                with self.assertRaisesRegex(ValueError,'supply a pinned raw archive root'):
                    data.load_known_e1_evaluation(root,reservation,manifest,artifact_root=root,output_dir=output)
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
