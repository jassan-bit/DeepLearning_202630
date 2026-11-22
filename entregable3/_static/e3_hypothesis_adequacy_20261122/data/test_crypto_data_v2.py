"""Deterministic software fixtures; no prices, network, fit or real reserve.

The generated actor, license and final-authorization records test byte binding;
they do not represent the real author or an actual market-data admission.
"""
from __future__ import annotations
import contextlib
import copy
import base64
import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('crypto_data_v2',HERE/'engine_data_v2.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)
UTC = dt.timezone.utc


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def archive(start,end,market='spot',header=False,microseconds=False):
    output = io.BytesIO()
    lines = []
    if header:
        lines.append('open_time,open,high,low,close,volume,close_time,'+
            ('base_volume' if market=='cm' else 'quote_volume')+
            ',count,taker_buy_volume,taker_buy_quote_volume,ignore')
    first,last = dt.date.fromisoformat(start),dt.date.fromisoformat(end)
    unit=1_000_000 if microseconds else 1000
    for offset in range((last-first).days+1):
        date=first+dt.timedelta(days=offset)
        t=(date-dt.date(2000,1,1)).days
        close=100*np.exp(.0001*t+.02*np.sin(t/5))
        volume=3.5+(t%13)/100
        contracts=2000+t
        stamp=int(dt.datetime.combine(date,dt.time(),UTC).timestamp())*unit
        fields=[stamp,close*.999,close*1.01,close*.99,close,
                contracts if market=='cm' else volume,stamp+86400*unit-1,
                volume if market=='cm' else volume*close,7,1,1,0]
        lines.append(','.join(map(str,fields)))
    with zipfile.ZipFile(output,'w') as z:
        info=zipfile.ZipInfo('SYNTHETIC_SOFTWARE_FIXTURE.csv',date_time=(1980,1,1,0,0,0))
        z.writestr(info,'\n'.join(lines)+'\n')
    return output.getvalue()


def author_inputs(root):
    license_file=root/'fixture-license.txt'
    license_file.write_text('SOFTWARE FIXTURE NOTICE; NOT AN ACTUAL DATA LICENSE')
    owner=root/'fixture-owner.json'
    form=dict(actor_name='SYNTHETIC SOFTWARE ACTOR',actor_contact='fixture@example.invalid',
        actor_role='software fixture',statement_utc='2026-01-01T00:00:00+00:00',
        reserve_mode='development_only',evidence_origin='Generated software fixture; no authentic human claim',
        acknowledge_prior_D00_D03_exposed=True,all_data_declared_exposed_development=True,
        authorize_test_preparation=False,observation_cutoff_utc='2025-12-31T00:00:00+00:00',
        dataset_scope='Generated software dates only',source_url='https://example.invalid/fixture',
        source_license_evidence_path=str(license_file),source_license_evidence_sha256=data.sha256_bytes(license_file.read_bytes()))
    dump(owner,form)
    exposure=data.record_owner_evidence(owner,owner,root/'fixture-exposure.json')
    rights=dict(status='approved',actor_name=form['actor_name'],actor_contact=form['actor_contact'],
        evidence_path=str(license_file),evidence_sha256=data.sha256_bytes(license_file.read_bytes()),
        license='SOFTWARE FIXTURE ONLY',scope='Generated observations only; no real data authorization',
        source_url=form['source_url'],allow_derived_publication=False,allow_raw_redistribution=False)
    return rights,exposure


class CryptoDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.rights,self.exposure=author_inputs(self.root)
        self.calls=[]

    def tearDown(self):
        self.temp.cleanup()

    def config(self,provider='binance_vision_cm',symbol='BTCUSD_PERP',start='2010-01-01',end='2010-05-20',policy=None):
        result=dict(provider=provider,instrument=symbol,development_start=start,development_end=end)
        if policy:result['archive_policy']=policy
        return result

    def fake_get(self,config):
        bodies={}
        market=data.BINANCE_MARKETS[config['provider']]['market_type']
        for item in data.binance_archive_plan(config):
            body=archive(item['first_session'],item['last_session'],market,header=market!='spot')
            bodies[item['url']]=body
            bodies[item['url']+'.CHECKSUM']=(data.sha256_bytes(body)+'  '+item['path']+'\n').encode()
        def get(url):
            self.calls.append(url)
            if url not in bodies:raise AssertionError('Unplanned URL; no actual network allowed')
            return bodies[url]
        return get,bodies

    def collect(self,config,path=None,get=None):
        path=path or self.root/'raw'
        get=get or self.fake_get(config)[0]
        with patch.object(data,'_network_get',side_effect=get), patch('urllib.request.urlopen',side_effect=AssertionError('NETWORK_FORBIDDEN')), contextlib.redirect_stdout(io.StringIO()):
            return data.collect_development(config,path,rights_receipt=self.rights,
                exposure_receipt=self.exposure,enable_market_acquisition=True)

    def split(self,config):
        n=len(data.expected_sessions(config['development_start'],config['development_end'],'24/7'))
        ends=[n//4,n//2,3*n//4,n]
        return dict(raw_start=config['development_start'],raw_end=config['development_end'],calendar='24/7',
            embargo_bars=0,raw_bounds={name:[lo,hi-1] for name,lo,hi in zip(('fit','selection','planning','calibration'),[0]+ends[:-1],ends)})

    def test_spot_um_cm_base_volume_units_are_explicit(self):
        for market in ('spot','um','cm'):
            body=archive('2020-01-01','2020-01-02',market,header=market!='spot')
            result=data.parse_binance_zip(body,market_type=market,acquired_utc='2020-01-04T00:00:00+00:00')
            self.assertEqual(result['metadata']['volume_units'],'base_asset')
            self.assertEqual(result['metadata']['volume_column_zero_based'],7 if market=='cm' else 5)
            self.assertTrue((result['ohlcv'][:,4]<10).all())
            self.assertEqual(result['dates'].tolist(),['2020-01-01','2020-01-02'])

    def test_wrong_header_market_units_and_open_candles_rejected(self):
        with self.assertRaisesRegex(data.AdmissionError,'Wrong market-volume'):
            data.parse_binance_zip(archive('2020-01-01','2020-01-01','cm',header=True),market_type='um')
        with self.assertRaisesRegex(data.AdmissionError,'Futures timestamp unit'):
            data.parse_binance_zip(archive('2025-01-01','2025-01-01','um',microseconds=True),market_type='um')
        with self.assertRaisesRegex(data.AdmissionError,'Incomplete'):
            data.parse_binance_zip(archive('2020-01-01','2020-01-01'),acquired_utc='2020-01-01T12:00:00+00:00')

    def test_spot_microseconds_remain_supported(self):
        x=data.parse_binance_zip(archive('2025-01-01','2025-01-01',microseconds=True),acquired_utc='2025-01-03T00:00:00+00:00')
        self.assertEqual(x['metadata']['epoch_units_per_second'],[1_000_000])

    def test_archive_plan_has_no_out_of_bound_bars_and_handles_leap_year(self):
        c=self.config(start='2020-01-17',end='2020-03-05')
        p=data.binance_archive_plan(c)
        self.assertEqual(sum(x['expected_bars'] for x in p),49)
        self.assertEqual([x['path'] for x in p if x['archive_frequency']=='monthly'],['BTCUSD_PERP-1d-2020-02.zip'])
        dates=[]
        for item in p:dates.extend(data.expected_sessions(item['first_session'],item['last_session'],'24/7'))
        self.assertEqual(dates,data.expected_sessions(c['development_start'],c['development_end'],'24/7'))
        self.assertTrue(all('2020-01.zip' not in x['path'] and '2020-03.zip' not in x['path'] for x in p))

    def test_frozen_plan_counts_and_two_axes_are_metadata_not_admission(self):
        proposal=json.loads((HERE/'data_panels_v2.json').read_text())
        expected=[2328,1096,730,1095]
        total=0
        for p,n in zip(proposal['panels'],expected):
            config=dict(provider=p['provider'],instrument=p['instrument'],development_start=p['proposed_development_start'],development_end=p['proposed_development_end'],archive_policy=p['archive_policy'])
            plan=data.binance_archive_plan(config)
            self.assertEqual(sum(x['expected_bars'] for x in plan),n)
            total+=len(plan)
            if p['role']=='additional':self.assertEqual(len(p['diversity_axes_required']),2)
        self.assertEqual(total,187)
        self.assertEqual(proposal['admitted_panels'],0)
        self.assertIn('does not claim',proposal['prior_exposure_disclosure'])

    def test_explicit_activation_before_any_network_or_output(self):
        with patch.object(data,'_network_get',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(data.AdmissionError,'activation'):
                data.collect_development({},self.root/'uncreated',rights_receipt={},exposure_receipt={})
        self.assertFalse((self.root/'uncreated').exists())

    def test_collect_full_months_and_edges_preserves_source_bytes(self):
        c=self.config(start='2010-01-17',end='2010-03-05')
        get,bodies=self.fake_get(c)
        manifest=self.collect(c,get=get)
        self.assertEqual(len(manifest['files']),21)
        self.assertEqual(manifest['market_type'],'cm')
        for row in manifest['files']:
            self.assertEqual((self.root/'raw'/row['path']).read_bytes(),bodies[row['url']])
            self.assertIn('/futures/cm/',row['url'])
        self.assertEqual(len(self.calls),2*len(manifest['files']))

    def test_interrupted_acquisition_resumes_without_rewriting_verified_pairs(self):
        c=self.config(start='2010-01-17',end='2010-03-05')
        get,bodies=self.fake_get(c)
        def interrupted(url):
            if len(self.calls)>=4:raise data.AdmissionError('FIXTURE interruption')
            return get(url)
        with self.assertRaisesRegex(data.AdmissionError,'FIXTURE interruption'):
            self.collect(c,get=interrupted)
        raw=self.root/'raw'
        self.assertFalse((raw/'raw_manifest.json').exists())
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in raw.iterdir() if p.name.endswith('.zip') or p.name.endswith('.CHECKSUM') or p.name=='acquisition_pending.json'}
        original_start=json.loads((raw/'acquisition_pending.json').read_text())['acquisition_started_utc']
        state=data.inspect_binance_pending(c,raw,rights_receipt=self.rights,exposure_receipt=self.exposure)
        self.assertEqual(state['complete_pairs'],2)
        self.calls=[]
        manifest=self.collect(c,get=get)
        self.assertEqual(manifest['acquired_utc'],original_start)
        for name,previous in before.items():self.assertEqual((raw/name).read_bytes(),previous[0]);self.assertEqual((raw/name).stat().st_mtime_ns,previous[1])
        self.assertEqual(len(self.calls),2*(len(manifest['files'])-2))
        attempts=[json.loads(x) for x in (raw/'acquisition_attempts.jsonl').read_text().splitlines()]
        self.assertEqual([x['status'] for x in attempts],['started','failed','started','complete'])

    def test_orphan_zip_or_checksum_is_completed_only_by_matching_bytes(self):
        for orphan in ('zip','checksum'):
            with self.subTest(orphan=orphan):
                raw=self.root/orphan
                raw.mkdir()
                c=self.config(start='2010-01-01',end='2010-01-02')
                pending=data._pending_record(c,self.rights,self.exposure,'2026-01-01T00:00:00+00:00')
                dump(raw/'acquisition_pending.json',pending)
                get,bodies=self.fake_get(c);item=data.binance_archive_plan(c)[0]
                path=raw/(item['path'] if orphan=='zip' else item['path']+'.CHECKSUM')
                path.write_bytes(bodies[item['url'] if orphan=='zip' else item['url']+'.CHECKSUM'])
                before=(path.read_bytes(),path.stat().st_mtime_ns)
                result=self.collect(c,path=raw,get=get)
                self.assertEqual(len(result['files']),2)
                self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),before)

    def test_pending_different_config_rights_or_corrupt_source_stops_before_network(self):
        c=self.config(start='2010-01-01',end='2010-01-03')
        get,_=self.fake_get(c)
        def interrupted(url):
            if len(self.calls)>=2:raise data.AdmissionError('FIXTURE interruption')
            return get(url)
        with self.assertRaises(data.AdmissionError):self.collect(c,get=interrupted)
        raw=self.root/'raw';self.calls=[]
        for kind in ('config','rights','bytes'):
            with self.subTest(kind=kind):
                change=dict(c)
                rights=dict(self.rights)
                first=raw/data.binance_archive_plan(c)[0]['path']
                previous=first.read_bytes()
                if kind=='config':change['development_end']='2010-01-04'
                elif kind=='rights':rights['scope']='changed'
                else:first.write_bytes(b'corrupted')
                with patch.object(data,'_network_get',side_effect=AssertionError('network before metadata check')):
                    with self.assertRaises(data.AdmissionError):
                        data.collect_development(change,raw,rights_receipt=rights,exposure_receipt=self.exposure,enable_market_acquisition=True)
                first.write_bytes(previous)

    def test_unknown_partial_files_without_pending_are_preserved(self):
        raw=self.root/'raw';raw.mkdir();(raw/'unbound.zip').write_bytes(b'partial')
        with self.assertRaisesRegex(data.AdmissionError,'Unbound pre-existing'):
            self.collect(self.config())
        self.assertEqual((raw/'unbound.zip').read_bytes(),b'partial')

    def test_wrong_archive_dates_and_checksum_mismatch_never_create_manifest(self):
        c=self.config(start='2010-01-01',end='2010-01-02')
        for kind in ('dates','checksum'):
            with self.subTest(kind=kind):
                get,bodies=self.fake_get(c)
                row=data.binance_archive_plan(c)[0]
                if kind=='dates':
                    body=archive('2010-01-03','2010-01-03','cm')
                    bodies[row['url']]=body;bodies[row['url']+'.CHECKSUM']=(data.sha256_bytes(body)+'  fixture\n').encode()
                else:bodies[row['url']+'.CHECKSUM']=('0'*64+'  fixture\n').encode()
                with self.assertRaises(data.AdmissionError):self.collect(c,path=self.root/kind,get=get)
                self.assertFalse((self.root/kind/'raw_manifest.json').exists())

    def test_provider_market_url_identity_cannot_alias_spot_or_other_product(self):
        good='https://data.binance.vision/data/futures/cm/monthly/klines/BTCUSD_PERP/1d/BTCUSD_PERP-1d-2020-01.zip'
        self.assertEqual(data._recovery_download_url(good,'binance_vision_cm'),good)
        for wrong in ('binance_vision','binance_vision_um'):
            with self.assertRaisesRegex(data.AdmissionError,'market or frequency'):
                data._recovery_download_url(good,wrong)
        with self.assertRaisesRegex(data.AdmissionError,'provider/market'):
            data._parse_binance_manifest_payload(b'',dict(provider='binance_vision_cm',market_type='um',acquired_utc='2026-01-01T00:00:00+00:00'))

    def test_real_prepare_and_exact_recovery_for_spot_usdm_and_coinm(self):
        for provider,symbol in (('binance_vision','BTCUSDT'),('binance_vision_um','ETHUSDT'),('binance_vision_cm','BTCUSD_PERP')):
            with self.subTest(provider=provider):
                root=self.root/provider;root.mkdir()
                # Separate actor/evidence files keep each replay self-contained.
                self.rights,self.exposure=author_inputs(root)
                raw,target=root/'raw',root/'panel'
                config=self.config(provider,symbol)
                get,bodies=self.fake_get(config)
                manifest=self.collect(config,path=raw,get=get)
                admission=data.prepare_development(raw/'raw_manifest.json',self.split(config),target,rights=self.rights,exposure=self.exposure)
                self.assertEqual(admission['provenance']['volume_units'],'base_asset')
                self.assertEqual(admission['provenance']['market_type'],data.BINANCE_MARKETS[provider]['market_type'])
                self.assertEqual(set(admission['partition_paths']),{'fit','selection','planning','calibration'})
                originals={name:(target/name).read_bytes() for name in (*admission['partition_paths'].values(),'admission.json','recovery_metadata.json')}
                original_manifest=(raw/'raw_manifest.json').read_bytes()
                for part in ('fit','selection','planning','calibration'):
                    values=data.load_partition(target,part)
                    self.assertLess(values['raw'][:,:,4].max(),10)
                    self.assertTrue((values['support_end']-values['support_start']==14).all())
                bundle=root/'saved-recovery.json';bundle.write_bytes(originals['recovery_metadata.json'])
                shutil.rmtree(raw);shutil.rmtree(target)
                for name in ('fixture-license.txt','fixture-owner.json'):(root/name).unlink()
                with patch.object(data,'_network_get',side_effect=AssertionError('metadata restoration queried network')):
                    data.restore_development_metadata(bundle,root)
                self.calls=[]
                with patch.object(data,'_network_get',side_effect=get),patch.object(data,'utc_now',return_value='2030-01-01T00:00:00+00:00'),contextlib.redirect_stdout(io.StringIO()):
                    result=data.reacquire_development(bundle,target,raw,root,enable_market_acquisition=True)
                self.assertEqual(result['status'],'DEVELOPMENT_RESTORED_ORIGINAL_IDENTITY')
                self.assertEqual((raw/'raw_manifest.json').read_bytes(),original_manifest)
                for name,body in originals.items():self.assertEqual((target/name).read_bytes(),body)
                event=json.loads((raw/'reacquisition_events.jsonl').read_text().splitlines()[-1])
                self.assertEqual(event['finished_utc'],'2030-01-01T00:00:00+00:00')
                self.assertNotEqual(event['finished_utc'],event['original_acquired_utc'])
                with patch.object(data,'_network_get',side_effect=AssertionError('repeat recovery queried network')):
                    again=data.reacquire_development(bundle,target,raw,root,enable_market_acquisition=True)
                self.assertEqual(again['status'],'DEVELOPMENT_ALREADY_RESTORED')
                with self.assertRaisesRegex(data.AdmissionError,'sealed'):data.load_partition(target,'test')

    def test_final_preparation_is_disabled_before_any_file_or_network_access(self):
        with patch.object(data,'_network_get',side_effect=AssertionError('network')):
            with self.assertRaisesRegex(data.AdmissionError,'activation'):
                data.prepare_final_partition('unread',{},self.root/'test-uncreated',rights={},exposure={},authorization={})
        self.assertFalse((self.root/'test-uncreated').exists())

    def test_coinm_final_parser_uses_base_volume_under_explicit_fixture_authorization(self):
        # This is a manufactured software authorization over generated bars;
        # it is not a real author signature, reserve, or final market opening.
        config=self.config()
        manifest=self.collect(config)
        protocol=self.root/'fixture-protocol.json';protocol.write_text('{"synthetic_software_fixture":true}')
        checkpoints=self.root/'fixture-checkpoints.json';checkpoints.write_text('{"synthetic_software_fixture":true}')
        digest=data.sha256_bytes(protocol.read_bytes())
        form=dict(self.exposure['form'],reserve_mode='post_protocol',protocol_sha256=digest,
            exposure_inventory_sha256='a'*64,protocol_lock_utc='2009-12-31T00:00:00+00:00',
            first_raw_support_session='2010-01-01',authorize_test_preparation=True)
        form_path=self.root/'fixture-final-owner.json';dump(form_path,form)
        exposure=data.record_owner_evidence(form_path,form_path,self.root/'fixture-final-exposure.json')
        final_manifest={k:v for k,v in manifest.items() if k!='manifest_sha256'}
        final_manifest.update(scope='reserved_final',protocol_sha256=digest,exposure=exposure)
        final_manifest['manifest_sha256']=data.identity(final_manifest)
        path=self.root/'raw/final-manifest.json';dump(path,final_manifest)
        authorization=dict(self.rights,status='authorized',authorize_test_preparation=True,
            protocol_path=str(protocol),protocol_sha256=digest,sources_methods_size_frozen=True,
            selected_checkpoints_manifest_path=str(checkpoints),selected_checkpoints_sha256=data.sha256_bytes(checkpoints.read_bytes()),
            development_bar_index_origin=config['development_start'])
        split=dict(protocol_sha256=digest,raw_start=config['development_start'],raw_end=config['development_end'],
            calendar='24/7',raw_bounds={'test':[105,139]},target_start='2010-04-16',target_end=config['development_end'],
            embargo_bars=0,fixed_test_count=21)
        target=self.root/'fixture-final-panel'
        result=data.prepare_final_partition(path,split,target,rights=self.rights,exposure=exposure,
            authorization=authorization,enable_final_preparation=True)
        self.assertEqual(result['provenance']['provider'],'binance_vision_cm')
        self.assertEqual(result['provenance']['market_type'],'cm')
        with self.assertRaisesRegex(data.AdmissionError,'sealed'):data.load_partition(target,'test')
        values=data.load_partition(target,'test',allow_test=True)
        self.assertEqual(len(values['y']),21)
        self.assertLess(values['raw'][:,:,4].max(),10)

    def test_self_hashed_relabelling_cannot_admit_sources_or_construct_labels(self):
        config=self.config('binance_vision_um','BTCUSDT')
        manifest=self.collect(config)
        for mutation in ('instrument','provider','URL','filename','frequency','period'):
            with self.subTest(mutation=mutation):
                wrong=copy.deepcopy(manifest)
                if mutation=='instrument':wrong['instrument']='ETHUSDT'
                elif mutation=='provider':wrong['provider']='binance_vision';wrong['market_type']='spot'
                elif mutation=='URL':wrong['files'][0]['url']=wrong['files'][0]['url'].replace('/1d/','/1m/')
                elif mutation=='filename':wrong['files'][0]['path']=wrong['files'][0]['path'].replace('BTCUSDT','ETHUSDT')
                elif mutation=='frequency':wrong['files'][0]['archive_frequency']='daily'
                elif mutation=='period':wrong['files'][0]['first_session']='2009-12-31'
                wrong.pop('manifest_sha256');wrong['manifest_sha256']=data.identity(wrong)
                path=self.root/'raw/tampered-manifest.json';dump(path,wrong)
                with patch.object(data,'build_windows',side_effect=AssertionError('Labels accessed before source binding')):
                    with self.assertRaises(data.AdmissionError):
                        data.prepare_development(path,self.split(config),self.root/('bad-'+mutation),rights=self.rights,exposure=self.exposure)
                self.assertFalse((self.root/('bad-'+mutation)).exists())

    def test_checksum_filename_cannot_relabel_payload_even_with_recomputed_hashes(self):
        config=self.config('binance_vision_um','BTCUSDT')
        manifest=self.collect(config)
        wrong=copy.deepcopy(manifest)
        row=wrong['files'][0]
        checksum=(row['sha256']+'  ETHUSDT-1d-2010-01.zip\n').encode()
        (self.root/'raw'/(row['path']+'.CHECKSUM')).write_bytes(checksum)
        row['provider_checksum_sha256']=data.sha256_bytes(checksum)
        wrong.pop('manifest_sha256');wrong['manifest_sha256']=data.identity(wrong)
        with self.assertRaisesRegex(data.AdmissionError,'exact archive filename'):
            data.validate_binance_manifest(wrong,self.root/'raw/raw_manifest.json')

    def test_relabelled_recovery_bundle_is_rejected_before_network_or_partition_read(self):
        config=self.config('binance_vision_um','BTCUSDT')
        manifest=self.collect(config)
        target=self.root/'panel'
        data.prepare_development(self.root/'raw/raw_manifest.json',self.split(config),target,rights=self.rights,exposure=self.exposure)
        bundle=json.loads((target/'recovery_metadata.json').read_text())
        wrong=copy.deepcopy(manifest);wrong['instrument']='ETHUSDT'
        wrong.pop('manifest_sha256');wrong['manifest_sha256']=data.identity(wrong)
        body=(json.dumps(wrong,indent=2)+'\n').encode()
        bundle['raw_manifest'].update(content_base64=base64.b64encode(body).decode(),sha256=data.sha256_bytes(body))
        bundle.pop('metadata_sha256');bundle['metadata_sha256']=data.identity(bundle)
        path=self.root/'tampered-recovery.json';dump(path,bundle)
        with patch.object(data,'_network_get',side_effect=AssertionError('network before semantic validation')), \
             patch.object(data,'load_partition',side_effect=AssertionError('partition before semantic validation')):
            with self.assertRaises(data.AdmissionError):
                data.reacquire_development(path,target,self.root/'raw',self.root,enable_market_acquisition=True)

    def test_legacy_spot_source_A_replays_exactly_without_new_provenance_fields(self):
        candidates=[Path(os.environ['E3_SOURCE_A_DATA'])] if os.environ.get('E3_SOURCE_A_DATA') else []
        if len(HERE.parents)>2:
            candidates.append(HERE.parents[2]/'experiments/e3_confirmatory_engine_20261121/engine/data.py')
        candidates.append(Path('/workspace/e3_confirmatory_engine_20261121/engine/data.py'))
        original=next((path for path in candidates if path.is_file()),None)
        self.assertIsNotNone(original,'Set E3_SOURCE_A_DATA to the verified source-A engine/data.py for legacy replay QA')
        self.assertEqual(data.sha256_bytes(original.read_bytes()),'3851281f00fc7b43bfb681d10be988f0efc5fb3d879e6b4faab2fe4798a9d917')
        old_spec=importlib.util.spec_from_file_location('original_source_A_data',original)
        old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
        config=self.config('binance_vision','BTCUSDT',policy='daily_only_v1')
        get,bodies=self.fake_get(config)
        raw,target=self.root/'raw',self.root/'legacy-panel'
        with patch.object(old,'_network_get',side_effect=get):
            old.collect_development(config,raw,rights_receipt=self.rights,exposure_receipt=self.exposure,enable_market_acquisition=True)
        old_admission=old.prepare_development(raw/'raw_manifest.json',self.split(config),target,rights=self.rights,exposure=self.exposure)
        self.assertNotIn('data_adapter_version',old_admission['provenance'])
        original_files={p.name:p.read_bytes() for p in target.iterdir()}
        bundle=self.root/'legacy-recovery.json';bundle.write_bytes((target/'recovery_metadata.json').read_bytes())
        shutil.rmtree(raw);shutil.rmtree(target)
        with patch.object(data,'_network_get',side_effect=get):
            result=data.reacquire_development(bundle,target,raw,self.root,enable_market_acquisition=True)
        self.assertEqual(result['status'],'DEVELOPMENT_RESTORED_ORIGINAL_IDENTITY')
        for name,body in original_files.items():self.assertEqual((target/name).read_bytes(),body)
        self.assertEqual(data.read_admission(target)['provenance'],old_admission['provenance'])


if __name__=='__main__':unittest.main(verbosity=2)
