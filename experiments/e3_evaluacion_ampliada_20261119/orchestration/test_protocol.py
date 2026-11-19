"""Freeze and byte-binding contracts; no HTTP, market targets or model fits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('extended_protocol_tested',Path(__file__).with_name('protocol.py'))
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
SOURCE='a'*40
RUN='20261119T020000000000Z'


class FakeAPI:
    REPOSITORY='jassan-bit/DeepLearning_202630'
    class StorageError(RuntimeError):pass
    def __init__(self,lost=False):
        self.obj=None;self.ref=None;self.posts=0;self.lost=lost
    def _request(self,method,path,token,payload=None,allow_missing=False):
        if method=='GET' and '/git/ref/tags/' in path:
            return copy.deepcopy(self.ref)
        if method=='POST' and path.endswith('/git/tags'):
            self.posts+=1
            self.obj={'sha':'b'*40,'tag':payload['tag'],'message':payload['message'],
                      'object':{'type':payload['type'],'sha':payload['object']},
                      'tagger':{'date':'2026-11-19T02:00:00Z'}}
            return copy.deepcopy(self.obj)
        if method=='POST' and path.endswith('/git/refs'):
            self.ref={'ref':payload['ref'],'object':{'type':'tag','sha':payload['sha']}}
            if self.lost:
                self.lost=False
                raise self.StorageError('simulated_lost_ack')
            return copy.deepcopy(self.ref)
        if method=='GET' and '/git/tags/' in path:
            return copy.deepcopy(self.obj)
        raise AssertionError((method,path))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        rows=[]
        for identity in ('D01','D02','D03'):
            p=self.root/'datasets'/(identity+'.csv');p.parent.mkdir(exist_ok=True)
            raw=b'date,open,high,low,close,volume\n'
            p.write_bytes(raw)
            rows.append({'id':identity,'admitted':True,'path':'datasets/'+identity+'.csv',
                         'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                         'target_values_computed':False})
        self.report={'kind':'e3_additional_OHLCV_admission','source_commit':SOURCE,
                     'datasets_required':3,'datasets_admitted':3,'datasets_evaluated_by_models':0,
                     'target_values_computed':False,'datasets':rows}
        self.admission=self.root/'admission.json';self.write_admission()
    def tearDown(self):self.temp.cleanup()
    def write_admission(self):self.admission.write_bytes(protocol.canonical(self.report))
    def make_protocol(self):
        value=protocol.build(self.admission,SOURCE)
        value['compute_profile']={'device':'cuda:0','GPU':'synthetic-fixture'}
        path=self.root/'protocol.json';path.write_bytes(protocol.canonical(value));return path,value

    def test_policy_is_explicit_nineteen_family_and_has_no_R3_certification(self):
        p=protocol.build(self.admission,SOURCE)
        self.assertEqual(len(p['family']),19)
        self.assertEqual(p['family'][-2:],['floor_only','floor_plus_mean_excess'])
        self.assertEqual(len(p['seeds']),10)
        self.assertNotIn(p['hpo_seed'],p['seeds'])
        self.assertTrue(p['training']['SAE_three_phases_share_one_fit_budget'])
        self.assertFalse(p['R3_accredited'])
        self.assertIn('not_dimensioned',p['planning_status'])

    def test_incomplete_or_changed_inputs_cannot_freeze(self):
        self.report['datasets_admitted']=2;self.write_admission()
        with self.assertRaisesRegex(ValueError,'Three complete'):
            protocol.build(self.admission,SOURCE)
        self.report['datasets_admitted']=3;self.write_admission()
        (self.root/'datasets/D02.csv').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            protocol.build(self.admission,SOURCE)

    def test_annotation_binds_full_protocol_and_retries_are_idempotent(self):
        path,p=self.make_protocol();api=FakeAPI()
        first=protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        second=protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        self.assertEqual(first['tag_object_sha'],second['tag_object_sha'])
        self.assertEqual(api.posts,1)
        message=json.loads(api.obj['message'])
        self.assertEqual(message['protocol'],p)
        self.assertEqual(message['protocol_sha256'],protocol.digest(path.read_bytes()))
        self.assertEqual(protocol.verify_receipt(path,first,SOURCE),first)

    def test_ref_lost_ack_is_recovered_without_replacing_an_existing_tag(self):
        path,_=self.make_protocol();api=FakeAPI(lost=True)
        protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        self.assertEqual(api.posts,1)
        api.obj['message']='other scope'
        with self.assertRaisesRegex(ValueError,'different immutable'):
            protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        self.assertEqual(api.posts,1)

    def test_lightweight_tag_and_modified_protocol_receipt_are_rejected(self):
        path,_=self.make_protocol();api=FakeAPI();receipt=protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        api.ref['object']['type']='commit'
        with self.assertRaisesRegex(ValueError,'annotated protocol'):
            protocol.freeze(path,SOURCE,RUN,'fixture-token',api)
        receipt['R3_accredited']=True
        with self.assertRaisesRegex(ValueError,'unchanged annotated'):
            protocol.verify_receipt(path,receipt,SOURCE)

    def test_compute_cohort_change_is_not_silently_mixed(self):
        value={'compute_profile':{'device':'cuda:0','GPU':'synthetic-A100','torch':'fixture'}}
        self.assertEqual(protocol.verify_compute_profile(value,value['compute_profile']),value['compute_profile'])
        with self.assertRaisesRegex(ValueError,'cohort changed'):
            protocol.verify_compute_profile(value,{'device':'cuda:0','GPU':'synthetic-L4','torch':'fixture'})


if __name__=='__main__':unittest.main()
