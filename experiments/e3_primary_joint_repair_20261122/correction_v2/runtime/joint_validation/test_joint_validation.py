"""Software fixtures only. These tests run no formal calibration replicas."""
from __future__ import annotations

import copy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from joint_validation import cli
from joint_validation.dgps import SCENARIOS,SYSTEM_IDS,DETERMINISTIC,generate,panel_covariance,true_risks
from joint_validation.logic import binomial_capacity,composition_bound,joint_rule,verify_witnesses,witness
from joint_validation.validation import digest,mask_hex,seed_for,unmask_hex


def fixture_record(plan,i,r):
    return dict(software_fixture=True,scenario_index=i,scenario=plan['scenarios'][i]['id'],replicate=r,
        data_seed=seed_for(plan['data_seed'],i,r),true_global=i==8,true_contrasts='00'*10,
        joint_acceptance=False,global_false_acceptance=False,contrast_pass='00'*10,additional_inferiority=False,
        witness=dict(coverage={f'CI{level}_{kind}':False for level in (95,99) for kind in ('rmse_difference','relative_rmse_gain')}),
        CPU_seconds=.001,wall_seconds=.001)


class FakePool:
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def submit(self,function,plan,i,r):
        future=Future();future.set_result(fixture_record(plan,i,r));return future


class JointLogicTests(unittest.TestCase):
    def test_truth_fixed_witness_exhaustion(self):
        self.assertEqual(verify_witnesses()['checked_false_truth_accepted_decision_pairs'],48)

    def test_lifting_to_twenty_rivals_retains_all_fixed_null_witnesses(self):
        import itertools
        checked=0
        for rival in range(20):
            for t in itertools.product((False,True),repeat=4):
                truth=np.ones((4,20),dtype=bool);truth[:,rival]=t
                if joint_rule(truth):continue
                chosen=witness(truth)
                for d in itertools.product((False,True),repeat=4):
                    flags=np.ones((4,20),dtype=bool);flags[:,rival]=d
                    if not joint_rule(flags):continue
                    self.assertTrue(any(flags[p,r] and not truth[p,r] for p,r in chosen));checked+=1
        self.assertEqual(checked,960)

    def test_two_of_three_is_per_rival_not_same_two_panels_for_everyone(self):
        flags=np.ones((4,20),dtype=bool)
        flags[1,0]=False;flags[2,1]=False;flags[3,2]=False
        self.assertTrue(joint_rule(flags))
        flags[2,0]=False;self.assertFalse(joint_rule(flags))

    def test_adverse_screen_can_only_reduce_acceptance(self):
        flags=np.ones((4,20),dtype=bool)
        self.assertTrue(joint_rule(flags));self.assertFalse(joint_rule(flags,True))

    def test_old_marginal_bound_does_not_certify_current_nominal_joint(self):
        self.assertAlmostEqual(composition_bound([.025]*4),.05)
        self.assertAlmostEqual(composition_bound([.05]*4),.10)

    def test_binomial_capacity_uses_one_107_claim_family(self):
        cap=binomial_capacity()
        self.assertEqual(cap['simultaneous_total_claims'],107)
        self.assertLess(cap['upper_zero_global_errors'],.05)
        self.assertGreater(cap['lower_perfect_80_contrast_power'],.9)
        self.assertEqual(cap['CI99_maximum_failures_to_certify_coverage_floor95'],10)

    def test_covariance_matrices_are_positive_definite(self):
        for mode in ('independent','equicorrelation_0.8','additional_pair_minus_0.8'):
            c=panel_covariance(mode)
            self.assertTrue((np.linalg.eigvalsh(c)>0).all())
            np.testing.assert_array_equal(np.diag(c),np.ones(4))

    def test_exact_truth_including_positive_risk_ranking_reversal(self):
        for i,case in enumerate(SCENARIOS):
            mu,theta,gains,truth=true_risks(case,10)
            self.assertEqual(joint_rule(truth),i==8)
            self.assertTrue((mu>0).all())
        mu,theta,_,_=true_risks(SCENARIOS[7],10)
        self.assertGreater(theta[0,0],theta[0,1])
        self.assertLess(mu[0,0].mean(),mu[0,1].mean())

    def test_generator_shapes_and_deterministic_controls(self):
        for case in (SCENARIOS[0],SCENARIOS[5],SCENARIOS[7]):
            data=generate(case,10,32,991)
            self.assertEqual(data['losses'].shape,(4,21,10,32))
            self.assertEqual(data['instruments'].shape,(4,32,2))
            self.assertTrue(np.isfinite(data['losses']).all())
            for j,name in enumerate(SYSTEM_IDS):
                if name in DETERMINISTIC:
                    np.testing.assert_array_equal(data['losses'][:,j],np.repeat(data['losses'][:,j,:1],10,axis=1))

    def test_replica_seed_and_data_are_exactly_reproducible(self):
        s=seed_for(99,3,20)
        self.assertNotEqual(s,seed_for(99,3,21))
        a=generate(SCENARIOS[3],10,32,s)
        b=generate(SCENARIOS[3],10,32,s)
        np.testing.assert_array_equal(a['losses'],b['losses'])

    def test_mask_roundtrip_preserves_all_eighty_positions(self):
        flags=np.arange(80)%3==0
        np.testing.assert_array_equal(unmask_hex(mask_hex(flags)),flags)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.case=self.root/'case'
        cli.freeze(self.case,cli.HERE/'protocol.json','1'*40)

    def tearDown(self):self.temp.cleanup()

    def test_freeze_runs_no_replicas_and_refuses_overwrite(self):
        plan,identity,rows,*_=cli.load_case(self.case)
        self.assertEqual(rows,[])
        self.assertFalse(cli.summarize(plan,identity,[],[])['accepted'])
        with self.assertRaisesRegex(ValueError,'IMMUTABLE_CASE_EXISTS'):
            cli.freeze(self.case,cli.HERE/'protocol.json','1'*40)

    def test_concurrent_writer_is_rejected(self):
        import os
        cli.atomic(self.case/'.run.lock',dict(pid=os.getpid(),started_utc='SOFTWARE FIXTURE'))
        with self.assertRaisesRegex(ValueError,'CASE_ALREADY_RUNNING'):
            cli.run(self.case,True,1,100,1)

    def test_partial_chunk_resume_skip_and_round_robin_order(self):
        with mock.patch.object(cli,'ProcessPoolExecutor',FakePool):
            a=cli.run(self.case,True,2,100,1)
            b=cli.run(self.case,True,2,100,1)
        self.assertEqual(a['completed_replicates'],2)
        self.assertEqual(b['completed_replicates'],4)
        self.assertFalse(b['accepted'])
        rows=cli.load_case(self.case)[2]
        self.assertEqual([(x['scenario_index'],x['replicate']) for x in rows],[(0,0),(1,0),(2,0),(3,0)])

    def test_catastrophic_false_acceptance_is_reported_not_a_none_comparison_error(self):
        plan,identity,*_=cli.load_case(self.case)
        rows=[]
        for rep in range(plan['replicates_per_scenario']):
            row=fixture_record(plan,0,rep)
            row.update(joint_acceptance=True,global_false_acceptance=True,contrast_pass='ff'*10)
            rows.append(row)
        state=cli.summarize(plan,identity,rows,[])
        self.assertEqual(state['scenarios'][0]['global_false_acceptances'],512)
        self.assertFalse(state['scenarios'][0]['criterion_pass'])
        self.assertFalse(state['accepted'])

    def test_truncated_ledger_is_rejected_without_erasing_bytes(self):
        path=self.case/'ledger.jsonl';path.write_bytes(b'{"truncated":')
        before=path.read_bytes()
        with self.assertRaisesRegex(ValueError,'TRUNCATED_LEDGER'):cli.load_case(self.case)
        self.assertEqual(path.read_bytes(),before)

    def test_modified_ledger_is_rejected(self):
        with mock.patch.object(cli,'ProcessPoolExecutor',FakePool):cli.run(self.case,True,1,100,1)
        p=self.case/'ledger.jsonl';p.write_bytes(p.read_bytes().replace(b'"CPU_seconds":0.001',b'"CPU_seconds":0.002'))
        with self.assertRaisesRegex(ValueError,'LEDGER_HASH_MISMATCH'):cli.load_case(self.case)

    def test_modified_protocol_and_identity_are_rejected(self):
        p=self.case/'protocol.json';body=p.read_bytes();p.write_bytes(body+b' ')
        with self.assertRaisesRegex(ValueError,'PROTOCOL_HASH_MISMATCH'):cli.load_case(self.case)
        p.write_bytes(body)
        p=self.case/'source_identity.json';obj=cli.read(p);obj['source_commit']='2'*40;cli.atomic(p,obj)
        with self.assertRaisesRegex(ValueError,'SOURCE_IDENTITY_HASH_MISMATCH'):cli.load_case(self.case)

    def test_export_restore_preserves_exact_ledger_and_completed_replicas(self):
        with mock.patch.object(cli,'ProcessPoolExecutor',FakePool):cli.run(self.case,True,2,100,1)
        export=self.root/'exports'/'checkpoint.zip'
        cli.export(self.case,export)
        restored=self.root/'nested'/'restored'
        result=cli.restore(export,restored,'1'*40)
        self.assertEqual(result['completed_replicates'],2)
        self.assertEqual((self.case/'ledger.jsonl').read_bytes(),(restored/'ledger.jsonl').read_bytes())
        self.assertEqual((self.case/'source_identity.json').read_bytes(),(restored/'source_identity.json').read_bytes())

    def test_changed_python_minor_can_inspect_and_restore_but_cannot_append(self):
        alien=dict(cli.cohort(),python_minor='99.99')
        with mock.patch.object(cli,'cohort',return_value=alien):
            plan,identity,rows,*_=cli.load_case(self.case)
            self.assertEqual(rows,[])
            with self.assertRaisesRegex(ValueError,'COHORT_INCOMPATIBLE'):
                cli.run(self.case,True,1,10,1)

    def test_unsafe_export_paths_are_rejected_before_write(self):
        bad=self.root/'bad.zip'
        with zipfile.ZipFile(bad,'w') as z:
            z.writestr('../escape.json','{}');z.writestr('EXPORT_MANIFEST.json','{}')
        target=self.root/'never-created'
        with self.assertRaisesRegex(ValueError,'Unsafe'):cli.restore(bad,target,'1'*40)
        self.assertFalse(target.exists())

    def test_open_session_marks_unknown_cost_instead_of_inventing_exact_cpu(self):
        cli.atomic(self.case/'SESSION_OPEN.json',dict(previous='SOFTWARE FIXTURE'))
        with mock.patch.object(cli,'ProcessPoolExecutor',FakePool):
            state=cli.run(self.case,True,1,100,1)
        self.assertTrue(state['CPU_cost_uncertain'])


if __name__=='__main__':unittest.main(verbosity=2)
