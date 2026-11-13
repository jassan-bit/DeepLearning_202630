"""Real H/P ledger and Git-proof fixtures; no training or real budget writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import e2_confirmed_budget_continuation as continuation
import test_e2_resource_budget_extension as fixtures
from test_e2_resource_budget_extension import rec, git, commit


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ResourceBudgetTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.f = fixture
        with fixture.make(fixture.cls):
            pass
        self.path = fixture.path
        self.proposal = json.loads((ROOT / continuation.PROPOSAL_PATH).read_bytes())
        funding = json.loads((self.path / 'funding_extension.json').read_bytes())
        self.proposal.update(prior_funding_registration_commit=fixture.h,
            prior_funding_policy_sha256=funding['funding_policy_sha256'],
            historical_reconciliation_sha256=rec()['sha256'])
        self.proposal.pop('sha256')
        self.proposal['sha256'] = continuation.digest(self.proposal)
        for key, value in [('FUNDING_COMMIT', fixture.h),
                ('FUNDING_SHA', funding['funding_policy_sha256']),
                ('RECONCILIATION_SHA', rec()['sha256']),
                ('PROPOSAL_SHA', self.proposal['sha256'])]:
            item = patch.object(continuation, key, value)
            item.start()
            self.addCleanup(item.stop)
        proposal_path = fixture.repo / continuation.PROPOSAL_PATH
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_bytes = continuation.canonical(self.proposal) + b'\n'
        proposal_path.write_bytes(proposal_bytes)
        source = (ROOT / continuation.SOURCE_PATH).read_bytes()
        source_path = fixture.repo / continuation.SOURCE_PATH
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source)
        record = {'protocol_id': 'e2-reduced-delivery-v1',
            'engine_commit': continuation.ENGINE_COMMIT,
            'proposal_sha256': hashlib.sha256(proposal_bytes).hexdigest(),
            'proposal_policy_sha256': self.proposal['sha256'],
            'files_sha256': {continuation.SOURCE_PATH: hashlib.sha256(source).hexdigest()}}
        (fixture.repo / continuation.REGISTRATION_PATH).write_bytes(continuation.canonical(record) + b'\n')
        self.registration = commit(fixture.repo)
        self.approval = continuation.build_approval_receipt(
            continuation_commit=self.registration, confirmed_text='80')
        self.cls = self.constructor()

    def constructor(self, approval=None):
        return continuation.make_continued_budget_class(self.f.cls, self.proposal,
            registration_commit=self.registration, registration_git_root=self.f.repo,
            approval_receipt=self.approval if approval is None else approval)

    def make(self):
        return self.f.make(self.cls)

    def test_explicit_input_and_actual_proposal_registration_required(self):
        for text in [None, '', '0', '72', '80 ', 'yes', 80]:
            with self.assertRaises((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired)):
                continuation.build_approval_receipt(continuation_commit=self.registration, confirmed_text=text)
        wrong = deepcopy(self.approval)
        wrong['approved_total_hours'] = 88
        wrong.pop('sha256')
        wrong['sha256'] = continuation.digest(wrong)
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'confirmation'):
            self.constructor(wrong)
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'registration_object'):
            continuation.make_continued_budget_class(self.f.cls, self.proposal,
                registration_commit='f'*40, registration_git_root=self.f.repo,
                approval_receipt=continuation.build_approval_receipt(
                    continuation_commit='f'*40, confirmed_text='80'))
        self.assertFalse((self.path / continuation.ANCHOR_NAME).exists())

    def test_original_anchors_prefix_and_whole_worker_bounds_preserved(self):
        prefix = (self.path / 'pilot_budget.jsonl').read_bytes()
        anchors = {name: (self.path / name).read_bytes() for name in
            ['reduced_migration.json', 'funding_extension.json', 'resource_extension.json']}
        with self.f.make(self.f.cls) as old:
            before = old.remaining_seconds
        with self.make() as new:
            self.assertAlmostEqual(new.remaining_seconds, before + 28800)
            self.assertEqual(new.reservation_for(), 1860)
            self.assertEqual(new.reservation_for(3), 5460)
            self.assertEqual(new.report()['effective_authorized_total_hours'], 80)
            self.assertEqual(new.report()['funding_registration_commit'], self.f.h)
            new.reserve('continued', 1860)
            self.f.clock.advance(11)
            new.charge('continued', 11, status='completed')
        self.assertTrue((self.path / 'pilot_budget.jsonl').read_bytes().startswith(prefix))
        for name, raw in anchors.items():
            self.assertEqual((self.path / name).read_bytes(), raw)
        rows = [json.loads(line) for line in (self.path / 'pilot_budget.jsonl').read_bytes().splitlines()]
        row = next(r for r in rows if r.get('attempt_id') == 'continued')
        self.assertEqual(row['funding_registration_commit'], self.f.h)
        self.assertEqual(row['continuation_registration_commit'], self.registration)
        self.assertEqual(row['effective_total_hours'], 80)

    def test_replay_after_spending_past_72_does_not_clip_or_grant_credit_twice(self):
        with self.f.make(self.f.cls) as old:
            self.f.clock.advance(34000)
            old.charge_overhead(0)
            before = old.remaining_seconds
        with self.make() as new:
            self.f.clock.advance(4000)
            new.charge_overhead(0)
            expected = before + 28800 - 4000
            self.assertAlmostEqual(new.remaining_seconds, expected)
        anchor = (self.path / continuation.ANCHOR_NAME).read_bytes()
        # Fresh confirmation timestamp is not another grant of time.
        self.approval = continuation.build_approval_receipt(
            continuation_commit=self.registration, confirmed_text='80')
        self.cls = self.constructor()
        with self.make() as again:
            self.assertAlmostEqual(again.remaining_seconds, expected)
        self.assertEqual((self.path / continuation.ANCHOR_NAME).read_bytes(), anchor)

    def test_unknown_lease_blocks_continuation_and_no_consent_is_published(self):
        old = self.f.make(self.f.cls)
        old.reserve('unknown', 1860)
        self.f.clock.advance(2)
        old.close()
        before = (self.path / 'pilot_budget.jsonl').read_bytes()
        with self.assertRaises((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired)):
            self.make()
        self.assertEqual((self.path / 'pilot_budget.jsonl').read_bytes(), before)
        self.assertFalse((self.path / continuation.ANCHOR_NAME).exists())
        self.assertFalse((self.path / continuation.APPROVAL_NAME).exists())

    def test_existing_writer_or_repeated_scientific_job_cannot_be_bypassed(self):
        first = self.make()
        try:
            with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'active'):
                self.make()
            with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'already_attempted'):
                first.reserve('another', 1860, metadata={'logical_job_id': 'prior-failed'})
        finally:
            first.close()

    def test_changed_old_prefix_or_continued_authority_is_rejected(self):
        with self.make() as new:
            new.reserve('one', 1860)
            self.f.clock.advance(1)
            new.charge('one', 1, status='completed')
        path = self.path / 'pilot_budget.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[-1].pop('continuation_approval_sha256')
        rows[-1].pop('sha256')
        rows[-1]['sha256'] = continuation.digest(rows[-1])
        path.write_bytes(b''.join(continuation.canonical(r) + b'\n' for r in rows))
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'immutable_confirmation'):
            self.make()

    def test_old_72_hour_prefix_cannot_be_rewritten_even_with_a_valid_new_chain(self):
        with self.make():
            pass
        path = self.path / 'pilot_budget.jsonl'
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        rows[1]['elapsed_seconds'] += 1
        previous = '0'*64
        for row in rows:
            row.pop('sha256')
            row['previous_sha256'] = previous
            row['sha256'] = continuation.digest(row)
            previous = row['sha256']
        path.write_bytes(b''.join(continuation.canonical(r) + b'\n' for r in rows))
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'prefix_bytes_changed'):
            self.make()

    def test_original_H_anchor_and_stored_confirmation_are_not_replaceable(self):
        with self.make():
            pass
        anchor = self.path / 'funding_extension.json'
        original = anchor.read_bytes()
        value = json.loads(original)
        value['effective_total_hours'] = 80
        value.pop('sha256')
        value['sha256'] = continuation.digest(value)
        anchor.write_bytes(continuation.canonical(value) + b'\n')
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'anchors_changed'):
            self.make()
        anchor.write_bytes(original)
        approval_path = self.path / continuation.APPROVAL_NAME
        value = json.loads(approval_path.read_bytes())
        value['approved_total_hours'] = 88
        value.pop('sha256')
        value['sha256'] = continuation.digest(value)
        approval_path.write_bytes(continuation.canonical(value) + b'\n')
        with self.assertRaisesRegex((ValueError, continuation.BudgetError, continuation.BudgetReconciliationRequired), 'consent_changed'):
            self.make()


if __name__ == '__main__':
    unittest.main()
