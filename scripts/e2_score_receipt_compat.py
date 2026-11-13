"""Adapt one immutable pre-data failure disposition to the backup receipt API.

The source helper and its existing JSON artifact remain unchanged. Only the
transport receipt describes the original request's terminal failed status.
Actual scoring receipts pass through untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

DISPOSITION_FILENAME = 'pre_payload_failure_disposition.json'
DISPOSITION_SHA256 = '2fd9f653963277dd0b6e5c0fc98a305da0f5d56cb59ff0e3726612eed8c1dc55'
DISPOSITION_STATUS = 'pre_payload_failure_reconciled'
SOURCE_REGISTRATION_COMMIT = '24344bd6db20f0546b26167eda4c3076a38c0f4b'
FAILED_JOB = 'ff53264094dc1e03fe8586b6aaabeb7b91b951e4ed54b7d9b6e9c020a7c2435a'
RECEIPT_ATTEMPT = 'score-source-recovery-admission-' + FAILED_JOB[:32]


def compatible_persist_result(persist_result, *, registration_commit):
    """Return a narrowly bound backup callback; perform no reads until called."""
    if not callable(persist_result):
        raise TypeError('An existing GitHub persistence callback is required')
    if not isinstance(registration_commit,str) or not re.fullmatch('[0-9a-f]{40}',registration_commit):
        raise ValueError('Use the immutable receipt compatibility registration commit')
    def persist(directory,receipt):
        if not isinstance(receipt,dict) or receipt.get('status') != DISPOSITION_STATUS:
            return persist_result(directory,receipt)
        root=Path(directory).absolute()
        artifact=root/DISPOSITION_FILENAME
        if (root.name!='scoring_source_recovery' or root.parent.name!='final'
                or any(p.is_symlink() for p in (artifact,*artifact.parents))
                or not artifact.is_file()):
            raise ValueError('The registered pre-payload disposition path is required')
        raw=artifact.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=DISPOSITION_SHA256:
            raise ValueError('The original acquisition disposition changed; it was preserved')
        disposition=json.loads(raw)
        expected={**disposition,'attempt_id':RECEIPT_ATTEMPT,
            'artifacts_sha256':{DISPOSITION_FILENAME:DISPOSITION_SHA256}}
        if (disposition.get('status')!=DISPOSITION_STATUS
                or disposition.get('version')!='e2-score-source-recovery-v1'
                or disposition.get('original_failed_job_id')!=FAILED_JOB
                or disposition.get('policy_proof',{}).get('registration_commit')!=SOURCE_REGISTRATION_COMMIT
                or receipt!=expected):
            raise ValueError('Only the exact original metadata disposition receipt can be adapted')
        normalized={**receipt,'status':'failed','reconciliation_status':DISPOSITION_STATUS,
            'kind':'pre_payload_failure_disposition','scientific_result':False,
            'benchmark_complete':False,'backup_compat_registration_commit':registration_commit}
        return persist_result(directory,normalized)
    return persist
