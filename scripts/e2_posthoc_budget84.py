"""Append one verified 80-to-84-hour amendment; preserve all original costs.

This accounting adapter leaves the original scientific source and 80-hour
anchor intact. The extra allowance exists only after its zero-cost, receipt-bound
amendment row appears in the append-only posthoc journal. A replay therefore
checks old reservations against their original allowance until that exact row.
"""
from __future__ import annotations
import hashlib,json,math,re

VERSION='e2-posthoc-budget84-adapter-v1'
AMENDMENT_ID='e2-posthoc-budget-84-v1'
DESCRIPTION='authorized_budget_extension_80_to_84'
ADDED_SECONDS=4*3600
MARKER='_e2_posthoc_budget84_authorization_v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def validate_authorization(value):
    if not isinstance(value,dict):raise ValueError('A verified GitHub budget authorization is required')
    # Freeze caller-owned data so later mutation cannot change ledger meaning.
    auth=json.loads(canonical(value))
    # A manifest can be returned at a later branch head on an idempotent upload.
    # Its commit is transport provenance, never the identity of the allowance.
    auth.pop('receipt_remote_commit',None)
    expected={'amendment_id':AMENDMENT_ID,'previous_total_hours':80,'total_hours':84,
              'added_seconds':ADDED_SECONDS,'user_statement':'si amplialo',
              'persistent_destination':'GitHub'}
    if any(auth.get(k)!=v for k,v in expected.items()):
        raise ValueError('Only the authorized 80-to-84-hour extension is admitted')
    if (auth.get('receipt_manifest_path')!='runs/e2-budget84-approval-v1/manifest.json'
            or not isinstance(auth.get('receipt_sha256'),str)
            or re.fullmatch(r'[0-9a-f]{64}',auth['receipt_sha256'])is None
            or not isinstance(auth.get('approval_sha256'),str)
            or re.fullmatch(r'[0-9a-f]{64}',auth['approval_sha256'])is None):
        raise ValueError('Durable GitHub authorization receipt identity is required')
    return auth


def _candidate(row):
    if not isinstance(row,dict):raise ValueError('Budget journal row must be an object')
    description=row.get('description','')
    evidence=row.get('evidence',{})
    return ((isinstance(description,str)and description.startswith('authorized_budget_extension'))
            or(isinstance(evidence,dict)and'budget_amendment'in evidence))


def _check_amendment(row,auth):
    if (row.get('kind')!='overhead'or row.get('description')!=DESCRIPTION
            or type(row.get('elapsed_seconds'))not in(int,float)
            or not math.isfinite(row['elapsed_seconds'])or row['elapsed_seconds']!=0
            or not isinstance(row.get('evidence'),dict)
            or canonical(row['evidence'].get('budget_amendment'))!=canonical(auth)):
        raise ValueError('Budget extension differs from the verified authorization')


def _checked_row(row):
    if row.get('sha256')!=hashlib.sha256(canonical({k:v for k,v in row.items()if k!='sha256'})).hexdigest():
        raise ValueError('Budget amendment row checksum differs')
    return row


def _amendments(events,auth,checked):
    found=[]
    for row in events:
        if _candidate(row):
            checked(row);_check_amendment(row,auth);found.append(row)
    if len(found)>1:raise ValueError('Duplicate budget extension is prohibited')
    return found


def install(old,authorization):
    """Bind accounting methods to one externally verified authorization receipt.

    The caller must validate and durably archive consent in GitHub before
    calling this function. This adapter neither uploads consent nor starts fits.
    """
    auth=validate_authorization(authorization);binding=canonical(auth)
    if (getattr(old,'TOTAL_SECONDS',None)!=80*3600
            or not math.isclose(old.REMAINING_SECONDS,old.TOTAL_SECONDS-old.ORIGINAL_CHARGED_SECONDS,abs_tol=1e-9)):
        raise ValueError('The unchanged original 80-hour budget anchor is required')
    cls=old.PosthocLedger;prior=cls.__dict__.get(MARKER)
    if prior is not None:
        if prior!=binding:raise ValueError('Another budget authorization is already installed')
        return {'version':VERSION,'authorization_sha256':hashlib.sha256(binding).hexdigest(),'installed':False}
    original_apply=cls._apply;original_report=cls.report

    def apply(self,row):
        if _candidate(row):
            _check_amendment(row,auth)
            if self.active:raise ValueError('An unresolved reservation cannot be forgiven by an extension')
            if _amendments(self.events,auth,old.checked):raise ValueError('Duplicate budget extension is prohibited')
        return original_apply(self,row)

    def remaining(self):
        credit=ADDED_SECONDS if _amendments(self.events,auth,old.checked)else 0
        return max(0.,old.REMAINING_SECONDS+credit-self.charges-sum(self.active.values()))

    def report(self):
        result=original_report(self)
        granted=bool(_amendments(self.events,auth,old.checked))
        result.update(approved_total_hours=84 if granted else 80,
                      original_approved_total_hours=80,
                      authorized_extension_seconds=ADDED_SECONDS if granted else 0,
                      budget_reset=False)
        if granted:result['budget_amendment']=json.loads(binding)
        return result

    cls._apply=apply;cls.remaining_seconds=property(remaining);cls.report=report
    setattr(cls,MARKER,binding)
    return {'version':VERSION,'authorization_sha256':hashlib.sha256(binding).hexdigest(),'installed':True}


def apply_extension(budget,authorization):
    """Append the grant once, only while durable persistence is attached."""
    auth=validate_authorization(authorization)
    if type(budget).__dict__.get(MARKER)!=canonical(auth):
        raise ValueError('Install this exact authorization on the original ledger first')
    amendments=_amendments(budget.events,auth,_checked_row)
    if amendments:
        return {'status':'budget_extension_already_recorded','approved_total_hours':84,
                'added_seconds':ADDED_SECONDS,'remaining_seconds':budget.remaining_seconds,
                'amendment_record_sha256':amendments[0]['sha256'],'budget_reset':False}
    if budget.active:raise ValueError('An unresolved reservation cannot be forgiven by an extension')
    if getattr(budget,'persistence',None)is None:raise RuntimeError('GitHub persistence is required before recording the budget extension')
    row=budget._append('overhead',elapsed_seconds=0.,description=DESCRIPTION,
                       evidence={'budget_amendment':auth})
    return {'status':'budget_extension_recorded','approved_total_hours':84,
            'added_seconds':ADDED_SECONDS,'remaining_seconds':budget.remaining_seconds,
            'amendment_record_sha256':row['sha256'],'budget_reset':False}
