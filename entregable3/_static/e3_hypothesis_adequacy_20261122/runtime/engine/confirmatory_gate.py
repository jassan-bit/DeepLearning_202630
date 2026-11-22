"""Prospective final-operation guard for this planning-only data version.

No market outcome is computed by this guard. Internal historical controller
fixture APIs retain their explicit software-only tests; no CLI flag or JSON
boolean bypass enables a market confirmation in this version.
"""
from __future__ import annotations

from pathlib import Path
from .common import read_json

VERSION = 'crypto-planning-only-confirmation-gate-1'
CODES = ('PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED',
         'JOINT_DECISION_VALIDATION_NOT_ESTABLISHED')


class ConfirmatoryGateError(RuntimeError):
    def __init__(self, operation, *, invalid=False):
        self.operation = operation
        self.gate_codes = CODES
        self.invalid_metadata = invalid
        super().__init__('Final market execution is not enabled in this source version: '+
                         ' '.join(CODES)+('; gate metadata differs' if invalid else ''))


def block_final_operation(source_root, operation):
    """Always block this version, even if an approval boolean was edited."""
    path = Path(source_root)/'configs/confirmatory_gate.json'
    try:
        value = read_json(path)
        valid = (value.get('version') == VERSION and tuple(value.get('gate_codes',())) == CODES
                 and value.get('final_market_execution_enabled') is False
                 and value.get('primary_endpoint_alignment_established') is False
                 and value.get('joint_decision_validation_established') is False)
    except (OSError,ValueError,TypeError):
        valid = False
    raise ConfirmatoryGateError(operation,invalid=not valid)


def preflight(args):
    """Read stage metadata only; run before credentials, network, fit or test."""
    command = args.command
    # Preserve the existing explicit-activation refusal before reading a case.
    if command in ('run','evaluate') and not args.activate:
        return
    if command in ('freeze-protocol','seal-training','evaluate'):
        block_final_operation(args.source_root,command)
    if command == 'prepare' and args.stage == 'final':
        block_final_operation(args.source_root,'prepare-final')
    if command == 'run':
        plan_path = Path(args.case)/'plan.json'
        if not plan_path.is_file():
            raise ValueError('Prepare and verify a planning case before activating run; no credentials read')
        plan = read_json(plan_path)
        if plan.get('stage') == 'final':
            block_final_operation(args.source_root,'run-final')
        if plan.get('stage') != 'planning':
            raise ValueError('Unknown campaign stage; no credentials read')
