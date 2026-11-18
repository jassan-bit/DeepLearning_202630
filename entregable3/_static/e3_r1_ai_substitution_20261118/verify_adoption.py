"""Verify declared AI reviewer adoption, its evidence and its limited scope."""
from pathlib import Path
import hashlib
import json


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, row in manifest['files'].items():
        raw = (root / name).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], name
    instruction = json.loads((root / 'instructor_instruction.json').read_text())
    adoption = json.loads((root / 'review_adoption.json').read_text())
    assert instruction['ai_substitution_authorized'] is True
    assert instruction['requires_additional_confirmation_to_apply'] is False
    assert adoption['ai_reviewer_substitution_authorized'] is True
    assert adoption['AI_use_disclosed'] is True
    assert adoption['reviewer_component_status'] == 'COMPLETED_USING_DECLARED_AI'
    assert adoption['reviewer_component_completed'] is True
    assert adoption['human_external_review_performed'] is False
    assert adoption['human_reviewer_name'] is None and adoption['human_signature_asserted'] is False
    assert adoption['instructor_substantive_approval_asserted'] is False
    assert adoption['R1_accredited'] is False
    assert adoption['substantive_algorithmic_difference_established'] is False
    assert adoption['candidate']['trainable_parameters'] == 1210
    assert adoption['real_campaign_allowed'] is False
    for key in ('new_real_data_fits','new_market_predictions','new_statistical_contrasts'):
        assert type(adoption[key]) is int and adoption[key] == 0
    declaration = (root / adoption['declaration']).read_text()
    assert 'inteligencia artificial' in declaration and 'revisión externa de R1' in declaration
    evidence = json.loads((root / adoption['query_check']).read_text())
    assert evidence['attribution']['reviewer_type'] == 'AI'
    assert evidence['status']['native_ui_successful_queries'] == adoption['native_query_repetitions_observed'] == 2
    assert evidence['status']['two_native_ui_lexical_query_repetitions_observed'] is True
    assert evidence['status']['human_external_review_completed'] is False
    expected = {'Q2':'CNN LSTM attention cryptocurrency','Q5':'neural network output constraints'}
    assert {q['query_id']:q['exact_query'] for q in evidence['queries']} == expected
    assert all(q['successful_bibliographic_response'] is True for q in evidence['queries'])
    qmanifest = json.loads((root / 'query_check/manifest.json').read_text())
    for row in qmanifest['files']:
        raw = (root / 'query_check' / row['path']).read_bytes()
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
    repository = Path(__file__).resolve().parents[2]
    for report in adoption['technical_reports']:
        raw = (repository / report['path']).read_bytes()
        assert len(raw) == report['bytes'] and hashlib.sha256(raw).hexdigest() == report['sha256']
    return {'status':'passed_AI_modality_adoption_and_integrity',
            'reviewer_component_completed':True,'review_modality':'AI with explicit disclosure',
            'native_query_repetitions_observed':2,'referenced_reports_verified':len(adoption['technical_reports']),
            'human_external_review_performed':False,'R1_accredited':False,
            'new_real_data_fits':0,'new_market_predictions':0}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
