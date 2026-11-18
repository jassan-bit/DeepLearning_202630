"""Offline integrity and scope verification; does not judge academic novelty."""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import json
import zipfile


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['kind'] == 'e3_r1_review_packet_manifest'
    expected = manifest['files']
    assert expected
    for name, row in expected.items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        raw = path.read_bytes()
        assert len(raw) == row['bytes'], name
        assert hashlib.sha256(raw).hexdigest() == row['sha256'], name
        if path.suffix == '.json':
            json.loads(raw)
        elif path.suffix == '.py':
            ast.parse(raw, filename=name)
    status = json.loads((root / 'status.json').read_text())
    assert status['candidate']['trainable_parameters'] == 1210
    assert status['candidate']['variant'] == 'additive_positive_head'
    assert status['R1_accredited'] is False
    assert status['R1_substantively_rejected'] is False
    assert status['real_campaign_allowed'] is False
    assert status['existing_training_repeated'] is False
    for field in ('new_model_fits', 'new_market_predictions', 'new_statistical_contrasts'):
        assert type(status[field]) is int and status[field] == 0
    review = json.loads((root / 'reviewer/pending_review.json').read_text())
    assert review['status'] == 'NOT_PERFORMED'
    assert review['reviewer_name'] is None and review['human_concept'] is None
    assert review['human_recommendation'] is None and review['queries_repeated'] == []
    assert review['person_contacted'] is False and review['R1_certified'] is False
    assert review['source_commit'] == status['baseline_source_commit']
    identity = json.loads((root / 'reviewer/code_identity.json').read_text())
    assert identity['source_commit'] == status['baseline_source_commit']
    assert identity['candidate']['trainable_parameters'] == 1210
    notebook = json.loads((root / 'revision_bibliografica_colab.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == [] and cell['execution_count'] is None
            ast.parse(''.join(cell['source']))
    # The archive is optional in an extraction; in the repository verify every member.
    archive = root / 'expediente_r1.zip'
    if archive.exists():
        metadata = json.loads((root / 'bundle.json').read_text())
        raw = archive.read_bytes()
        assert metadata['bytes'] == len(raw)
        assert metadata['sha256'] == hashlib.sha256(raw).hexdigest()
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
            assert len(names) == len(set(names))
            assert set(names) == set(expected) | {'manifest.json'}
            for name in names:
                assert package.read(name) == (root / name).read_bytes(), name
    result = {'status': 'passed_integrity_and_scope_only', 'files_verified': len(expected),
              'candidate_parameters': 1210, 'R1_accredited': False,
              'human_external_review': 'NOT_PERFORMED', 'new_model_fits': 0,
              'new_market_predictions': 0, 'real_campaign_allowed': False}
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
