"""Create the exact audit tree through small immutable Git tree objects.

Operational IO only, with disclosed AI assistance. All requests use the caller's
original paced transport. This module never creates blobs, commits or refs.
"""
import hashlib
from pathlib import PurePosixPath
import re

ENDPOINT = '/repos/jassan-bit/DeepLearning_202630/git/trees'
SHA = re.compile(r'[0-9a-f]{40}')
STAMP = re.compile(r'[0-9]{8}T[0-9]{12}Z')
MODES = {'100644': 'blob', '100755': 'blob', '120000': 'blob',
         '040000': 'tree', '160000': 'commit'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def component(name):
    require(isinstance(name, str) and name not in ('', '.', '..')
            and '/' not in name and '\0' not in name, 'Invalid Git tree basename')
    try:
        return name.encode('utf-8')
    except UnicodeError:
        raise ValueError('Git tree basename is not UTF-8') from None


def normalized(row):
    require(isinstance(row, dict), 'Git tree row must be an object')
    name, mode, kind, sha = (row.get(k) for k in ('path', 'mode', 'type', 'sha'))
    component(name)
    require(isinstance(mode, str) and isinstance(kind, str) and mode in MODES and kind == MODES[mode]
            and isinstance(sha, str) and SHA.fullmatch(sha), 'Invalid Git tree entry')
    return {'path': name, 'mode': mode, 'type': kind, 'sha': sha}


def tree_sha(rows):
    clean = [normalized(row) for row in rows]
    require(len({r['path'] for r in clean}) == len(clean), 'Duplicate Git tree basenames')
    clean.sort(key=lambda r: component(r['path']) + (b'/' if r['type'] == 'tree' else b''))
    raw = b''.join((r['mode'].lstrip('0') or '0').encode() + b' ' + component(r['path'])
                   + b'\0' + bytes.fromhex(r['sha']) for r in clean)
    return hashlib.sha1(b'tree ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def validate_payload(payload):
    require(isinstance(payload, dict) and set(payload) == {'base_tree', 'tree'}
            and isinstance(payload['base_tree'], str) and SHA.fullmatch(payload['base_tree']),
            'Only an exact base_tree and tree audit request can be split')
    rows = payload['tree']
    require(isinstance(rows, list) and 0 < len(rows) <= 4096, 'Invalid flat audit tree size')
    common = None
    root = {}
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'mode', 'type', 'sha'}
                and row['mode'] == '100644' and row['type'] == 'blob'
                and isinstance(row['sha'], str) and SHA.fullmatch(row['sha']),
                'Only recorded regular audit blobs can be split')
        path = row['path']
        require(isinstance(path, str) and 0 < len(path) <= 1024
                and '\\' not in path and not any(ord(c) < 32 or ord(c) == 127 for c in path),
                'Unsafe flat audit path')
        p = PurePosixPath(path)
        require(str(p) == path and not p.is_absolute() and '..' not in p.parts
                and 4 <= len(p.parts) <= 64 and p.parts[:2] == ('runs', 'e3_extended_audit')
                and STAMP.fullmatch(p.parts[2]), 'Flat tree must add one audit part')
        require(path not in seen, 'Duplicate flat audit path')
        seen.add(path)
        prefix = p.parts[:3]
        if common is None:
            common = prefix
        require(prefix == common, 'Flat tree contains multiple audit parts')
        node = root
        for name in p.parts[3:-1]:
            component(name)
            require(name not in node or isinstance(node[name], dict), 'Audit file/directory conflict')
            node = node.setdefault(name, {})
        name = p.parts[-1]
        component(name)
        require(name not in node, 'Audit file/directory conflict')
        node[name] = row['sha']
    return common[2], root


def build_incremental_tree(request, token, payload, emit=None):
    """Return the exact flat-request root SHA using bottom-up direct children.

    Validate the whole flat request and all base ancestors before mutations.
    Existing tree objects are verified/reused, so interrupted runs are resumable.
    """
    part, forest = validate_payload(payload)
    emit = emit or (lambda report: None)
    cache = {}
    counts = {'created': 0, 'reused': 0}

    def checked_tree(value, expected):
        require(isinstance(value, dict) and value.get('sha') == expected
                and value.get('truncated') is False and isinstance(value.get('tree'), list),
                'A complete immutable Git tree is required')
        rows = [normalized(row) for row in value['tree']]
        require(tree_sha(rows) == expected, 'Git tree content differs from its SHA')
        return {r['path']: r for r in rows}

    def read(sha):
        if sha not in cache:
            cache[sha] = checked_tree(request('GET', ENDPOINT + '/' + sha, token), sha)
        return dict(cache[sha])

    def child(rows, name):
        found = rows.get(name)
        if found is None:
            return {}
        require(found['mode'] == '040000' and found['type'] == 'tree',
                'Audit ancestor is not a directory')
        return read(found['sha'])

    base_rows = read(payload['base_tree'])
    runs_rows = child(base_rows, 'runs')
    audit_rows = child(runs_rows, 'e3_extended_audit')
    require(part not in audit_rows, 'Audit part already exists in the parent tree')

    def ensure(rows):
        values = list(rows.values())
        expected = tree_sha(values)
        if expected in cache:
            require(cache[expected] == rows, 'Existing tree cache content differs')
            counts['reused'] += 1
        else:
            existing = request('GET', ENDPOINT + '/' + expected, token, allow_missing=True)
            if existing is not None:
                require(checked_tree(existing, expected) == rows, 'Existing immutable tree differs')
                counts['reused'] += 1
            else:
                values.sort(key=lambda r: component(r['path']) + (b'/' if r['type'] == 'tree' else b''))
                result = request('POST', ENDPOINT, token, {'tree': values})
                require(isinstance(result, dict) and result.get('sha') == expected,
                        'Created Git tree SHA differs from the intended tree')
                if result.get('truncated') is not False or not isinstance(result.get('tree'), list):
                    result = request('GET', ENDPOINT + '/' + expected, token)
                require(checked_tree(result, expected) == rows, 'Created Git tree content differs')
                counts['created'] += 1
            cache[expected] = dict(rows)
        emit({'event': 'incremental_git_tree', **counts, 'entries': len(rows), 'sha': expected})
        return expected

    def build(node):
        rows = {}
        for name, value in node.items():
            if isinstance(value, dict):
                rows[name] = {'path': name, 'mode': '040000', 'type': 'tree', 'sha': build(value)}
            else:
                rows[name] = {'path': name, 'mode': '100644', 'type': 'blob', 'sha': value}
        return ensure(rows)

    part_sha = build(forest)
    audit_rows[part] = {'path': part, 'mode': '040000', 'type': 'tree', 'sha': part_sha}
    audit_sha = ensure(audit_rows)
    runs_rows['e3_extended_audit'] = {'path': 'e3_extended_audit', 'mode': '040000',
                                    'type': 'tree', 'sha': audit_sha}
    runs_sha = ensure(runs_rows)
    base_rows['runs'] = {'path': 'runs', 'mode': '040000', 'type': 'tree', 'sha': runs_sha}
    return {'sha': ensure(base_rows)}
