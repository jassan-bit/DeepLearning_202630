"""Publication safety regressions. All remotes are temporary local bare repositories."""
import copy
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import io
import inspect
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import safe_publish_books as pub



def cases(name, values):
    """Expand small independent cases without adding a test dependency."""
    def decorate(function):
        function.cases = (name, values)
        return function
    return decorate


def expect_error(exception, match=None):
    assertion = unittest.TestCase()
    return (assertion.assertRaisesRegex(exception, match) if match
            else assertion.assertRaises(exception))


class Patches:
    def __init__(self, owner):
        self.owner = owner

    def apply(self, context):
        context.start()
        self.owner.addCleanup(context.stop)

    def setenv(self, name, value):
        self.apply(patch.dict(os.environ, {name: value}))

    def delenv(self, name, raising=True):
        self.apply(patch.dict(os.environ))
        os.environ.pop(name, None)

    def setattr(self, target, name, value):
        self.apply(patch.object(target, name, value))

def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode().strip()


def make_site(path, sha):
    path.mkdir(parents=True, exist_ok=True)
    records = []
    for identity, (source, prefix) in pub.BOOKS.items():
        directory = path / prefix
        directory.mkdir(exist_ok=True)
        for name in ('intro.html', 'index.html'):
            (directory / name).write_text('<html><body id="top">Book</body></html>')
        for name in ('objects.inv', 'searchindex.js'):
            (directory / name).write_text('book asset')
        entry = (prefix + '/' if prefix else '') + 'intro.html'
        records.append(dict(id=identity, source=source, prefix=prefix, entrypoint=entry,
                            intro_sha256=pub.sha256((path / entry).read_bytes()).hexdigest()))
    (path / '.nojekyll').touch()
    (path / 'entregables.html').write_text('<a href="intro.html#top">Book</a>')
    (path / 'chapter.html').write_text('<p>Existing chapter</p>')
    (path / '_downloads').mkdir(exist_ok=True)
    (path / '_downloads/data.bin').write_bytes(b'original downloadable data')
    (path / 'report.pdf').write_bytes(b'%PDF-test')
    (path / 'books_manifest.json').write_text(json.dumps(dict(source_commit=sha, books=records)))
    pub.write_inventory(path, sha)


def world(tmp_path, monkeypatch):
    # Avoid ambient global Git configuration, credentials, hooks, or transports.
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.delenv('GH_TOKEN', raising=False)
    remote, repo, site = [tmp_path / name for name in ('remote.git', 'repo', 'site')]
    remote.mkdir(); repo.mkdir()
    git(remote, 'init', '--bare')
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'Local Test')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    (repo / 'source.txt').write_text('first source')
    git(repo, 'add', '.'); git(repo, 'commit', '-m', 'source one')
    old_source = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'remote', 'add', 'origin', str(remote))
    git(repo, 'push', 'origin', 'main')
    make_site(site, old_source)
    git(repo, 'checkout', '--orphan', 'gh-pages')
    git(repo, 'rm', '-rf', '.')
    shutil.copytree(site, repo, dirs_exist_ok=True)
    git(repo, 'add', '.'); git(repo, 'commit', '-m', 'initial published site')
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'push', 'origin', 'gh-pages')
    git(repo, 'checkout', 'main')
    (repo / 'source.txt').write_text('second source')
    git(repo, 'commit', '-am', 'source two')
    sha = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'push', 'origin', 'main')
    make_site(site, sha)
    return dict(repo=repo, remote=remote, site=site, sha=sha, old=old_source, base=base)


def publish(w, **kwargs):
    return pub.publish(w['repo'], w['site'], w['sha'], **kwargs)


def heads(w):
    return git(w['remote'], 'show-ref')


def test_dry_run_leaves_remote_and_checkout_unchanged(world):
    before, status = heads(world), git(world['repo'], 'status', '--porcelain')
    result = publish(world)
    assert result['mode'] == 'dry-run' and result['pushed'] is False
    assert result['web_deployment_verified'] is False
    assert result['unchanged_downloads'] == 2
    assert heads(world) == before
    assert git(world['repo'], 'status', '--porcelain') == status
    assert git(world['repo'], 'rev-parse', 'HEAD') == world['sha']


def test_execute_is_normal_descendant_and_preserves_checkout(world, tmp_path):
    report = tmp_path / 'report.json'
    result = publish(world, execute=True, report_path=report)
    candidate = result['candidate_pages_commit']
    assert git(world['remote'], 'rev-parse', 'gh-pages') == candidate
    assert git(world['remote'], 'rev-parse', 'gh-pages^') == world['base']
    assert git(world['remote'], 'rev-parse', 'main') == world['sha']
    assert git(world['repo'], 'rev-parse', 'HEAD') == world['sha']
    assert not git(world['repo'], 'status', '--porcelain')
    assert result['pushed'] and not result['web_deployment_verified']
    assert json.loads(report.read_text()) == result
    for name, info in pub.file_inventory(world['site']).items():
        data = subprocess.check_output(['git', '-C', str(world['remote']), 'show', f'gh-pages:{name}'])
        assert pub.sha256(data).hexdigest() == info['sha256']


@cases('branch', ['main', 'gh-pages'])
def test_requires_existing_branches(world, branch):
    git(world['remote'], 'update-ref', '-d', 'refs/heads/' + branch)
    before = heads(world)
    with expect_error(pub.PublicationError):
        publish(world, execute=True)
    assert heads(world) == before


def test_obsolete_source_cannot_publish(world):
    git(world['repo'], 'checkout', world['old'])
    make_site(world['site'], world['old'])
    before = heads(world)
    with expect_error(pub.PublicationError, match='Obsolete'):
        pub.publish(world['repo'], world['site'], world['old'], execute=True)
    assert heads(world) == before


@cases('staged', [False, True])
def test_dirty_source_cannot_publish(world, staged):
    (world['repo'] / 'source.txt').write_text('uncommitted')
    if staged:
        git(world['repo'], 'add', 'source.txt')
    before = heads(world)
    with expect_error(pub.PublicationError):
        publish(world, execute=True)
    assert heads(world) == before


@cases('branch', ['main', 'gh-pages'])
def test_remote_race_before_final_check(world, monkeypatch, branch):
    original = pub.candidate_commit
    advanced = []
    def race(repo, site, source_sha, base, expected_files):
        result = original(repo, site, source_sha, base, expected_files)
        parent = world['sha'] if branch == 'main' else world['base']
        tree = git(repo, 'rev-parse', parent + '^{tree}')
        child = git(repo, 'commit-tree', tree, '-p', parent, '-m', 'concurrent writer')
        git(repo, 'push', 'origin', child + ':refs/heads/' + branch)
        advanced.append(child)
        return result
    monkeypatch.setattr(pub, 'candidate_commit', race)
    with expect_error(pub.PublicationError):
        publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', branch) == advanced[0]
    if branch == 'main':
        assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


@cases('path', ['intro.html', 'entregable2/index.html', 'entregable3/objects.inv',
                                 'searchindex.js', 'entregables.html', '.nojekyll'])
def test_missing_required_site_file(world, path):
    (world['site'] / path).unlink()
    with expect_error(pub.PublicationError):
        pub.validate_site(world['site'], world['sha'])


@cases('path', ['chapter.html', '_downloads/data.bin', 'report.pdf'])
def test_no_previous_path_can_disappear_even_with_new_inventory(world, path):
    (world['site'] / path).unlink()
    pub.write_inventory(world['site'], world['sha'])
    before = heads(world)
    with expect_error(pub.PublicationError, match='disappeared'):
        publish(world, execute=True)
    assert heads(world) == before


@cases('path', ['_downloads/data.bin', 'report.pdf'])
def test_published_download_bytes_are_immutable(world, path):
    (world['site'] / path).write_bytes(b'changed bytes')
    pub.write_inventory(world['site'], world['sha'])
    with expect_error(pub.PublicationError, match='download changed'):
        publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


def test_new_versioned_download_is_allowed(world):
    (world['site'] / 'report-v2.pdf').write_bytes(b'%PDF-new')
    pub.write_inventory(world['site'], world['sha'])
    assert publish(world, execute=True)['pushed']


@cases('mutation', ['source', 'identity', 'duplicate', 'missing', 'prefix', 'entrypoint', 'hash'])
def test_manifest_integrity(world, mutation):
    path = world['site'] / 'books_manifest.json'
    manifest = json.loads(path.read_text())
    if mutation == 'source': manifest['source_commit'] = world['old']
    elif mutation == 'identity': manifest['books'][0]['id'] = 'unknown'
    elif mutation == 'duplicate': manifest['books'][1] = copy.deepcopy(manifest['books'][0])
    elif mutation == 'missing': manifest['books'].pop()
    elif mutation == 'hash': manifest['books'][0]['intro_sha256'] = '0' * 64
    else: manifest['books'][0][mutation] = 'wrong'
    path.write_text(json.dumps(manifest))
    with expect_error(pub.PublicationError):
        pub.validate_site(world['site'], world['sha'])


@cases('mutation', ['bytes', 'extra', 'schema', 'source', 'hash', 'missing'])
def test_inventory_integrity(world, mutation):
    path = world['site'] / pub.INVENTORY
    inventory = json.loads(path.read_text())
    if mutation == 'bytes': (world['site'] / 'chapter.html').write_text('tampered')
    elif mutation == 'extra': (world['site'] / 'extra.html').write_text('extra')
    elif mutation == 'missing': path.unlink()
    else:
        if mutation == 'schema': inventory['schema_version'] = 2
        elif mutation == 'source': inventory['source_commit'] = world['old']
        else: inventory['files']['chapter.html']['sha256'] = '0' * 64
        path.write_text(json.dumps(inventory))
    with expect_error((pub.PublicationError, OSError)):
        pub.validate_site(world['site'], world['sha'])


@cases('kind', ['symlink', 'directory_symlink', 'hardlink', 'fifo', 'git', 'github'])
def test_no_special_files_or_repository_metadata(world, kind):
    target = world['site'] / 'unsafe'
    if kind == 'symlink': target.symlink_to(world['site'] / 'intro.html')
    elif kind == 'directory_symlink': target.symlink_to(world['site'] / 'entregable2', target_is_directory=True)
    elif kind == 'hardlink': os.link(world['site'] / 'intro.html', target)
    elif kind == 'fifo': os.mkfifo(target)
    else:
        target = world['site'] / ('.git' if kind == 'git' else '.github')
        target.mkdir(); (target / 'config').write_text('metadata')
    with expect_error(pub.PublicationError): pub.file_inventory(world['site'])


def test_publish_rejects_symlink_site_root(world, tmp_path):
    alias = tmp_path / 'site-alias'; alias.symlink_to(world['site'], target_is_directory=True)
    with expect_error(pub.PublicationError):
        pub.publish(world['repo'], alias, world['sha'], execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


def test_candidate_must_match_validated_inventory(world, monkeypatch):
    original = pub.candidate_commit
    def mutate(repo, site, source_sha, base, expected_files):
        (Path(site) / 'report.pdf').write_bytes(b'changed after verification')
        return original(repo, site, source_sha, base, expected_files)
    monkeypatch.setattr(pub, 'candidate_commit', mutate)
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


def test_rejected_push_never_retries_or_reports_success(world, monkeypatch, tmp_path):
    original, pushes = pub.git, []
    def reject(repo, *args, **kwargs):
        if args[0] == 'push':
            pushes.append(args)
            raise pub.PublicationError('simulated rejection')
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(pub, 'git', reject)
    report = tmp_path / 'report.json'
    with expect_error(pub.PublicationError, match='uncertain'): publish(world, execute=True, report_path=report)
    assert len(pushes) == 1
    saved = json.loads(report.read_text())
    assert saved['pushed'] is None and saved['push_attempted'] is True
    assert saved['push_outcome'] == 'unknown'
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


def test_acknowledgement_lost_after_success_is_verified_without_retry(world, monkeypatch):
    original, pushes = pub.git, []
    def lost_ack(repo, *args, **kwargs):
        result = original(repo, *args, **kwargs)
        if args[0] == 'push':
            pushes.append(args)
            raise pub.PublicationError('lost acknowledgement')
        return result
    monkeypatch.setattr(pub, 'git', lost_ack)
    result = publish(world, execute=True)
    assert result['pushed'] and len(pushes) == 1
    assert git(world['remote'], 'rev-parse', 'gh-pages') == result['candidate_pages_commit']


def test_concurrent_pages_push_after_final_check_is_rejected(world, monkeypatch):
    original, concurrent, pushes = pub.git, [], []
    def race(repo, *args, **kwargs):
        if args[0] == 'push':
            pushes.append(args)
            tree = git(repo, 'rev-parse', world['base'] + '^{tree}')
            child = git(repo, 'commit-tree', tree, '-p', world['base'], '-m', 'competing publish')
            git(repo, 'push', 'origin', child + ':refs/heads/gh-pages')
            concurrent.append(child)
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(pub, 'git', race)
    with expect_error(pub.PublicationError, match='uncertain'): publish(world, execute=True)
    assert len(pushes) == 1
    assert git(world['remote'], 'rev-parse', 'gh-pages') == concurrent[0]


def test_unverifiable_push_outcome_does_not_retry(world, monkeypatch, tmp_path):
    original, pushes = pub.git, []
    def uncertain(repo, *args, **kwargs):
        if args[0] == 'push':
            pushes.append(args)
            raise pub.PublicationError('transport lost')
        if args[0] == 'ls-remote' and pushes:
            raise pub.PublicationError('remote unreachable')
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(pub, 'git', uncertain)
    report = tmp_path / 'unknown-outcome.json'
    with expect_error(pub.PublicationError): publish(world, execute=True, report_path=report)
    assert len(pushes) == 1
    saved = json.loads(report.read_text())
    assert saved['push_outcome'] == 'unknown' and saved['pushed'] is None
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


@cases('change', ['add', 'delete', 'inventory', 'symlink'])
def test_candidate_rejects_all_postvalidation_mutations(world, monkeypatch, change):
    original = pub.candidate_commit
    def mutate(repo, site, source_sha, base, expected_files):
        site = Path(site)
        if change == 'add': (site / 'unexpected.txt').write_text('new')
        elif change == 'delete': (site / 'chapter.html').unlink()
        elif change == 'inventory': (site / pub.INVENTORY).write_text('{}')
        else: (site / 'unexpected').symlink_to(site / 'intro.html')
        return original(repo, site, source_sha, base, expected_files)
    monkeypatch.setattr(pub, 'candidate_commit', mutate)
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


@cases('link', ['missing.html', 'intro.html#absent', '../outside.txt'])
def test_broken_local_links_block_inventory(world, link):
    (world['site'] / 'chapter.html').write_text('<a href="' + link + '">bad</a>')
    with expect_error(ValueError): pub.write_inventory(world['site'], world['sha'])


@cases('value', ['', '0' * 39, 'A' * 40, 'main', None])
def test_invalid_source_sha(world, value):
    with expect_error(pub.PublicationError): pub.validate_site(world['site'], value)


def test_main_advance_during_push_is_not_reported_as_current(world, monkeypatch, tmp_path):
    original = pub.git
    def race(repo, *args, **kwargs):
        if args[0] == 'push':
            tree = git(repo, 'rev-parse', world['sha'] + '^{tree}')
            child = git(repo, 'commit-tree', tree, '-p', world['sha'], '-m', 'new main')
            git(repo, 'push', 'origin', child + ':refs/heads/main')
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(pub, 'git', race)
    report = tmp_path / 'late-main.json'
    with expect_error(pub.PublicationError): publish(world, execute=True, report_path=report)
    saved = json.loads(report.read_text())
    assert saved['pushed'] is True and saved['source_current_after_push'] is False
    assert saved['web_deployment_verified'] is False
    # The two refs are not atomically coupled: never roll back the completed push.
    assert git(world['remote'], 'rev-parse', 'gh-pages^') == world['base']


def test_cli_publish_defaults_to_dry_run(world):
    before = heads(world)
    result = subprocess.run([sys.executable, pub.__file__, 'publish', '--repo', str(world['repo']),
                             '--site', str(world['site']), '--source-commit', world['sha']],
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['mode'] == 'dry-run'
    assert heads(world) == before


def test_source_checkout_mismatch_is_rejected(world):
    with expect_error(pub.PublicationError, match='Checkout'):
        pub.publish(world['repo'], world['site'], world['old'], execute=True)


def test_unrelated_previous_source_is_rejected(world):
    tree = git(world['repo'], 'rev-parse', world['sha'] + '^{tree}')
    unrelated = git(world['repo'], 'commit-tree', tree, '-m', 'unrelated source')
    site = world['site']
    manifest_path = site / 'books_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_commit'] = unrelated
    manifest_path.write_text(json.dumps(manifest))
    pub.write_inventory(site, unrelated)
    _, expected = pub.validate_site(site, unrelated, return_inventory=True)
    malicious = pub.candidate_commit(world['repo'], site, unrelated, world['base'], expected)
    git(world['repo'], 'push', 'origin', malicious + ':refs/heads/gh-pages')
    make_site(site, world['sha'])
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == malicious


@cases('replacement', ['symlink', 'missing-manifest', 'invalid-manifest'])
def test_previous_tree_must_have_regular_files_and_valid_provenance(world, replacement):
    repo = world['repo']
    git(repo, 'checkout', 'gh-pages')
    if replacement == 'symlink':
        (repo / 'previous-link').symlink_to('intro.html')
    elif replacement == 'missing-manifest':
        (repo / 'books_manifest.json').unlink()
    else:
        (repo / 'books_manifest.json').write_text('{broken JSON')
    git(repo, 'add', '-A'); git(repo, 'commit', '-m', 'invalid old deployment')
    invalid = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'push', 'origin', 'gh-pages')
    git(repo, 'checkout', 'main')
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == invalid


def test_remote_move_after_successful_push_is_not_claimed_current(world, monkeypatch):
    original = pub.git
    concurrent = []
    def move_after(repo, *args, **kwargs):
        result = original(repo, *args, **kwargs)
        if args[0] == 'push':
            candidate = args[2].split(':')[0]
            tree = git(repo, 'rev-parse', candidate + '^{tree}')
            child = git(repo, 'commit-tree', tree, '-p', candidate, '-m', 'next publish')
            git(repo, 'push', 'origin', child + ':refs/heads/gh-pages')
            concurrent.append(child)
        return result
    monkeypatch.setattr(pub, 'git', move_after)
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == concurrent[0]


def test_exactly_one_normal_push_without_force_or_branch_creation(world, monkeypatch):
    original, pushes = pub.git, []
    def record(repo, *args, **kwargs):
        if args[0] == 'push': pushes.append(args)
        return original(repo, *args, **kwargs)
    monkeypatch.setattr(pub, 'git', record)
    result = publish(world, execute=True)
    assert pushes == [('push', 'origin', result['candidate_pages_commit'] + ':refs/heads/gh-pages')]


@cases('kind', ['manifest-list', 'manifest-null', 'book-null', 'inventory-list'])
def test_malformed_json_shapes_fail_cleanly(world, kind):
    site = world['site']
    if kind.startswith('manifest-'):
        (site / 'books_manifest.json').write_text('[]' if kind.endswith('list') else 'null')
    elif kind == 'book-null':
        manifest = json.loads((site / 'books_manifest.json').read_text())
        manifest['books'][0] = None
        (site / 'books_manifest.json').write_text(json.dumps(manifest))
    else: (site / pub.INVENTORY).write_text('[]')
    with expect_error(pub.PublicationError): pub.validate_site(site, world['sha'])


def test_untracked_source_is_rejected(world):
    (world['repo'] / 'untracked-source.md').write_text('unexpected build input')
    with expect_error(pub.PublicationError): publish(world, execute=True)
    assert git(world['remote'], 'rev-parse', 'gh-pages') == world['base']


def mock_web(world, monkeypatch, overrides=None):
    calls = []
    overrides = overrides or {}
    def fetch(request, timeout):
        assert timeout == 30
        assert request.get_header('Cache-control') == 'no-cache'
        parsed = urlsplit(request.full_url)
        assert parsed.scheme == 'https' and parsed.netloc == 'example.invalid'
        assert parsed.query == 'source_check=' + world['sha']
        path = parsed.path.removeprefix('/books/')
        calls.append(path)
        if path in overrides:
            value = overrides[path]
            if callable(value): value = value()
            if isinstance(value, Exception): raise value
        else: value = (world['site'] / path).read_bytes()
        return io.BytesIO(value)
    monkeypatch.setattr(pub, 'urlopen', fetch)
    return calls


def test_live_verify_exact_artifact_updates_report(world, monkeypatch, tmp_path):
    calls = mock_web(world, monkeypatch)
    report = tmp_path / 'deployment.json'
    report.write_text(json.dumps({'pushed': True, 'web_deployment_verified': False, 'preserved': 'value'}))
    before = heads(world)
    result = pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books', report_path=report)
    assert result['web_deployment_verified'] is True
    assert result['verified_source_commit'] == world['sha']
    assert calls == ['books_manifest.json', 'intro.html', 'entregable2/intro.html',
                     'entregable3/intro.html', 'entregables.html']
    saved = json.loads(report.read_text())
    assert saved['pushed'] and saved['web_deployment_verified'] and saved['preserved'] == 'value'
    assert heads(world) == before


@cases('failure', ['old-manifest', 'bad-page', 'network', 'invalid-json', 'non-object-json'])
def test_live_verify_failure_never_claims_delivery_or_changes_remote(world, monkeypatch, tmp_path, failure):
    if failure == 'old-manifest':
        manifest = json.loads((world['site'] / 'books_manifest.json').read_text())
        manifest['source_commit'] = world['old']
        overrides = {'books_manifest.json': json.dumps(manifest).encode()}
    elif failure == 'bad-page': overrides = {'entregable3/intro.html': b'stale page'}
    elif failure == 'network': overrides = {'books_manifest.json': OSError('offline')}
    else: overrides = {'books_manifest.json': b'{' if failure == 'invalid-json' else b'[]'}
    calls = mock_web(world, monkeypatch, overrides)
    report = tmp_path / 'deployment.json'
    report.write_text(json.dumps({'pushed': True, 'web_deployment_verified': False}))
    before = heads(world)
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/', report_path=report)
    assert calls.count('books_manifest.json') == 1
    assert not json.loads(report.read_text())['web_deployment_verified']
    assert heads(world) == before


def test_live_verify_bounded_retry_accepts_later_delivery(world, monkeypatch):
    responses = iter([OSError('not ready'), (world['site'] / 'books_manifest.json').read_bytes()])
    calls = mock_web(world, monkeypatch, {'books_manifest.json': lambda: next(responses)})
    sleeps = []
    monkeypatch.setattr(pub.time, 'sleep', lambda interval: sleeps.append(interval))
    result = pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/', attempts=2, interval=0)
    assert result['web_deployment_verified'] and calls.count('books_manifest.json') == 2
    assert sleeps == [0]


def test_failed_reverification_clears_previous_success(world, monkeypatch, tmp_path):
    mock_web(world, monkeypatch, {'books_manifest.json': OSError('offline')})
    report = tmp_path / 'old-success.json'
    report.write_text(json.dumps({'pushed': True, 'web_deployment_verified': True,
        'verified_source_commit': world['old'], 'verified_entrypoints': ['old.html']}))
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/', report_path=report)
    saved = json.loads(report.read_text())
    assert saved['pushed'] is True and saved['web_deployment_verified'] is False
    assert saved['verification_status'] == 'failed' and saved['verification_attempts'] == 1
    assert saved['verification_error'].endswith('offline')
    assert 'verified_source_commit' not in saved and 'verified_entrypoints' not in saved


def test_local_validation_failure_clears_previous_success(world, tmp_path):
    report = tmp_path / 'old-success.json'
    report.write_text(json.dumps({'web_deployment_verified': True}))
    (world['site'] / 'intro.html').write_text('changed')
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/', report_path=report)
    assert json.loads(report.read_text())['verification_status'] == 'failed'
    assert json.loads(report.read_text())['web_deployment_verified'] is False


@cases('state', ['failure', 'cancelled', 'timed_out', 'success', 'pending', 'missing', 'offline', 'truncated'])
def test_pages_diagnostic_never_claims_web_success_or_republishes(world, monkeypatch, tmp_path, state):
    before = heads(world)
    manifest = json.loads((world['site'] / 'books_manifest.json').read_text())
    manifest['source_commit'] = world['old']
    calls = mock_web(world, monkeypatch, {'books_manifest.json': json.dumps(manifest).encode()})
    report = tmp_path / 'publication.json'
    report.write_text(json.dumps({'source_commit': world['sha'], 'pushed': True,
                                 'candidate_pages_commit': world['base']}))
    diagnostics = []
    def diagnostic(repository, pages_sha):
        diagnostics.append((repository, pages_sha))
        if state == 'offline': raise OSError('API unavailable')
        if state == 'truncated': raise IncompleteRead(b'{', 100)
        if state == 'missing': return None
        return {'id': 123, 'head_sha': pages_sha, 'status': 'in_progress' if state == 'pending' else 'completed',
                'conclusion': None if state == 'pending' else state, 'run_attempt': 1,
                'url': 'https://github.com/owner/repo/actions/runs/123'}
    monkeypatch.setattr(pub, 'pages_workflow_status', diagnostic)
    monkeypatch.setattr(pub.time, 'sleep', lambda interval: None)
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/',
                        attempts=3, interval=0, report_path=report, repository='owner/repo')
    saved = json.loads(report.read_text())
    terminal = state in {'failure', 'cancelled', 'timed_out'}
    count = 1 if terminal else 3
    assert diagnostics == [('owner/repo', world['base'])] * count
    assert calls == ['books_manifest.json'] * count
    assert saved['observed_source_commit'] == world['old']
    assert saved['verification_attempts'] == count and saved['verification_status'] == 'failed'
    assert saved['pushed'] is True and saved['web_deployment_verified'] is False
    assert heads(world) == before
    if terminal: assert '/actions/runs/123' in saved['verification_error']
    if state == 'offline': assert saved['pages_diagnostic_error'] == 'API unavailable'
    if state == 'truncated': assert 'IncompleteRead' in saved['pages_diagnostic_error']


def test_truncated_web_response_is_recorded_and_retried(world, monkeypatch, tmp_path):
    responses = iter([IncompleteRead(b'{', 100), (world['site'] / 'books_manifest.json').read_bytes()])
    calls = mock_web(world, monkeypatch, {'books_manifest.json': lambda: next(responses)})
    report = tmp_path / 'publication.json'
    report.write_text(json.dumps({'pushed': True}))
    observations = []
    monkeypatch.setattr(pub.time, 'sleep', lambda interval: observations.append(json.loads(report.read_text())))
    result = pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/',
                             attempts=2, interval=0, report_path=report)
    assert result['web_deployment_verified'] is True and result['verification_attempts'] == 2
    assert calls.count('books_manifest.json') == 2
    assert observations[0]['verification_attempts'] == 1
    assert 'IncompleteRead' in observations[0]['verification_error']
    assert observations[0]['web_deployment_verified'] is False


def test_pages_status_is_read_only_and_selects_only_exact_dynamic_commit(world, monkeypatch):
    base = {'id': 2, 'head_sha': world['base'], 'head_branch': 'gh-pages', 'event': 'dynamic',
        'path': 'dynamic/pages/pages-build-deployment', 'run_attempt': 2,
        'status': 'completed', 'conclusion': 'failure'}
    runs = [dict(base, id=1, conclusion='success'), base]
    for key, value in [('head_sha', world['old']), ('head_branch', 'main'),
                       ('event', 'push'), ('path', '.github/workflows/publish-book.yml')]:
        runs.append(dict(base, id=999, **{key: value}))
    def fetch(request, timeout):
        assert request.get_method() == 'GET' and timeout == 30
        assert request.get_header('Authorization') is None
        assert request.full_url == ('https://api.github.com/repos/owner/repo/actions/runs?head_sha='
                                    + world['base'] + '&per_page=100')
        return io.BytesIO(json.dumps({'workflow_runs': runs}).encode())
    monkeypatch.setattr(pub, 'urlopen', fetch)
    result = pub.pages_workflow_status('owner/repo', world['base'])
    assert result['id'] == 2 and result['conclusion'] == 'failure' and result['run_attempt'] == 2
    assert result['url'] == 'https://github.com/owner/repo/actions/runs/2'


@cases('mutation', ['unconfirmed', 'source', 'missing-sha', 'repository'])
def test_pages_diagnostics_rejects_mismatched_publication_context(world, monkeypatch, tmp_path, mutation):
    calls = mock_web(world, monkeypatch)
    report = tmp_path / 'publication.json'
    payload = {'source_commit': world['sha'], 'pushed': True, 'candidate_pages_commit': world['base']}
    repository = 'owner/repo'
    if mutation == 'unconfirmed': payload['pushed'] = None
    elif mutation == 'source': payload['source_commit'] = world['old']
    elif mutation == 'missing-sha': payload.pop('candidate_pages_commit')
    else: repository = 'owner/repo?injected=1'
    report.write_text(json.dumps(payload))
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/',
                        report_path=report, repository=repository)
    assert calls == []
    assert json.loads(report.read_text())['verification_status'] == 'failed'


def test_live_verify_retry_exhaustion_is_bounded(world, monkeypatch):
    calls = mock_web(world, monkeypatch, {'books_manifest.json': OSError('offline')})
    sleeps = []
    monkeypatch.setattr(pub.time, 'sleep', lambda interval: sleeps.append(interval))
    with expect_error(pub.PublicationError):
        pub.verify_live(world['site'], world['sha'], 'https://example.invalid/books/', attempts=3, interval=0)
    assert calls == ['books_manifest.json'] * 3 and sleeps == [0, 0]


@cases('url', ['file:///tmp/local', 'ftp://example.invalid', 'not a URL'])
def test_live_verify_rejects_non_http_urls(world, monkeypatch, url):
    calls = mock_web(world, monkeypatch)
    with expect_error(pub.PublicationError): pub.verify_live(world['site'], world['sha'], url)
    assert not calls


def test_workflow_only_publishes_new_main_pushes_with_separate_permissions():
    import re
    import yaml
    path = Path(__file__).resolve().parents[1] / '.github/workflows/publish-book.yml'
    text = path.read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert set(workflow['on']) == {'push', 'pull_request', 'workflow_dispatch'}
    assert 'paths' not in workflow['on']['push']
    assert workflow['permissions'] == {'contents': 'read'}
    deploy = workflow['jobs']['publish']
    assert deploy['permissions'] == {'contents': 'write'}
    assert deploy['if'].strip() == "github.event_name == 'push' && github.ref == 'refs/heads/main' && github.run_attempt == 1"
    assert deploy['concurrency'] == {'group': 'publish-book', 'cancel-in-progress': 'false'}
    for job in workflow['jobs'].values():
        for step in job['steps']:
            if 'uses' in step:
                assert re.fullmatch(r'[^@]+@[0-9a-f]{40}', step['uses'])
            if step.get('uses', '').startswith('actions/checkout@'):
                assert step['with']['persist-credentials'] == 'false'
    assert 'push --force' not in text and 'git init' not in text
    assert any('verify-live' in step.get('run', '') for step in deploy['steps'])


class SafePublishBooksTests(unittest.TestCase):
    """Each expanded case gets a fresh source, site and local bare remote."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="safe-publish-test-")
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.monkeypatch = Patches(self)
        self.world = world(self.tmp_path, self.monkeypatch)


def _install_cases():
    for name, function in list(globals().items()):
        if not name.startswith('test_') or not callable(function):
            continue
        parameter, values = getattr(function, 'cases', (None, [None]))
        for number, value in enumerate(values):
            def run(self, function=function, parameter=parameter, value=value):
                arguments = {key: (value if key == parameter else getattr(self, key))
                             for key in inspect.signature(function).parameters}
                function(**arguments)
            suffix = ('_' + str(number)) if parameter else ''
            setattr(SafePublishBooksTests, name + suffix, run)


_install_cases()

if __name__ == '__main__':
    unittest.main()

