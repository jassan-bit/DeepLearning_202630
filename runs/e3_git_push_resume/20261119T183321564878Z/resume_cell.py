import sys as _e3_sys
import types as _e3_types
_e3_module = _e3_types.ModuleType('incremental_git_trees')
_e3_sys.modules[_e3_module.__name__] = _e3_module
exec(compile('"""Create the exact audit tree through small immutable Git tree objects.\n\nOperational IO only, with disclosed AI assistance. All requests use the caller\'s\noriginal paced transport. This module never creates blobs, commits or refs.\n"""\nimport hashlib\nfrom pathlib import PurePosixPath\nimport re\n\nENDPOINT = \'/repos/jassan-bit/DeepLearning_202630/git/trees\'\nSHA = re.compile(r\'[0-9a-f]{40}\')\nSTAMP = re.compile(r\'[0-9]{8}T[0-9]{12}Z\')\nMODES = {\'100644\': \'blob\', \'100755\': \'blob\', \'120000\': \'blob\',\n         \'040000\': \'tree\', \'160000\': \'commit\'}\n\n\ndef require(condition, message):\n    if not condition:\n        raise ValueError(message)\n\n\ndef component(name):\n    require(isinstance(name, str) and name not in (\'\', \'.\', \'..\')\n            and \'/\' not in name and \'\\0\' not in name, \'Invalid Git tree basename\')\n    try:\n        return name.encode(\'utf-8\')\n    except UnicodeError:\n        raise ValueError(\'Git tree basename is not UTF-8\') from None\n\n\ndef normalized(row):\n    require(isinstance(row, dict), \'Git tree row must be an object\')\n    name, mode, kind, sha = (row.get(k) for k in (\'path\', \'mode\', \'type\', \'sha\'))\n    component(name)\n    require(isinstance(mode, str) and isinstance(kind, str) and mode in MODES and kind == MODES[mode]\n            and isinstance(sha, str) and SHA.fullmatch(sha), \'Invalid Git tree entry\')\n    return {\'path\': name, \'mode\': mode, \'type\': kind, \'sha\': sha}\n\n\ndef tree_sha(rows):\n    clean = [normalized(row) for row in rows]\n    require(len({r[\'path\'] for r in clean}) == len(clean), \'Duplicate Git tree basenames\')\n    clean.sort(key=lambda r: component(r[\'path\']) + (b\'/\' if r[\'type\'] == \'tree\' else b\'\'))\n    raw = b\'\'.join((r[\'mode\'].lstrip(\'0\') or \'0\').encode() + b\' \' + component(r[\'path\'])\n                   + b\'\\0\' + bytes.fromhex(r[\'sha\']) for r in clean)\n    return hashlib.sha1(b\'tree \' + str(len(raw)).encode() + b\'\\0\' + raw).hexdigest()\n\n\ndef validate_payload(payload):\n    require(isinstance(payload, dict) and set(payload) == {\'base_tree\', \'tree\'}\n            and isinstance(payload[\'base_tree\'], str) and SHA.fullmatch(payload[\'base_tree\']),\n            \'Only an exact base_tree and tree audit request can be split\')\n    rows = payload[\'tree\']\n    require(isinstance(rows, list) and 0 < len(rows) <= 4096, \'Invalid flat audit tree size\')\n    common = None\n    root = {}\n    seen = set()\n    for row in rows:\n        require(isinstance(row, dict) and set(row) == {\'path\', \'mode\', \'type\', \'sha\'}\n                and row[\'mode\'] == \'100644\' and row[\'type\'] == \'blob\'\n                and isinstance(row[\'sha\'], str) and SHA.fullmatch(row[\'sha\']),\n                \'Only recorded regular audit blobs can be split\')\n        path = row[\'path\']\n        require(isinstance(path, str) and 0 < len(path) <= 1024\n                and \'\\\\\' not in path and not any(ord(c) < 32 or ord(c) == 127 for c in path),\n                \'Unsafe flat audit path\')\n        p = PurePosixPath(path)\n        require(str(p) == path and not p.is_absolute() and \'..\' not in p.parts\n                and 4 <= len(p.parts) <= 64 and p.parts[:2] == (\'runs\', \'e3_extended_audit\')\n                and STAMP.fullmatch(p.parts[2]), \'Flat tree must add one audit part\')\n        require(path not in seen, \'Duplicate flat audit path\')\n        seen.add(path)\n        prefix = p.parts[:3]\n        if common is None:\n            common = prefix\n        require(prefix == common, \'Flat tree contains multiple audit parts\')\n        node = root\n        for name in p.parts[3:-1]:\n            component(name)\n            require(name not in node or isinstance(node[name], dict), \'Audit file/directory conflict\')\n            node = node.setdefault(name, {})\n        name = p.parts[-1]\n        component(name)\n        require(name not in node, \'Audit file/directory conflict\')\n        node[name] = row[\'sha\']\n    return common[2], root\n\n\ndef build_incremental_tree(request, token, payload, emit=None):\n    """Return the exact flat-request root SHA using bottom-up direct children.\n\n    Validate the whole flat request and all base ancestors before mutations.\n    Existing tree objects are verified/reused, so interrupted runs are resumable.\n    """\n    part, forest = validate_payload(payload)\n    emit = emit or (lambda report: None)\n    cache = {}\n    counts = {\'created\': 0, \'reused\': 0}\n\n    def checked_tree(value, expected):\n        require(isinstance(value, dict) and value.get(\'sha\') == expected\n                and value.get(\'truncated\') is False and isinstance(value.get(\'tree\'), list),\n                \'A complete immutable Git tree is required\')\n        rows = [normalized(row) for row in value[\'tree\']]\n        require(tree_sha(rows) == expected, \'Git tree content differs from its SHA\')\n        return {r[\'path\']: r for r in rows}\n\n    def read(sha):\n        if sha not in cache:\n            cache[sha] = checked_tree(request(\'GET\', ENDPOINT + \'/\' + sha, token), sha)\n        return dict(cache[sha])\n\n    def child(rows, name):\n        found = rows.get(name)\n        if found is None:\n            return {}\n        require(found[\'mode\'] == \'040000\' and found[\'type\'] == \'tree\',\n                \'Audit ancestor is not a directory\')\n        return read(found[\'sha\'])\n\n    base_rows = read(payload[\'base_tree\'])\n    runs_rows = child(base_rows, \'runs\')\n    audit_rows = child(runs_rows, \'e3_extended_audit\')\n    require(part not in audit_rows, \'Audit part already exists in the parent tree\')\n\n    def ensure(rows):\n        values = list(rows.values())\n        expected = tree_sha(values)\n        if expected in cache:\n            require(cache[expected] == rows, \'Existing tree cache content differs\')\n            counts[\'reused\'] += 1\n        else:\n            existing = request(\'GET\', ENDPOINT + \'/\' + expected, token, allow_missing=True)\n            if existing is not None:\n                require(checked_tree(existing, expected) == rows, \'Existing immutable tree differs\')\n                counts[\'reused\'] += 1\n            else:\n                values.sort(key=lambda r: component(r[\'path\']) + (b\'/\' if r[\'type\'] == \'tree\' else b\'\'))\n                result = request(\'POST\', ENDPOINT, token, {\'tree\': values})\n                require(isinstance(result, dict) and result.get(\'sha\') == expected,\n                        \'Created Git tree SHA differs from the intended tree\')\n                if result.get(\'truncated\') is not False or not isinstance(result.get(\'tree\'), list):\n                    result = request(\'GET\', ENDPOINT + \'/\' + expected, token)\n                require(checked_tree(result, expected) == rows, \'Created Git tree content differs\')\n                counts[\'created\'] += 1\n            cache[expected] = dict(rows)\n        emit({\'event\': \'incremental_git_tree\', **counts, \'entries\': len(rows), \'sha\': expected})\n        return expected\n\n    def build(node):\n        rows = {}\n        for name, value in node.items():\n            if isinstance(value, dict):\n                rows[name] = {\'path\': name, \'mode\': \'040000\', \'type\': \'tree\', \'sha\': build(value)}\n            else:\n                rows[name] = {\'path\': name, \'mode\': \'100644\', \'type\': \'blob\', \'sha\': value}\n        return ensure(rows)\n\n    part_sha = build(forest)\n    audit_rows[part] = {\'path\': part, \'mode\': \'040000\', \'type\': \'tree\', \'sha\': part_sha}\n    audit_sha = ensure(audit_rows)\n    runs_rows[\'e3_extended_audit\'] = {\'path\': \'e3_extended_audit\', \'mode\': \'040000\',\n                                    \'type\': \'tree\', \'sha\': audit_sha}\n    runs_sha = ensure(runs_rows)\n    base_rows[\'runs\'] = {\'path\': \'runs\', \'mode\': \'040000\', \'type\': \'tree\', \'sha\': runs_sha}\n    return {\'sha\': ensure(base_rows)}\n', 'incremental_git_trees.py', "exec"), _e3_module.__dict__)
_e3_module = _e3_types.ModuleType('git_stage_transport')
_e3_sys.modules[_e3_module.__name__] = _e3_module
exec(compile('"""Git PACK transport for immutable audit objects; isolated from scientific files.\n\nOnly an absent staging/audit branch may be created. The empty expected-value\nlease is deliberately stricter than an ordinary push: even a concurrent\nfast-forwardable branch creation is rejected, never overwritten.\n"""\nfrom datetime import datetime, timezone\nimport hashlib\nimport json\nimport os\nfrom pathlib import Path\nimport re\nimport secrets\nimport subprocess\nimport tempfile\nfrom urllib.parse import urlsplit\n\nimport incremental_git_trees as trees\n\nSHA = re.compile(r\'[0-9a-f]{40}\')\nPART = re.compile(r\'[0-9]{8}T[0-9]{12}Z\')\nREF = re.compile(r\'refs/heads/(e3-git-stage|e3-extended-audit)-([0-9]{8}T[0-9]{12}Z)\')\nCREDENTIAL_ENV = re.compile(r\'TOKEN|(?:^|_)PAT(?:_|$)|SECRET|CREDENTIAL\', re.IGNORECASE)\n\n\nclass GitStageStore:\n    def __init__(self, base, source_master, repo_url, token, identity=None):\n        trees.require(isinstance(source_master, str) and SHA.fullmatch(source_master), \'Invalid parent commit\')\n        url = urlsplit(repo_url)\n        trees.require(url.scheme in (\'https\', \'file\') and not url.username and not url.password\n                      and not url.query and not url.fragment and\n                      ((url.scheme == \'https\' and url.hostname) or\n                       (url.scheme == \'file\' and url.netloc in (\'\', \'localhost\') and url.path.startswith(\'/\'))),\n                      \'A credential-free HTTPS or local file Git URL is required\')\n        self.base = Path(base)\n        trees.require(not self.base.is_symlink(), \'Git transport base must not be a symlink\')\n        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)\n        self.cache_path = self.base / \'git-stage-cache.git\'\n        trees.require(not self.cache_path.is_symlink(), \'Git cache must not be a symlink\')\n        self.parent_sha, self.repo_url, self._token = source_master, repo_url, token\n        self.identity = identity or {\'name\': \'E3 audit staging\', \'email\': \'e3-audit-staging@users.noreply.github.com\'}\n        trees.require(set(self.identity) == {\'name\', \'email\'} and all(\n            isinstance(v, str) and v and not any(c in v for c in \'\\0\\n\\r<>\') for v in self.identity.values()),\n            \'Invalid Git author identity\')\n        self._trees, self._roots, self._commits = {}, {}, {}\n        self._blob_shas, self._commit_trees, self._scanned_blob_roots = set(), set(), set()\n        if not self.cache_path.exists():\n            self._git(\'init\', \'--bare\', \'--quiet\', str(self.cache_path), outside=True)\n            self._git(\'config\', \'remote.origin.url\', repo_url)\n            self._git(\'config\', \'remote.origin.promisor\', \'true\')\n            self._git(\'config\', \'remote.origin.partialclonefilter\', \'blob:none\')\n        trees.require(self._git(\'config\', \'--get\', \'remote.origin.url\').decode().strip() == repo_url,\n                      \'Git cache remote differs from the requested repository\')\n        if not self._has(source_master, \'commit\'):\n            self._fetch(source_master)\n        self.parent_tree = self.read_commit(source_master)[\'tree\'][\'sha\']\n\n    def _env(self):\n        env = {k: v for k, v in os.environ.items()\n               if not k.startswith(\'GIT_\') and not CREDENTIAL_ENV.search(k) and k not in (\'E3_GIT_PUSH_TOKEN\', \'SSH_ASKPASS\',\n                   \'GH_TOKEN\', \'GITHUB_TOKEN\', \'GH_ENTERPRISE_TOKEN\', \'GITHUB_ENTERPRISE_TOKEN\', \'GITHUB_PAT\')\n               and (not self._token or v != self._token)}\n        env.update(GIT_CONFIG_NOSYSTEM=\'1\', GIT_CONFIG_GLOBAL=os.devnull,\n                   GIT_TERMINAL_PROMPT=\'0\', GIT_NO_LAZY_FETCH=\'1\', LC_ALL=\'C\')\n        return env\n\n    def _git(self, *args, data=None, network=False, outside=False, extra_env=None, check=True):\n        trees.require(not self.base.is_symlink() and not self.cache_path.is_symlink(),\n                      \'Git transport base/cache must not be symlinks\')\n        env = self._env()\n        env.update(extra_env or {})\n        cmd = [\'git\'] + ([] if outside else [\'--git-dir=\' + str(self.cache_path)])\n        # Never persist credentials or consult a user/global credential helper.\n        cmd += [\'-c\', \'credential.helper=\']\n        if not network or \'push\' in args:\n            # Disable implicit object fetches even on Git versions predating GIT_NO_LAZY_FETCH.\n            cmd += [\'-c\', \'remote.origin.promisor=false\']\n        def run():\n            result = subprocess.run(cmd + list(args), input=data, stdout=subprocess.PIPE,\n                                    stderr=subprocess.PIPE, env=env, check=False)\n            if check and result.returncode:\n                detail = result.stderr.decode(\'utf-8\', \'replace\')[-4000:]\n                if self._token:\n                    detail = detail.replace(self._token, \'[REDACTED]\')\n                raise RuntimeError(\'Git transport command failed: \' + detail.strip())\n            return result.stdout if check else result\n        if not network:\n            return run()\n        with tempfile.TemporaryDirectory(prefix=\'git-askpass-\', dir=self.base) as folder:\n            script = Path(folder) / \'askpass\'\n            script.write_text(\'#!/bin/sh\\ncase "$1" in\\n*Username*) printf "%s\\\\n" x-access-token ;;\\n*) printf "%s\\\\n" "$E3_GIT_PUSH_TOKEN" ;;\\nesac\\n\')\n            script.chmod(0o700)\n            env.update(GIT_ASKPASS=str(script), E3_GIT_PUSH_TOKEN=self._token or \'\')\n            return run()\n\n    def _has(self, sha, kind):\n        result = self._git(\'cat-file\', \'-t\', sha, check=False)\n        return result.returncode == 0 and result.stdout.decode().strip() == kind\n\n    def _fetch(self, sha):\n        self._git(\'fetch\', \'--quiet\', \'--no-tags\', \'--depth=1\', \'--filter=blob:none\', \'origin\', sha, network=True)\n        trees.require(self._has(sha, \'commit\'), \'Fetched parent is not the exact required commit\')\n\n    def blob(self, raw):\n        trees.require(isinstance(raw, bytes), \'Git blob input must be exact bytes\')\n        expected = hashlib.sha1(b\'blob \' + str(len(raw)).encode() + b\'\\0\' + raw).hexdigest()\n        actual = self._git(\'hash-object\', \'-w\', \'--stdin\', data=raw).decode().strip()\n        trees.require(actual == expected, \'Stored blob SHA differs from the frozen content\')\n        return actual\n\n    def read_blob(self, sha, max_bytes=48 * 1024 * 1024):\n        """Read bounded receipt metadata, fetching only this exact object if absent."""\n        trees.require(isinstance(sha, str) and SHA.fullmatch(sha) and\n                      isinstance(max_bytes, int) and not isinstance(max_bytes, bool) and\n                      0 <= max_bytes <= 48 * 1024 * 1024, \'Invalid bounded blob request\')\n        kind = self._git(\'cat-file\', \'-t\', sha, check=False)\n        if kind.returncode:\n            for root in self._commit_trees - self._scanned_blob_roots:\n                self.read_tree(root)\n                listing = self._git(\'ls-tree\', \'-r\', \'-z\', root)\n                for item in listing.split(b\'\\0\'):\n                    if item:\n                        header = item.split(b\'\\t\', 1)[0].decode(\'ascii\').split(\' \')\n                        if header[1] == \'blob\':\n                            self._blob_shas.add(header[2])\n                self._scanned_blob_roots.add(root)\n                if sha in self._blob_shas:\n                    break\n            trees.require(sha in self._blob_shas, \'Missing blob must belong to a verified local commit tree\')\n            # An exact blob has no children: this retrieves no commit history or sibling weights.\n            self._git(\'fetch\', \'--quiet\', \'--no-tags\', \'--no-filter\', \'--no-write-fetch-head\',\n                      \'origin\', sha, network=True)\n        trees.require(self._has(sha, \'blob\'), \'Requested metadata object is not a Git blob\')\n        size = int(self._git(\'cat-file\', \'-s\', sha).decode().strip())\n        trees.require(0 <= size <= max_bytes, \'Metadata blob exceeds the permitted byte limit\')\n        raw = self._git(\'cat-file\', \'blob\', sha)\n        expected = hashlib.sha1(b\'blob \' + str(len(raw)).encode() + b\'\\0\' + raw).hexdigest()\n        trees.require(len(raw) == size and expected == sha, \'Metadata blob size or Git SHA differs\')\n        return raw\n\n    def read_tree(self, sha):\n        trees.require(isinstance(sha, str) and SHA.fullmatch(sha), \'Invalid tree SHA\')\n        raw = self._git(\'ls-tree\', \'-z\', sha)\n        rows = []\n        for value in raw.split(b\'\\0\'):\n            if value:\n                header, path = value.split(b\'\\t\', 1)\n                mode, kind, obj = header.decode(\'ascii\').split(\' \')\n                rows.append({\'path\': path.decode(\'utf-8\'), \'mode\': mode, \'type\': kind, \'sha\': obj})\n        trees.require(trees.tree_sha(rows) == sha, \'Local Git tree differs from its SHA\')\n        self._blob_shas.update(r[\'sha\'] for r in rows if r[\'type\'] == \'blob\')\n        return {\'sha\': sha, \'truncated\': False, \'tree\': rows}\n\n    def _request(self, method, path, token=None, payload=None, *, allow_missing=False):\n        if method == \'GET\' and path.startswith(trees.ENDPOINT + \'/\'):\n            sha = path.rsplit(\'/\', 1)[1]\n            # Optional reuse probes consult only trees created in this process.\n            return self._trees.get(sha) if allow_missing else self.read_tree(sha)\n        trees.require(method == \'POST\' and path == trees.ENDPOINT and set(payload) == {\'tree\'},\n                      \'Unexpected local Git tree request\')\n        rows = [trees.normalized(r) for r in payload[\'tree\']]\n        raw = b\'\'.join((r[\'mode\']+\' \'+r[\'type\']+\' \'+r[\'sha\']+\'\\t\'+r[\'path\']).encode()+b\'\\0\' for r in rows)\n        sha = self._git(\'mktree\', \'--missing\', \'-z\', data=raw).decode().strip()\n        trees.require(sha == trees.tree_sha(rows), \'Git mktree differs from the intended tree\')\n        self._trees[sha] = self.read_tree(sha)\n        return self._trees[sha]\n\n    def stage_tree(self, payload):\n        part, _ = trees.validate_payload(payload)\n        trees.require(payload[\'base_tree\'] == self.parent_tree, \'Audit base differs from the frozen parent\')\n        trees.require(all(self._has(r[\'sha\'], \'blob\') for r in payload[\'tree\']),\n                      \'Every frozen audit blob must be stored locally before staging\')\n        result = trees.build_incremental_tree(self._request, None, payload)\n        self._roots[result[\'sha\']] = part\n        return result\n\n    def read_commit(self, sha):\n        trees.require(isinstance(sha, str) and SHA.fullmatch(sha), \'Invalid commit SHA\')\n        raw = self._git(\'cat-file\', \'commit\', sha)\n        trees.require(hashlib.sha1(b\'commit \' + str(len(raw)).encode() + b\'\\0\' + raw).hexdigest() == sha,\n                      \'Local commit content differs from its Git SHA\')\n        raw = raw.split(b\'\\n\\n\', 1)[0]\n        tree = [v[5:].decode() for v in raw.splitlines() if v.startswith(b\'tree \')]\n        parents = [v[7:].decode() for v in raw.splitlines() if v.startswith(b\'parent \')]\n        trees.require(len(tree) == 1 and SHA.fullmatch(tree[0]) and all(SHA.fullmatch(p) for p in parents),\n                      \'Malformed local commit\')\n        self._commit_trees.add(tree[0])\n        return {\'sha\': sha, \'tree\': {\'sha\': tree[0]}, \'parents\': [{\'sha\': p} for p in parents]}\n\n    def create_commit(self, tree_sha, parent_sha, message, part, role=\'audit\'):\n        trees.require(self._roots.get(tree_sha) == part and parent_sha == self.parent_sha and PART.fullmatch(part)\n                      and role in (\'staging\', \'audit\') and isinstance(message, str) and message and \'\\0\' not in message,\n                      \'Commit must bind the registered audit tree and original parent\')\n        journal_path = self.base / (\'git-\' + role + \'-\' + part + \'.json\')\n        trees.require(not journal_path.is_symlink(), \'Immutable commit journal must not be a symlink\')\n        metadata = {\'schema\': 1, \'part\': part, \'role\': role, \'tree\': tree_sha, \'parents\': [parent_sha],\n                    \'message\': message, \'identity\': dict(self.identity)}\n        if journal_path.exists():\n            recorded = json.loads(journal_path.read_text())\n            trees.require(all(recorded.get(k) == v for k, v in metadata.items())\n                          and set(recorded) == set(metadata) | {\'nonce\', \'date\'}\n                          and re.fullmatch(\'[0-9a-f]{32}\', recorded.get(\'nonce\', \'\'))\n                          and re.fullmatch(r\'\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\dZ\', recorded.get(\'date\', \'\')),\n                          \'Existing immutable commit journal differs\')\n        else:\n            recorded = dict(metadata, nonce=secrets.token_hex(16),\n                            date=datetime.now(timezone.utc).strftime(\'%Y-%m-%dT%H:%M:%SZ\'))\n            # Write once before commit creation; retries retain the current-time identity/date/message.\n            fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)\n            with os.fdopen(fd, \'w\') as stream:\n                json.dump(recorded, stream, sort_keys=True, indent=2)\n                stream.write(\'\\n\')\n                stream.flush()\n                os.fsync(stream.fileno())\n        author = {\'GIT_AUTHOR_NAME\': recorded[\'identity\'][\'name\'], \'GIT_COMMITTER_NAME\': recorded[\'identity\'][\'name\'],\n                  \'GIT_AUTHOR_EMAIL\': recorded[\'identity\'][\'email\'], \'GIT_COMMITTER_EMAIL\': recorded[\'identity\'][\'email\'],\n                  \'GIT_AUTHOR_DATE\': recorded[\'date\'], \'GIT_COMMITTER_DATE\': recorded[\'date\']}\n        sha = self._git(\'commit-tree\', tree_sha, \'-p\', parent_sha, data=message.encode(), extra_env=author).decode().strip()\n        trees.require(self.read_commit(sha) == {\'sha\': sha, \'tree\': {\'sha\': tree_sha},\n                                              \'parents\': [{\'sha\': parent_sha}]}, \'Local commit content differs\')\n        self._commits[sha] = {\'part\': part, \'role\': role, \'tree\': tree_sha}\n        return {\'sha\': sha}\n\n    def _remote_ref(self, ref):\n        raw = self._git(\'ls-remote\', \'--refs\', \'origin\', ref, network=True).decode().splitlines()\n        trees.require(len(raw) <= 1, \'Remote returned multiple staging refs\')\n        if not raw:\n            return None\n        fields = raw[0].split(\'\\t\')\n        trees.require(len(fields) == 2 and fields[1] == ref and SHA.fullmatch(fields[0]), \'Malformed remote ref\')\n        return fields[0]\n\n    def publish_ref(self, ref, commit_sha):\n        match = REF.fullmatch(ref) if isinstance(ref, str) else None\n        record = self._commits.get(commit_sha)\n        trees.require(match and record and record[\'part\'] == match[2]\n                      and record[\'role\'] == (\'staging\' if match[1] == \'e3-git-stage\' else \'audit\'),\n                      \'Only the registered immutable audit/staging commit may be published\')\n        found = self._remote_ref(ref)\n        if found is not None:\n            trees.require(found == commit_sha, \'Conflicting remote audit/staging branch; never overwrite\')\n        else:\n            try:\n                self._git(\'-c\', \'push.negotiate=true\', \'-c\', \'pack.window=0\', \'-c\', \'pack.depth=0\', \'-c\', \'core.compression=1\',\n                          \'-c\', \'pack.threads=1\', \'push\', \'--atomic\', \'--porcelain\', \'--no-thin\',\n                          \'--force-with-lease=\' + ref + \':\', \'origin\', commit_sha + \':\' + ref, network=True)\n            except RuntimeError:\n                if self._remote_ref(ref) != commit_sha:\n                    raise\n            trees.require(self._remote_ref(ref) == commit_sha, \'Remote ref verification failed\')\n        return {\'ref\': ref, \'object\': {\'type\': \'commit\', \'sha\': commit_sha}}\n\n    def publish(self, tree_sha, part):\n        trees.require(self._roots.get(tree_sha) == part and PART.fullmatch(part), \'Unregistered staging tree\')\n        ref = \'refs/heads/e3-git-stage-\' + part\n        existing = self._remote_ref(ref)\n        if existing:\n            if not self._has(existing, \'commit\'):\n                self._fetch(existing)\n            value = self.read_commit(existing)\n            trees.require(value[\'tree\'][\'sha\'] == tree_sha and value[\'parents\'] == [{\'sha\': self.parent_sha}],\n                          \'Conflicting remote staging branch; never overwrite\')\n            return {\'sha\': existing, \'tree\': tree_sha, \'ref\': ref, \'reused\': True}\n        commit = self.create_commit(tree_sha, self.parent_sha, \'E3 Git object staging for audit part \' + part,\n                                    part, role=\'staging\')[\'sha\']\n        self.publish_ref(ref, commit)\n        return {\'sha\': commit, \'tree\': tree_sha, \'ref\': ref, \'reused\': False}\n', 'git_stage_transport.py', "exec"), _e3_module.__dict__)
_e3_module = _e3_types.ModuleType('git_api_provider')
_e3_sys.modules[_e3_module.__name__] = _e3_module
exec(compile('"""Git-backed metadata/writes for the unchanged audit publisher.\n\nObjects are read from the isolated Git cache only after authenticated Git\npublication or fetch. All content identity/manifest checks remain active.\n"""\nimport base64\nimport hashlib\nimport json\nfrom pathlib import Path\nimport re\nfrom urllib.parse import parse_qs, unquote, urlsplit\nfrom git_stage_transport import GitStageStore\nfrom incremental_git_trees import validate_payload, tree_sha\n\nSOURCE = \'5a07d6157a9796ca54416dc5dea1e1527e731597\'\nMASTER = \'20261119T031104515520Z\'\nPROTOCOL = \'5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a\'\nREPO = \'jassan-bit/DeepLearning_202630\'\nROOT = \'/repos/\' + REPO\nSHA = re.compile(r\'[0-9a-f]{40}\')\nPART = re.compile(r\'[0-9]{8}T[0-9]{12}Z\')\n\n\ndef require(condition, message):\n    if not condition:\n        raise ValueError(message)\n\n\nclass GitPackContext:\n    def __init__(self, base, token, identity=None, repo_url=None, emit=None):\n        self.base, self.token, self.identity = Path(base), token, identity\n        self.repo_url = repo_url or \'https://github.com/\' + REPO + \'.git\'\n        self.emit = emit or (lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))\n        self.stores, self.ready_trees, self.ready_commits, self.commit_owners = {}, {}, {}, {}\n        self.ready_blobs = {}\n        self.advertised, self.tree_owners = set(), {}\n        self.anchor = \'c5141ade210734bd4eaac8279a32d8d04c41269d\'\n\n    def head(self):\n        path = self.base / (\'e3-extended-continuation-\' + MASTER + \'.json\')\n        if not path.exists():\n            return self.anchor\n        require(not any(p.is_symlink() for p in (path, *path.parents)), \'Symlink state refused\')\n        state = json.loads(path.read_bytes())\n        require(state.get(\'kind\') == \'e3_audit_continuation_state\'\n                and state.get(\'repository\') == REPO and state.get(\'source_commit\') == SOURCE\n                and state.get(\'master_run_id\') == MASTER and state.get(\'protocol_sha256\') == PROTOCOL\n                and isinstance(state.get(\'head_commit\'), str) and SHA.fullmatch(state[\'head_commit\']),\n                \'Git transport cannot select another campaign or parent\')\n        return state[\'head_commit\']\n\n    def store(self, parent=None):\n        parent = parent or self.head()\n        if parent not in self.stores:\n            require(self.identity is not None, \'Git author identity was not authenticated\')\n            self.stores[parent] = GitStageStore(self.base / (\'e3-git-publisher-\' + MASTER),\n                parent, self.repo_url, self.token, identity=self.identity)\n        return self.stores[parent]\n\n    def blob(self, raw):\n        require(self.token.encode() not in raw, \'Credential bytes cannot enter a Git publication\')\n        return self.store().blob(raw)\n\n    def mark(self, store, commit):\n        self.ready_commits[commit] = store\n        self.ready_trees[store.read_commit(commit)[\'tree\'][\'sha\']] = store\n        # Tree walking reads metadata only, not historical checkpoint bytes.\n        raw = store._git(\'ls-tree\', \'-r\', \'-t\', \'-z\', store.read_commit(commit)[\'tree\'][\'sha\'])\n        for item in raw.split(b\'\\0\'):\n            if not item:\n                continue\n            header = item.split(b\'\\t\', 1)[0].decode(\'ascii\').split(\' \')\n            if header[1] == \'tree\':\n                self.ready_trees[header[2]] = store\n            elif header[1] == \'blob\':\n                self.ready_blobs[header[2]] = store\n\n    def tree(self, payload):\n        require(isinstance(payload, dict) and set(payload) == {\'base_tree\', \'tree\'}, \'Unexpected audit tree request\')\n        part, _ = validate_payload(payload)\n        store = self.store()\n        result = store.stage_tree(payload)\n        self.emit({\'event\': \'git_pack_publication_started\', \'part\': part, \'tree\': result[\'sha\']})\n        staged = store.publish(result[\'sha\'], part)\n        self.mark(store, staged[\'sha\'])\n        self.emit({\'event\': \'git_pack_staging_verified\', \'part\': part,\n                   \'commit\': staged[\'sha\'], \'ref\': staged[\'ref\']})\n        return result\n\n    def commit(self, payload):\n        require(isinstance(payload, dict) and set(payload) == {\'message\', \'tree\', \'parents\'}\n                and isinstance(payload[\'parents\'], list) and len(payload[\'parents\']) == 1,\n                \'Unexpected audit commit request\')\n        message = payload[\'message\']\n        part = message.removeprefix(\'Guardar delta auditoría E3 \') if isinstance(message, str) else \'\'\n        require(isinstance(message, str) and message == \'Guardar delta auditoría E3 \' + part\n                and PART.fullmatch(part), \'Only the original audit commit message is accepted\')\n        store = self.store(payload[\'parents\'][0])\n        result = store.create_commit(payload[\'tree\'], payload[\'parents\'][0], message, part)\n        self.commit_owners[result[\'sha\']] = store\n        return result\n\n    def reference(self, payload):\n        require(isinstance(payload, dict) and set(payload) == {\'ref\', \'sha\'}\n                and isinstance(payload[\'ref\'], str)\n                and re.fullmatch(r\'refs/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z\', payload[\'ref\'])\n                and payload[\'sha\'] in self.commit_owners, \'Only the registered audit reference may be published\')\n        store = self.commit_owners[payload[\'sha\']]\n        result = store.publish_ref(payload[\'ref\'], payload[\'sha\'])\n        self.mark(store, payload[\'sha\'])\n        self.emit({\'event\': \'git_pack_audit_ref_verified\', \'ref\': payload[\'ref\'], \'commit\': payload[\'sha\']})\n        return result\n\n    def refs(self, prefix):\n        store = self.store()\n        raw = store._git(\'ls-remote\', \'--refs\', \'origin\', prefix + \'*\', network=True).decode()\n        rows = []\n        for line in raw.splitlines():\n            sha, ref = line.split(\'\\t\')\n            require(SHA.fullmatch(sha) and ref.startswith(prefix), \'Unexpected remote Git reference\')\n            rows.append({\'ref\': ref, \'object\': {\'type\': \'commit\', \'sha\': sha}})\n            self.advertised.add(sha)\n        return sorted(rows, key=lambda row: row[\'ref\'])\n\n    def ref(self, ref):\n        store = self.store()\n        sha = store._remote_ref(ref)\n        if sha is not None:\n            self.advertised.add(sha)\n            self.fetch_commit(sha)\n        return None if sha is None else {\'ref\': ref, \'object\': {\'type\': \'commit\', \'sha\': sha}}\n\n    def fetch_commit(self, sha):\n        if sha not in self.ready_commits:\n            store = self.store()\n            if not store._has(sha, \'commit\'):\n                store._fetch(sha)\n            self.mark(store, sha)\n        return self.ready_commits[sha]\n\n    def read_tree(self, store, sha, recursive):\n        def walk(tree, prefix=\'\'):\n            result = []\n            for item in store.read_tree(tree)[\'tree\']:\n                row = dict(item, path=prefix + item[\'path\'])\n                if recursive and row[\'type\'] == \'blob\' and store._has(row[\'sha\'], \'blob\'):\n                    row[\'size\'] = int(store._git(\'cat-file\', \'-s\', row[\'sha\']).decode().strip())\n                result.append(row)\n                if recursive and row[\'type\'] == \'tree\':\n                    result.extend(walk(row[\'sha\'], row[\'path\'] + \'/\'))\n            return result\n        return {\'sha\': sha, \'truncated\': False, \'tree\': walk(sha)}\n\n    def contents(self, store, commit, path):\n        require(path and not path.startswith(\'/\') and \'\\\\\' not in path and \'..\' not in Path(path).parts,\n                \'Unsafe Git content path\')\n        listing = store._git(\'ls-tree\', \'-z\', commit, \'--\', path).split(b\'\\0\')\n        found = [value for value in listing if value]\n        require(len(found) == 1, \'Exact committed metadata file is required\')\n        header, actual_path = found[0].split(b\'\\t\', 1)\n        mode, kind, sha = header.decode(\'ascii\').split(\' \')\n        require(actual_path.decode(\'utf-8\') == path and mode == \'100644\' and kind == \'blob\',\n                \'Committed metadata has another path or type\')\n        raw = store.read_blob(sha)\n        return {\'type\': \'file\', \'path\': path, \'encoding\': \'base64\',\n                \'content\': base64.b64encode(raw).decode(), \'sha\': sha, \'size\': len(raw)}\n\n\nclass GitPackAdapter:\n    def __init__(self, api, context):\n        self.api, self.original, self.context = api, api._request, context\n\n    def normalize_tree(self, response, recursive):\n        require(isinstance(response, dict) and response.get(\'truncated\') is False\n                and isinstance(response.get(\'tree\'), list), \'A complete Git tree response is required\')\n        rows = []\n        for row in response[\'tree\']:\n            clean = {key: row[key] for key in (\'path\', \'mode\', \'type\', \'sha\')}\n            if recursive and \'size\' in row:\n                clean[\'size\'] = row[\'size\']\n            rows.append(clean)\n        if not recursive:\n            require(tree_sha(rows) == response.get(\'sha\'), \'Tree response does not match its Git SHA\')\n        return {\'sha\': response[\'sha\'], \'truncated\': False, \'tree\': rows}\n\n    def request(self, method, path, token, payload=None, **kwargs):\n        self.api._inputs(method, path, token, payload)\n        if method == \'POST\' and path == ROOT + \'/git/blobs\':\n            require(isinstance(payload, dict) and set(payload) == {\'encoding\', \'content\'}\n                    and payload[\'encoding\'] == \'base64\', \'Only original base64 blob writes are accepted\')\n            raw = base64.b64decode(payload[\'content\'], validate=True)\n            return {\'sha\': self.context.blob(raw)}\n        if method == \'POST\' and path == ROOT + \'/git/trees\' and isinstance(payload, dict):\n            entries = payload.get(\'tree\', [])\n            if isinstance(entries, list) and any(isinstance(row, dict)\n                    and str(row.get(\'path\', \'\')).startswith(\'runs/e3_extended_audit/\') for row in entries):\n                return self.context.tree(payload)\n        if method == \'POST\' and path == ROOT + \'/git/commits\' and isinstance(payload, dict):\n            if str(payload.get(\'message\', \'\')).startswith(\'Guardar delta auditoría E3 \'):\n                return self.context.commit(payload)\n        if method == \'POST\' and path == ROOT + \'/git/refs\' and isinstance(payload, dict):\n            if str(payload.get(\'ref\', \'\')).startswith(\'refs/heads/e3-extended-audit-\'):\n                return self.context.reference(payload)\n        prefix = ROOT + \'/git/matching-refs/heads/e3-extended-audit-\'\n        if method == \'GET\' and path == prefix:\n            return self.context.refs(\'refs/heads/e3-extended-audit-\')\n        if method == \'GET\' and path.startswith(ROOT + \'/git/ref/heads/e3-extended-audit-\'):\n            ref = \'refs/heads/\' + path.removeprefix(ROOT + \'/git/ref/heads/\')\n            result = self.context.ref(ref)\n            if result is None and not kwargs.get(\'allow_missing\', False):\n                raise self.api.StorageError(\'Required audit reference is not published: \' + ref)\n            return result\n        parsed = urlsplit(path)\n        if method == \'GET\' and parsed.path.startswith(ROOT + \'/git/trees/\'):\n            sha = parsed.path.rsplit(\'/\', 1)[1]\n            recursive = \'recursive\' in parse_qs(parsed.query)\n            if sha in self.context.ready_trees:\n                value = self.context.read_tree(self.context.ready_trees[sha], sha, recursive)\n            else:\n                try:\n                    value = self.original(method, path, token, payload, **kwargs)\n                except Exception as error:\n                    if (type(error).__name__ != \'StorageError\' or \'status=404\' not in str(error)\n                            or sha not in self.context.tree_owners):\n                        raise\n                    self.context.fetch_commit(self.context.tree_owners[sha])\n                    require(sha in self.context.ready_trees, \'Fetched commit does not contain the required tree\')\n                    value = self.context.read_tree(self.context.ready_trees[sha], sha, recursive)\n                if value is None:\n                    return None\n            owner = self.context.tree_owners.get(sha)\n            if owner is not None:\n                for row in value[\'tree\']:\n                    if row.get(\'type\') == \'tree\':\n                        self.context.tree_owners[row[\'sha\']] = owner\n            return self.normalize_tree(value, recursive)\n        if method == \'GET\' and path.startswith(ROOT + \'/git/commits/\'):\n            sha = path.rsplit(\'/\', 1)[1]\n            if sha in self.context.ready_commits:\n                return self.context.ready_commits[sha].read_commit(sha)\n            try:\n                value = self.original(method, path, token, payload, **kwargs)\n            except Exception as error:\n                if (type(error).__name__ != \'StorageError\' or \'status=404\' not in str(error)\n                        or sha not in self.context.advertised):\n                    raise\n                value = self.context.fetch_commit(sha).read_commit(sha)\n            require(value.get(\'sha\') == sha and isinstance(value.get(\'tree\'), dict), \'Commit response differs\')\n            self.context.tree_owners[value[\'tree\'][\'sha\']] = sha\n            return value\n        if method == \'GET\' and parsed.path.startswith(ROOT + \'/contents/\'):\n            query = parse_qs(parsed.query)\n            commits = query.get(\'ref\', [])\n            if len(commits) == 1 and commits[0] in self.context.ready_commits:\n                relative = unquote(parsed.path.removeprefix(ROOT + \'/contents/\'))\n                return self.context.contents(self.context.ready_commits[commits[0]], commits[0], relative)\n        return self.original(method, path, token, payload, **kwargs)\n', 'git_api_provider.py', "exec"), _e3_module.__dict__)

"""Publish the exact pending snapshot through Git PACK, then resume.

AI assistance declared. Scientific source, protocol, job bytes and deadlines
remain unchanged. Authoritative commits, refs and state are still handled by
the original frozen continuation implementation.
"""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from urllib.error import HTTPError
from getpass import getpass
from git_api_provider import GitPackAdapter, GitPackContext

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
REPO = 'jassan-bit/DeepLearning_202630'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119/orchestration'
ROOT_API = '/repos/' + REPO
SEED_SHA256 = '32a90c1eee50dfb1571dfee51b342923addbe34045a454b6b0871ba0803b5ae8'
ALLOWED_READ = re.compile(r'/repos/jassan-bit/DeepLearning_202630/git/'
    r'(?:trees/[0-9a-f]{40}(?:\?recursive=1)?|ref/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z)')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def emit(event):
    print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


class CaptureTreeError:
    def __init__(self, original, token):
        self.original, self.token = original, token

    def open(self, request, **kwargs):
        if request.get_method() == 'GET' and request.full_url.startswith('https://api.github.com' + ROOT_API + '/'):
            request.add_header('Cache-Control', 'no-cache')
            request.add_header('Pragma', 'no-cache')
        try:
            return self.original.open(request, **kwargs)
        except HTTPError as error:
            if (error.code == 422 and request.get_method() == 'POST'
                    and request.full_url == 'https://api.github.com' + ROOT_API + '/git/trees'):
                raw = error.read(65537)
                try:
                    body = json.loads(raw) if len(raw) <= 65536 else {}
                except (ValueError, UnicodeError):
                    body = {}
                if not isinstance(body, dict):
                    body = {}
                message = str(body.get('message', 'No readable JSON message')).replace(self.token, '[REDACTED]')[:2000]
                emit({'event': 'incremental_tree_rejected', 'http': 422, 'message': message})
            raise


def install_api(api, token, known_blobs):
    if getattr(api, '_e3_git_pack_installed', False):
        return api
    api._opener = CaptureTreeError(api._opener, token)
    adapter = GitPackAdapter(api, known_blobs)
    api._request = adapter.request
    original_blob = api.raw_blob

    def raw_blob(sha, size, supplied_token):
        api._inputs('GET', ROOT_API + '/git/blobs/' + str(sha), supplied_token, None)
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{40}', sha)
                and type(size) is int and 0 <= size <= 48 * 1024**2, 'Invalid raw blob identity')
        if sha in known_blobs.ready_blobs:
            raw = known_blobs.ready_blobs[sha].read_blob(sha)
            require(len(raw) == size and api._git_sha(raw) == sha, 'Fetched Git blob differs')
            return raw
        return original_blob(sha, size, supplied_token)

    api.raw_blob = raw_blob
    api._e3_git_pack_installed = True
    return api


def prepare_api(checkout, token, known_blobs):
    legacy = load(checkout / 'experiments/e3_remediacion_20261118/evaluation/colab_github.py', 'incremental_legacy')
    transport = load(checkout / SCOPE / 'github_transport.py', 'incremental_transport')
    api = transport.wrap(legacy)
    if known_blobs.identity is None:
        actor = api._request('GET', '/user', token)
        require(isinstance(actor, dict) and isinstance(actor.get('login'), str)
                and type(actor.get('id')) is int, 'An authenticated Git author is required')
        known_blobs.identity = {'name': actor.get('name') or actor['login'],
            'email': str(actor['id']) + '+' + actor['login'] + '@users.noreply.github.com'}
    return install_api(api, token, known_blobs)


def finish_pending(base, token, api, continuation, known_blobs):
    statepath = api._regular_path(base / ('e3-extended-continuation-' + MASTER + '.json'))
    if not statepath.is_file():
        return {'status': 'no_local_state'}
    state_raw = statepath.read_bytes()
    state = json.loads(state_raw)
    continuation._compatible(state, SOURCE, PROTOCOL, MASTER, api)
    require(state.get('kind') == 'e3_audit_continuation_state' and state.get('schema_version') == 1,
            'Unexpected continuation state')
    protocol = api._regular_path(base / ('e3-extended-audit-' + MASTER) / 'protocol.json')
    require(hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL, 'Frozen protocol differs')
    pending = state.get('pending')
    if pending is None:
        return {'status': 'no_pending_snapshot'}
    require(isinstance(pending, dict), 'Invalid pending snapshot')
    snapshot = api._regular_path(pending['snapshot'], directory=True)
    receipt = json.loads((snapshot / 'stage_receipt.json').read_bytes())
    part = pending.get('part_id')
    require(isinstance(part, str) and re.fullmatch(r'[0-9]{8}T[0-9]{12}Z', part)
            and receipt.get('part_id') == part and receipt.get('source_commit') == SOURCE
            and receipt.get('master_run_id') == MASTER and receipt.get('protocol_sha256') == PROTOCOL
            and receipt.get('sequence') == state.get('sequence', -1) + 1
            and receipt.get('previous_part') == {'commit_sha': state.get('head_commit'),
                'part_id': state.get('head_part'), 'receipt_sha256': state.get('head_receipt_sha256')},
            'Pending snapshot does not extend this saved head')
    files = continuation.STORE.snapshot(snapshot, token, SOURCE, part, api, 'audit')
    require(sum(map(len, files.values())) <= continuation.MAX_PART, 'Pending snapshot exceeds its limit')
    payload = {n.removeprefix('payload/'): b for n, b in files.items() if n.startswith('payload/')}
    require(isinstance(receipt.get('files'), dict) and set(payload) == set(receipt['files'])
            and all(continuation._meta(raw, api) == receipt['files'][name] for name, raw in payload.items()),
            'Pending payload differs from its immutable receipt')
    prefix = 'runs/e3_extended_audit/' + part
    reference = api._request('GET', ROOT_API + '/git/ref/heads/e3-extended-audit-' + part,
                             token, allow_missing=True)
    if reference is None:
        parent = continuation._commit(api, token, state['head_commit'])
        expected = {name: api._git_sha(raw) for name, raw in files.items()}
        request = {'base_tree': parent['tree']['sha'], 'tree': [
            {'path': prefix + '/' + name, 'mode': '100644', 'type': 'blob', 'sha': sha}
            for name, sha in sorted(expected.items())]}
        print('Preparando Git PACK de la misma parte pendiente:', part, flush=True)
        for raw in files.values():
            known_blobs.blob(raw)
        tree = api._request('POST', ROOT_API + '/git/trees', token, request)
        leaf = continuation._subtree(api, token, tree['sha'], prefix)
        rows = continuation._tree(api, token, leaf, recursive=True)
        blobs = {name: row for name, row in rows.items() if row.get('type') == 'blob'}
        require(set(blobs) == set(files) and all(
            blobs[name].get('mode') == '100644' and blobs[name].get('sha') == expected[name]
            and blobs[name].get('size') == len(raw) for name, raw in files.items()),
            'Remote incremental tree differs from the exact pending snapshot')
        # Cache only blobs whose actual remote tree inventory and sizes were
        # checked. The frozen publisher will still verify the complete tree.
        # Git publication and all original inventory checks have completed.
    require(statepath.read_bytes() == state_raw, 'Continuation state changed concurrently')
    saved = continuation._finish_pending(state, statepath, api, token)
    committed = json.loads(statepath.read_bytes())
    require(committed.get('pending') is None and committed.get('head_part') == part
            and committed.get('source_commit') == SOURCE and committed.get('protocol_sha256') == PROTOCOL,
            'Original publisher did not complete this exact pending snapshot')
    print('Parte pendiente guardada y verificada:', saved['run_url'], flush=True)
    return {'status': 'pending_saved_by_original_publisher', **saved}


def install_controller(controller, checkout, token, known_blobs):
    original_load = controller.load
    target = (checkout / SCOPE / 'github_transport.py').resolve()

    def installed_load(path, name):
        module = original_load(path, name)
        if name == 'extended_controller_transport' and Path(path).resolve() == target:
            original_wrap = module.wrap

            def wrap(legacy, **kwargs):
                return install_api(original_wrap(legacy, **kwargs), token, known_blobs)
            module.wrap = wrap
        return module
    controller.load = installed_load
    return controller


def retryable_read(error, checkout):
    if type(error).__name__ != 'StorageError':
        return False
    if str(error).startswith('github_transport_network_read_failed '):
        return True
    if str(error) != 'github_transport_http_error operation=git_trees_read status=404 attempts=1':
        return False
    target = (checkout / SCOPE / 'github_transport.py').resolve()
    trace = error.__traceback__
    while trace is not None:
        frame, values = trace.tb_frame, trace.tb_frame.f_locals
        if (frame.f_code.co_name == '_perform' and Path(frame.f_code.co_filename).resolve() == target
                and values.get('method') == 'GET' and type(values.get('status')) is int
                and values['status'] == 404 and values.get('allow_missing') is False
                and isinstance(values.get('path'), str) and ALLOWED_READ.fullmatch(values['path'])):
            return True
        trace = trace.tb_next
    return False


def checkout_source(base):
    checkout = base / ('e3-extended-source-' + SOURCE)
    require(not any(p.is_symlink() for p in (checkout, base, *base.parents)), 'Symlink checkout refused')
    env = dict(os.environ)
    for key in list(env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)', key.upper()):
            env.pop(key, None)
    env.update(GIT_TERMINAL_PROMPT='0', PYTHONNOUSERSITE='1')
    if not checkout.exists():
        sparse = ['src', 'scripts', 'experiments/e3_evaluacion_ampliada_20261119', 'deployment/science',
            'experiments/e3_cierre_integral_20261118/campaign', 'experiments/e3_cierre_integral_20261118/data',
            'experiments/e3_cierre_integral_20261118/orchestration', 'experiments/e3_remediacion_20261118/evaluation',
            'experiments/e3_dataset_admission_20261118/inputs', 'experiments/e3_d01_admission_20261118',
            'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification', 'experiments/e3_estadistica_posterior_20261117']
        subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                        'https://github.com/' + REPO + '.git', str(checkout)], env=env, check=True)
        for args in (['sparse-checkout', 'init', '--cone'], ['sparse-checkout', 'set', *sparse],
                     ['fetch', '--depth', '1', 'origin', SOURCE], ['checkout', '--detach', SOURCE]):
            subprocess.run(['git', *args], cwd=checkout, env=env, check=True)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, env=env, text=True).strip()
    require(actual == SOURCE, 'Scientific checkout SHA differs')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=checkout, env=env, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return checkout


def main(namespace, *, base=None):
    base = Path('/content') if base is None else Path(base).absolute()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    print('GPU asignada:', gpu, flush=True)
    require(gpu == 'NVIDIA A100-SXM4-80GB', 'Resume requires the original A100 80 GB cohort')
    checkout = checkout_source(base)
    token = (os.environ.get('GITHUB_TOKEN') or namespace.get('_E3_GITHUB_TOKEN') or '').strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get('GITHUB_TOKEN') or '').strip()
        except Exception:
            token = ''
    if not token:
        token = getpass('GITHUB_TOKEN (entrada oculta): ').strip()
    require(bool(token), 'A GitHub token is required')
    namespace['_E3_GITHUB_TOKEN'] = token
    seed_url = ('https://raw.githubusercontent.com/' + REPO + '/88aa4a7bdd20f6485880a895cb251b21ae049fcc/'
                'runs/e3_view_reseed_resume/20261119T151114001132Z/resume_cell.py')
    with urllib.request.urlopen(seed_url, timeout=60) as response:
        seed_source = response.read()
    require(hashlib.sha256(seed_source).hexdigest() == SEED_SHA256, 'Immutable view helper hash differs')
    seed = {'__name__': 'incremental_view_helper'}
    exec(compile(seed_source, 'e3_verified_view_seed_helper', 'exec'), seed)
    known_blobs = GitPackContext(base, token)
    view_retry_used = False
    read_retries = 0
    while True:
        try:
            api = prepare_api(checkout, token, known_blobs)
            continuation = load(checkout / SCOPE / 'continuation.py', 'incremental_continuation')
            namespace['_E3_INCREMENTAL_PENDING_SAVE'] = finish_pending(base, token, api, continuation, known_blobs)
            seed['seed_view'](base)
            controller = load(checkout / SCOPE / 'controller.py', 'incremental_original_controller')
            install_controller(controller, checkout, token, known_blobs)
            result = controller.run(SOURCE, token, base=base, start_new=False, master_run_id=MASTER)
            namespace['_E3_EXTENDED_AUDIT_RESULT'] = result
            return result
        except Exception as error:
            if not view_retry_used and seed['_expected_view_error'](error, base):
                view_retry_used = True
                seed['seed_view'](base, required=True)
                continue
            if read_retries == 3 or not retryable_read(error, checkout):
                raise
            read_retries += 1
            print(f'Lectura de GitHub interrumpida; reanudación del mismo master en 30 s ({read_retries}/3).', flush=True)
            time.sleep(30)


if __name__ == '__main__':
    main(globals())
