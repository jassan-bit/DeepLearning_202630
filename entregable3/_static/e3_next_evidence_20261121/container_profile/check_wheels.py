from pathlib import Path
import hashlib,json
root=Path('/opt/e3');m=json.loads((root/'WHEELHOUSE_MANIFEST.json').read_text());w=root/'wheels'
assert hashlib.sha256((root/'requirements-book.lock.txt').read_bytes()).hexdigest()==m['lock_sha256']
assert {p.name for p in w.iterdir()}=={x['path'] for x in m['files']}
for row in m['files']:
 p=w/row['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'wheel_files_verified':len(m['files']),'lock_sha256':m['lock_sha256'],'network_used_for_install':False}))
