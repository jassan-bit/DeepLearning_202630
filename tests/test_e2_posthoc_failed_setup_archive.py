import fcntl,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_failed_setup_archive as m
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def checked(v):
    if v.get('sha256')!=digest({k:x for k,x in v.items() if k!='sha256'}):raise ValueError('checksum')
def immutable(path,value):
    path=Path(path);raw=canonical(value)+b'\n';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:raise ValueError('immutable proof differs')
    if not path.exists():path.write_bytes(raw)
def fixture(temp,administrative=False):
    root=Path(temp)/'campaign';root.mkdir();(root/'budget').mkdir();remote=SimpleNamespace(backups=[])
    def backup(directory,receipt,client):
        for relative,expected in receipt['artifacts_sha256'].items():
            if m.sha((Path(directory)/relative).read_bytes())!=expected:raise ValueError('backup bytes')
        client.backups.append((Path(directory),receipt));return {'status':'backed_up'}
    old=SimpleNamespace(VERSION='original-test-version',ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7',
        ZERO='0'*64,checked=checked,digest=digest,immutable_json=immutable,backup_completed=backup)
    rows=[]
    def append(kind,**fields):
        row={'version':old.VERSION,'kind':kind,'sequence':len(rows),'utc':'2026-11-13T12:00:00+00:00',
             'previous_sha256':rows[-1]['sha256'] if rows else old.ZERO,'anchor_sha256':'a'*64,**fields}
        row['sha256']=digest(row);rows.append(row)
    append('reserved',job_id='clean-setup-original',seconds=2400.,metadata={'scope':'source_checkout_and_actual_new_venv_install'})
    append('charged',job_id='clean-setup-original',reservation_seconds=2400.,elapsed_seconds=2.,status='failed')
    active={}
    if administrative:
        append('reserved',job_id='archive-administrative',seconds=240.,metadata={'scope':'failed_clean_setup_archive','training_performed':False});active={'archive-administrative':240.}
    path=root/'budget/posthoc_budget.jsonl';path.write_bytes(b''.join(canonical(r)+b'\n' for r in rows))
    lock=(root/'budget/.posthoc-budget.lock').open('a+b');original=(root/'original.lock').open('a+b')
    for value in (lock,original):fcntl.flock(value.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=SimpleNamespace(root=root/'budget',path=path,events=rows,active=active,persistence=remote,lock=lock,original_lock=original)
    clean=root/'reproduction/clean';clean.mkdir(parents=True)
    immutable(clean/'setup_started.json',{'version':old.VERSION,'scientific_commit':old.ENGINE_COMMIT,'no_system_site_packages':True})
    (clean/'checkout.log').write_bytes(b'Cloning exact original source\nfatal: local clone failure\n')
    objects=clean/'source/.git/objects/pack';objects.mkdir(parents=True);(objects/'partial.pack').write_bytes(b'original Git object bytes')
    immutable(root/'robustness/result.json',{'status':'completed','run_id':'frozen-same-run'})
    for i in range(79):immutable(root/'robustness/predictors'/str(i)/'result.json',{'status':'completed','run_id':'frozen-same-run'})
    registration={'original_configuration_commit':m.PIN,'files_sha256':{m.HELPER:m.sha(Path(m.__file__).read_bytes())}}
    return SimpleNamespace(root=root,old=old,budget=budget,remote=remote,registration=registration,clean=clean,append=append)
class ArchiveTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    def f(self,administrative=False):
        x=fixture(self.temp.name,administrative);self.addCleanup(x.budget.lock.close);self.addCleanup(x.budget.original_lock.close);return x
    def test_exact_tree_and_failed_charge_preserved_only_metadata_backed_up(self):
        x=self.f();before=x.budget.path.read_bytes();tree=m.fingerprint(x.clean)
        result=m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration);archive=Path(result['archive'])
        self.assertFalse(x.clean.exists());self.assertEqual(m.fingerprint(archive/'preserved_clean'),tree)
        self.assertEqual(x.budget.path.read_bytes(),before)
        self.assertEqual(set(x.remote.backups[0][1]['artifacts_sha256']),{'inventory.json','checkout.log','setup_started.json','amendment.json'})
    def test_administrative_lease_untouched(self):
        x=self.f(True);active=dict(x.budget.active);m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration);self.assertEqual(x.budget.active,active)
    def test_verified_rerun_reuses_original_proof(self):
        x=self.f();first=m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        path=Path(first['archive'])/'amendment.json';raw=path.read_bytes()
        second=m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        self.assertTrue(second['reused_verified_archive']);self.assertEqual(path.read_bytes(),raw)
    def test_new_environment_or_modified_archive_rejected(self):
        x=self.f();result=m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration);x.clean.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError,'new clean setup'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        x.clean.rmdir();(Path(result['archive'])/'preserved_clean/checkout.log').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'bytes differ'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
    def test_optimizer_or_historical_fit_reservation_blocks_archive(self):
        x=self.f();job=x.root/'reproduction/jobs/fit/worker_spec.json';job.parent.mkdir(parents=True);job.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'optimizer'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        job.unlink();job.parent.rmdir();job.parent.parent.rmdir()
        x.append('reserved',job_id='posthoc-reproduction-already-admitted',seconds=1860.,metadata={'kind':'reproduction'})
        x.budget.path.write_bytes(b''.join(canonical(r)+b'\n' for r in x.budget.events))
        with self.assertRaisesRegex(RuntimeError,'model fit'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
    def test_setup_progress_weights_symlinks_rejected(self):
        x=self.f();(x.clean/'venv').mkdir()
        with self.assertRaisesRegex(RuntimeError,'beyond clone'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        (x.clean/'venv').rmdir();(x.clean/'model.pt').write_bytes(b'weights')
        with self.assertRaisesRegex(ValueError,'Weights'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        (x.clean/'model.pt').unlink();(x.clean/'unsafe').symlink_to(x.clean/'checkout.log')
        with self.assertRaisesRegex(ValueError,'Symlink'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
    def test_incomplete_diagnostics_or_changed_helper_rejected(self):
        x=self.f();(x.root/'robustness/predictors/78/result.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'All79'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
        x.registration['files_sha256'][m.HELPER]='0'*64
        with self.assertRaisesRegex(ValueError,'registration'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
    def test_failed_charge_must_be_terminal(self):
        x=self.f();row=x.budget.events[1];row['status']='completed';row['sha256']=digest({k:v for k,v in row.items() if k!='sha256'})
        x.budget.path.write_bytes(b''.join(canonical(r)+b'\n' for r in x.budget.events))
        with self.assertRaisesRegex(RuntimeError,'FAILED'):m.archive_failed_setup(x.old,x.root,x.budget,x.remote,x.registration)
if __name__=='__main__':unittest.main()
