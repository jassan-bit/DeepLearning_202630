import base64,hashlib,io,json,sys,tempfile,time,unittest,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_fresh_metadata as m

def archive(files):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
    return stream.getvalue()

class Client:
    def __init__(self,payloads,fallback=False):self.payloads=payloads;self.calls=[];self.fallback=fallback
    def _content(self,path,head):
        self.calls.append(path);raw=self.payloads[path]
        return {'sha':m.blob_sha(raw),'encoding':'none' if self.fallback else 'base64',
                'content':'' if self.fallback else base64.b64encode(raw).decode()}
    def _request(self,method,path):
        assert method=='GET'
        raw=next(raw for raw in self.payloads.values() if m.blob_sha(raw)==path.rsplit('/',1)[-1])
        return {'sha':m.blob_sha(raw),'content':base64.b64encode(raw).decode()}

def fixture(root,versions,expected):
    payloads={};deltas=[]
    for i,files in enumerate(versions):
        raw=archive(files);path='control/test/deltas/'+str(i)+'.zip';payloads[path]=raw
        deltas.append({'path':path,'bytes':len(raw),'sha256':m.sha(raw),'git_blob_sha':m.blob_sha(raw)})
    state={'version':'e2-guide-closure-posthoc-state-v1','scientific_commit':m.ENGINE_COMMIT,
           'output_path':str(root),'files':{p:{'bytes':len(raw),'sha256':m.sha(raw)} for p,raw in expected.items()},
           'metadata_deltas':deltas,'result_backups':{}}
    state['state_id']=m.sha(m.canonical(state));return state,Client(payloads)

class MetadataTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'output'
    def test_newest_subset_stops_before_old_history(self):
        state,client=fixture(self.root,[{'a.json':b'old'},{'b.json':b'oldb'},{'a.json':b'new'}],{'a.json':b'new','b.json':b'oldb'})
        result=m.restore_metadata(self.root,client,state,'a'*40,paths=['a.json'],parallel=1)
        self.assertEqual((self.root/'a.json').read_bytes(),b'new');self.assertEqual(len(client.calls),1)
        self.assertEqual(result['requested_files'],1);self.assertFalse((self.root/'b.json').exists())
    def test_all_files_and_exact_existing_reuse(self):
        state,client=fixture(self.root,[{'a.json':b'a'},{'b.json':b'b'}],{'a.json':b'a','b.json':b'b'})
        self.root.mkdir();(self.root/'a.json').write_bytes(b'a')
        result=m.restore_metadata(self.root,client,state,'a'*40,parallel=1)
        self.assertEqual(result['existing_identical_files'],1);self.assertEqual(len(client.calls),1)
        result=m.restore_metadata(self.root,client,state,'a'*40);self.assertEqual(result['deltas_downloaded'],0)
    def test_old_matching_size_but_wrong_hash_is_skipped(self):
        state,client=fixture(self.root,[{'a.json':b'new'},{'a.json':b'old'}],{'a.json':b'new'})
        m.restore_metadata(self.root,client,state,'a'*40,parallel=1)
        self.assertEqual((self.root/'a.json').read_bytes(),b'new');self.assertEqual(len(client.calls),2)
    def test_divergent_local_file_preserved_without_network(self):
        state,client=fixture(self.root,[{'a.json':b'new'}],{'a.json':b'new'});self.root.mkdir();(self.root/'a.json').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Existing local'):m.restore_metadata(self.root,client,state,'a'*40)
        self.assertEqual((self.root/'a.json').read_bytes(),b'changed');self.assertFalse(client.calls)
    def test_cache_reused_across_requested_subsets(self):
        state,client=fixture(self.root,[{'a.json':b'a','b.json':b'b'}],{'a.json':b'a','b.json':b'b'});cache=Path(self.temp.name)/'cache'
        m.restore_metadata(self.root,client,state,'a'*40,paths=['a.json'],cache_root=cache)
        result=m.restore_metadata(self.root,client,state,'a'*40,paths=['b.json'],cache_root=cache)
        self.assertEqual(result['cached_deltas_reused'],1);self.assertEqual(len(client.calls),1)
    def test_tampered_state_or_wrong_path_rejected_before_reads(self):
        state,client=fixture(self.root,[{'a.json':b'a'}],{'a.json':b'a'});state['files']['a.json']['bytes']=2
        with self.assertRaisesRegex(ValueError,'identity'):m.restore_metadata(self.root,client,state,'a'*40)
        state,client=fixture(self.root,[{'a.json':b'a'}],{'a.json':b'a'})
        with self.assertRaisesRegex(ValueError,'absolute'):m.restore_metadata(self.root/'other',client,state,'a'*40)
        self.assertFalse(client.calls)
    def test_path_traversal_in_even_unused_zip_member_rejected(self):
        state,client=fixture(self.root,[{'a.json':b'a','../outside.json':b'x'}],{'a.json':b'a'})
        with self.assertRaisesRegex(ValueError,'Unsafe'):m.restore_metadata(self.root,client,state,'a'*40)
        self.assertFalse((self.root/'a.json').exists())
    def test_corrupted_remote_blob_rejected(self):
        state,client=fixture(self.root,[{'a.json':b'a'}],{'a.json':b'a'})
        path=state['metadata_deltas'][0]['path'];client.payloads[path]=b'changed'
        with self.assertRaisesRegex(ValueError,'identity'):m.restore_metadata(self.root,client,state,'a'*40)
    def test_git_blob_fallback_and_primary_version(self):
        state,client=fixture(self.root,[{'a.json':b'a'}],{'a.json':b'a'});state['version']='e2-github-primary-state-v1'
        state['state_id']=m.sha(m.canonical({k:v for k,v in state.items() if k!='state_id'}));client.fallback=True
        m.restore_metadata(self.root,client,state,'a'*40);self.assertEqual((self.root/'a.json').read_bytes(),b'a')
    def test_expired_deadline_and_missing_current_version_fail(self):
        state,client=fixture(self.root,[{'a.json':b'old'}],{'a.json':b'new'})
        with self.assertRaises(TimeoutError):m.restore_metadata(self.root,client,state,'a'*40,deadline=time.monotonic()-1)
        self.assertFalse(client.calls)
        with self.assertRaisesRegex(ValueError,'not found'):m.restore_metadata(self.root,client,state,'a'*40)
    def test_symlink_destination_rejected(self):
        state,client=fixture(self.root,[{'a.json':b'a'}],{'a.json':b'a'});self.root.mkdir()
        elsewhere=Path(self.temp.name)/'elsewhere';elsewhere.write_bytes(b'a');(self.root/'a.json').symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError,'Symlink'):m.restore_metadata(self.root,client,state,'a'*40)
        self.assertFalse(client.calls)

if __name__=='__main__':unittest.main()
