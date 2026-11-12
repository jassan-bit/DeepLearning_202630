"""Transport-only synthetic restore tests: no market values or fitting."""
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
for candidate in ('/workspace/work/e2architecture/scripts',str(Path(__file__).resolve().parents[1]/'reference/scripts')):
    if Path(candidate,'e2_reduced_github_backup.py').is_file():sys.path.append(candidate)
from e2_reduced_github_backup import BackupPause, canonical, git_blob_sha
from e2_reduced_github_restore import restore_results


def sha(data):return hashlib.sha256(data).hexdigest()


class Response(io.BytesIO):
    def geturl(self):return 'https://github.com/owner/repo/releases/download/tag/artifact'


class Client:
    timeout=1
    def __init__(self, files, assets):
        self.files,self.assets=files,assets
        class Public:
            def open(_,url,timeout=None):return Response(assets[url])
        self._public_opener=Public()
    def _content(self,path,head):
        value=self.files.get(path)
        return None if value is None else {'content':base64.b64encode(value).decode(),'encoding':'base64','sha':git_blob_sha(value)}


def fixture(rows,assets=None,transport=None):
    files={}
    manifest={'artifacts':rows}
    if transport is not None:manifest['archive_transport']=transport
    raw=canonical(manifest)+b'\n';files['runs/test/manifest.json']=raw
    index={'jobs/test':{'manifest_path':'runs/test/manifest.json','manifest_sha256':sha(raw)}}
    return Client(files,assets or {}),index


class RestoreTests(unittest.TestCase):
    def test_git_metadata_and_weight_asset_restored_exactly_and_reused(self):
        metadata=b'{"epoch":1}\n';weights=b'synthetic-only-tensor-bytes'
        rows=[{'path':'config.json','storage':'git_blob','remote_path':'runs/test/config.json','sha256':sha(metadata),'bytes':len(metadata),'git_blob_sha':git_blob_sha(metadata)},
              {'path':'selected.pt','storage':'release_asset','url':'https://github.com/x/y/releases/download/z/selected.pt','sha256':sha(weights),'bytes':len(weights)}]
        client,index=fixture(rows,{rows[1]['url']:weights});client.files[rows[0]['remote_path']]=metadata
        with tempfile.TemporaryDirectory() as root:
            r=restore_results(root,client,index,'a'*40)
            self.assertEqual(r['files_restored'],2)
            self.assertEqual(Path(root,'jobs/test/selected.pt').read_bytes(),weights)
            self.assertEqual(restore_results(root,client,index,'a'*40)['existing_identical_files'],2)

    def test_zip_members_and_manifest_hashes_are_verified(self):
        payload=b'epoch,loss\n1,0.1\n';stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as z:z.writestr('curves.csv',payload)
        archive=stream.getvalue();url='https://github.com/x/y/releases/download/z/meta.zip'
        transport={'kind':'deterministic_zip','url':url,'sha256':sha(archive),'bytes':len(archive)}
        row={'path':'curves.csv','archive_member':'curves.csv','storage':'release_archive_member','archive_sha256':sha(archive),'sha256':sha(payload),'bytes':len(payload)}
        client,index=fixture([row],{url:archive},transport)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(restore_results(root,client,index,'a'*40)['files_restored'],1)
            self.assertEqual(Path(root,'jobs/test/curves.csv').read_bytes(),payload)
            index['jobs/test']['manifest_sha256']='0'*64
            with self.assertRaisesRegex(BackupPause,'manifest_sha256'):restore_results(root,client,index,'a'*40)

    def test_bad_remote_bytes_and_local_conflicts_are_preserved(self):
        data=b'expected';row={'path':'weights.pt','storage':'release_asset','url':'https://github.com/x/y/releases/download/z/w','sha256':sha(data),'bytes':len(data)}
        client,index=fixture([row],{row['url']:b'badbytes'})
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(BackupPause,'sha256'):restore_results(root,client,index,'a'*40)
            target=Path(root,'jobs/test/weights.pt');self.assertFalse(target.exists())
            target.write_bytes(b'local-original')
            with self.assertRaisesRegex(BackupPause,'local_artifact_differs'):restore_results(root,client,index,'a'*40)
            self.assertEqual(target.read_bytes(),b'local-original')

    def test_unsafe_paths_and_asset_hosts_are_rejected(self):
        row={'path':'../escape.pt','storage':'release_asset','url':'https://example.com/w','sha256':sha(b'x'),'bytes':1}
        client,index=fixture([row])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(BackupPause,'invalid_restore_artifact_path'):restore_results(root,client,index,'a'*40)
            row['path']='w.pt';client,index=fixture([row])
            with self.assertRaisesRegex(BackupPause,'untrusted_asset'):restore_results(root,client,index,'a'*40)


if __name__=='__main__':unittest.main()
