"""Archive and data-scope guard fixtures only; no market/model execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('consolidate',Path(__file__).with_name('consolidate_archive.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def manifest_for(files,**fields):
    return dict(files=[{'path':name,'bytes':len(raw),'sha256':m.sha(raw)} for name,raw in sorted(files.items())],**fields)


def base_fixture(path,files,override_manifest=None,unsafe_item=None):
    manifest=override_manifest or manifest_for(files,source_commit=m.SOURCE,capsule_version='0.3.0-REVIEW',
                                              Zenodo_publication_performed=False,global_MIT_or_CC_license_assigned=False)
    members={**files,'CAPSULE_MANIFEST.json':m.json_bytes(manifest)}
    with tarfile.open(path,'w:gz') as tar:
        for name,raw in members.items():
            info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
        if unsafe_item is not None:tar.addfile(unsafe_item)
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':m.sha(raw),'members':len(members)+(unsafe_item is not None),
            'expanded_bytes':sum(len(x) for x in members.values())}


def nine_fixture(model_value=b'ONNX fixture; not executed',model_expected=None,unsafe_path=False):
    name='../model.onnx' if unsafe_path else 'standard/fixture/model.onnx'
    metadata=b'{"q": 0.123456789, "unchanged": true}\n'
    files={name:model_value,'standard/fixture/metadata.json':metadata}
    records={name:{'bytes':len(raw),'sha256':m.sha(raw)} for name,raw in files.items()}
    manifest={'source_commit':'a'*40,'files':records,'models':{name:records[name]}}
    manifest_raw=m.json_bytes(manifest);stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for key,value in {**files,'exports_manifest.json':manifest_raw}.items():z.writestr(key,value)
    raw=stream.getvalue();pinned={'bytes':len(raw),'sha256':m.sha(raw),'manifest_bytes':len(manifest_raw),
                                'manifest_sha256':m.sha(manifest_raw),'members':3}
    value=model_value if model_expected is None else model_expected
    models={name:(len(value),m.sha(value))}
    return raw,pinned,models,metadata


class Guards(unittest.TestCase):
    def test_pinned_base_and_every_inner_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'base.tgz';payload=p/'payload';payload.mkdir();files={'README.md':b'review','a/model.onnx':b'fixture'}
            expected=base_fixture(f,files)
            report,_=m.read_base(f,payload,expected,{m.sha(b'fixture')})
            self.assertEqual(report['members'],3);self.assertEqual((payload/'README.md').read_bytes(),b'review')

    def test_corrupt_base_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'base.tgz';expected=base_fixture(f,{'README.md':b'original'})
            raw=bytearray(f.read_bytes());raw[-1]^=1;f.write_bytes(raw);payload=p/'payload';payload.mkdir()
            with self.assertRaisesRegex(ValueError,'SHA256'):m.read_base(f,payload,expected,set())
            self.assertFalse(any(payload.iterdir()))

    def test_rehashed_outer_archive_does_not_hide_bad_inner_member(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'base.tgz';files={'README.md':b'changed'}
            manifest=manifest_for({'README.md':b'original'},source_commit=m.SOURCE,capsule_version='0.3.0-REVIEW',
                                  Zenodo_publication_performed=False,global_MIT_or_CC_license_assigned=False)
            expected=base_fixture(f,files,manifest);payload=p/'payload';payload.mkdir()
            with self.assertRaisesRegex(ValueError,'member hash'):m.read_base(f,payload,expected,set())

    def test_base_traversal_and_symlink_are_rejected(self):
        for kind in ['traversal','symlink']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                p=Path(d);f=p/'base.tgz';item=tarfile.TarInfo('../escape' if kind=='traversal' else 'link')
                if kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname='/outside'
                expected=base_fixture(f,{'README.md':b'review'},unsafe_item=item);payload=p/'payload';payload.mkdir()
                with self.assertRaises(ValueError):m.read_base(f,payload,expected,set())
                self.assertFalse((p/'escape').exists())

    def test_nine_member_metadata_bytes_preserved(self):
        raw,pinned,models,metadata=nine_fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=m.add_nine(raw,p,pinned,models)
            self.assertTrue(r['all_member_bytes_preserved']);self.assertFalse(r['weights_scalers_calibrators_changed'])
            self.assertEqual((p/'approved_external_nine/standard/fixture/metadata.json').read_bytes(),metadata)

    def test_nine_zip_inner_graph_must_match_independent_approved_hash(self):
        raw,pinned,models,_=nine_fixture(model_value=b'tampered graph',model_expected=b'original graph')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.add_nine(raw,Path(d),pinned,models)

    def test_nine_zip_cannot_escape_or_replace_members(self):
        raw,pinned,models,_=nine_fixture(unsafe_path=True)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.add_nine(raw,Path(d),pinned,models)

    def test_C_manifest_and_source_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/'C';root.mkdir();source=root/'driver.py';source.write_bytes(b'# source only\n')
            manifest=root/'MANIFEST.json';manifest.write_bytes(m.json_bytes(manifest_for({'driver.py':source.read_bytes()})))
            result=m.add_C(root,manifest,m.file_sha(manifest),p/'payload')
            self.assertTrue(result['included']);self.assertFalse(result['training_executed'])
            source.write_bytes(b'# changed\n')
            with self.assertRaises(ValueError):m.add_C(root,manifest,m.file_sha(manifest),p/'payload2')

    def test_C_rejects_raw_NPZ_and_unapproved_CSV_even_with_manifest(self):
        for name,raw in [('inputs/cohort.npz',b'raw fixture'),('references/other.csv',b'open,high,low,close\n1,2,0,1\n')]:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                p=Path(d);root=p/'C';root.mkdir();f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
                manifest=root/'MANIFEST.json';manifest.write_bytes(m.json_bytes(manifest_for({name:raw})))
                with self.assertRaises(ValueError):m.add_C(root,manifest,m.file_sha(manifest),p/'payload')

    def test_rebuild_is_deterministic_and_reextraction_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);manifests=[];files=[]
            for i in range(2):
                payload=p/('payload'+str(i));payload.mkdir();(payload/'model.onnx').write_bytes(b'model fixture');(payload/'note.txt').write_bytes(b'fixed\n')
                target=p/('archive'+str(i)+'.tgz');manifest,_=m.build(payload,target,{'fixture':True});manifests.append(manifest);files.append(target)
            self.assertEqual(files[0].read_bytes(),files[1].read_bytes())
            actual=m.verify_final(files[0],manifests[0],{m.sha(b'model fixture')});self.assertTrue(actual['all_member_hashes_verified'])
            bad=json.loads(json.dumps(manifests[0]));bad['files'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):m.verify_final(files[0],bad,{m.sha(b'model fixture')})

    def test_public_download_origin_credentials_and_protocol_guards(self):
        for url in ['http://github.com/a','https://evil.example/a','https://user:secret@github.com/a','https://github.com:444/a']:
            with self.subTest(url=url),self.assertRaises(ValueError):m.allowed_url(url)

    def test_production_pins_distinguish14_historical_plus1_experimental(self):
        self.assertEqual(len(m.MODELS),9);self.assertEqual(len(m.BASE_MODELS),6);self.assertEqual(len(m.ALL_MODELS),15)
        self.assertEqual(m.BASE['bytes'],58823392);self.assertEqual(m.NINE['bytes'],12987835)


if __name__=='__main__':unittest.main()
