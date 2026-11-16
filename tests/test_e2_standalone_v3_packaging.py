"""Synthetic packaging contracts; no deliverable ZIPs or scientific runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/filename)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


package=module("_packager_v3_tests","package_e2_standalone_deliverables.py")
reports=module("_reports_v3_tests","e2_standalone_reports.py")
analysis=module("_analysis_v3_tests","e2_standalone_analysis.py")
checker=module("_checker_v3_tests","check_e2_standalone_deliverables.py")


def blob(path):
    raw=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


class SourceAndNamespaceTests(unittest.TestCase):
    def test_journals_hydrate_from_committed_zip_without_individual_git_blobs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);science=root/"science";(science/"preregistration").mkdir(parents=True)
            archive=science/"preregistration_bundle.zip"
            member="Preregistro_E2/journals/"+"a"*64+"/supervised.jsonl.gz"
            raw=b"immutable synthetic journal member"
            with zipfile.ZipFile(archive,"w") as z:z.writestr(member,raw)
            archive_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
            definitions={"preregistration":{"file":archive.name,"prefix":"Preregistro_E2/journals/","count":1,"sha256":archive_hash}}
            blobs={archive.relative_to(root).as_posix():blob(archive)}
            with patch.multiple(package,ROOT=root,SCIENTIFIC=science,SOURCE_BLOBS=blobs,JOURNAL_ARCHIVES=definitions):
                rows=package.registration_journal_entries(root/"stage","preregistration")
            self.assertEqual(len(rows),1)
            self.assertFalse(rows[0]["individual_Git_blob_claimed"])
            self.assertEqual(rows[0]["source_binding"],"member_of_committed_registration_zip")
            self.assertEqual((root/"stage"/rows[0]["path"]).read_bytes(),raw)
            self.assertFalse((science/"preregistration/journals").exists())

    def test_main_rejects_modified_packager_before_generating_any_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);script=root/"scripts/package_e2_standalone_deliverables.py";script.parent.mkdir()
            script.write_bytes(b"original executing program")
            old_blob=blob(script);script.write_bytes(b"changed executing program")
            tree=("100644 blob "+old_blob+"\tscripts/package_e2_standalone_deliverables.py\0").encode()
            public=root/"new_v3_output"
            with patch.multiple(package,ROOT=root,__file__=str(script),SOURCE=None,SOURCE_BLOBS={}), \
                 patch.object(sys,"argv",[str(script),"--source-commit","a"*40,"--only","D03","--public-output",str(public)]), \
                 patch.object(package.subprocess,"check_output",side_effect=[("a"*40+"\n").encode(),tree]) as call:
                # rev-parse requests text=True, unlike ls-tree.
                call.side_effect=["a"*40+"\n",tree]
                with self.assertRaisesRegex(ValueError,"Working-tree bytes differ"):
                    package.main()
            self.assertFalse(public.exists())

    def test_changed_source_cannot_be_packaged_under_old_commit_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"a.json";source.write_text('{"status":"PASS"}')
            with patch.object(package,"ROOT",root),patch.object(package,"SOURCE_BLOBS",{"a.json":blob(source)}):
                package.verified_repository_source(source)
                source.write_text('{"status":"FAIL"}')
                with self.assertRaises(ValueError):package.verified_repository_source(source)

    def test_generations_keep_colliding_names_and_failed_status_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/"old/data_audit";new=root/"new/data_audit"
            old.mkdir(parents=True);new.mkdir(parents=True)
            (old/"summary.json").write_text('{"status":"PARTIAL"}')
            (new/"summary.json").write_text('{"status":"FAIL"}')
            (new/"journal.jsonl.gz").write_bytes(b"compressed synthetic fixture")
            (new/"model.pt").write_bytes(b"weights must not enter report bundles")
            (new/"frozen_inputs/work").mkdir(parents=True)
            (new/"frozen_inputs/work/input.json").write_text('{"required_source":true}')
            bound={p.relative_to(root).as_posix():blob(p) for p in root.rglob('*') if p.is_file()}
            dest=root/"stage";dest.mkdir()
            with patch.multiple(package,ROOT=root,CORRECTIONS=root/"old",SCIENTIFIC=root/"new",SOURCE_BLOBS=bound):
                result=package.correction_entries(dest,"D04")
            self.assertEqual(json.loads((dest/"corrections/data_audit/summary.json").read_text())["status"],"PARTIAL")
            self.assertEqual(json.loads((dest/"scientific_closure/data_audit/summary.json").read_text())["status"],"FAIL")
            self.assertTrue((dest/"scientific_closure/data_audit/journal.jsonl.gz").is_file())
            self.assertFalse((dest/"scientific_closure/data_audit/model.pt").exists())
            self.assertTrue((dest/"scientific_closure/data_audit/frozen_inputs/work/input.json").is_file())
            self.assertEqual({r["generation"] for r in result["areas"]},{"20261115","20261116"})

    def test_heavy_traces_remain_downloadable_without_embedding_them(self):
        self.assertFalse(package.correction_renderable(Path("causal_receipts/reference/a.json"),100))
        self.assertFalse(package.correction_renderable(Path("predictions_train.csv"),2_000_000))
        self.assertTrue(package.correction_renderable(Path("statistics/summary.json"),200))

    def test_new_default_is_v3_and_retains_twelve_independent_ids(self):
        self.assertEqual(package.PUBLIC.name,"entregables_independientes_v3_20261116")
        self.assertEqual(set(package.NAMES),{f"D{i:02}" for i in range(1,13)})
        self.assertIn("cpu_test_iut",package.SCIENTIFIC_AREAS["D05"])
        self.assertIn("reproduction",package.SCIENTIFIC_AREAS["D11"])


class CurrentDatasetBindingTests(unittest.TestCase):
    def fixture(self,root):
        science=root/"science";directory=science/"data_audit";directory.mkdir(parents=True)
        external=root/"work/known_E1_daily.csv";external.parent.mkdir();external.write_bytes(b"synthetic daily fixture\n")
        h=hashlib.sha256(external.read_bytes()).hexdigest()
        npz=directory/"current_cohort_arrays.npz";npz.write_bytes(b"synthetic I/O fixture, not valid arrays")
        for name in ["cohort_array_manifest.json","reconstructed_training_scaler.json"]:(directory/name).write_text('{}')
        audit={"actual_cohorts_audited":{"training":655,"calibration":53,"evaluation":608},"data_ready_for_bounded_controls":True,
            "provenance":[{"path":"work/known_E1_daily.csv","sha256":h,"bytes":external.stat().st_size,"expected_sha256_matched":True}],
            "arrays_npz_path":npz.relative_to(root).as_posix(),"arrays_npz_sha256":hashlib.sha256(npz.read_bytes()).hexdigest(),
            "array_keys":{"training":{},"calibration":{},"evaluation":{}}}
        (directory/"audit_result.json").write_text(json.dumps(audit))
        blobs={p.relative_to(root).as_posix():blob(p) for p in directory.iterdir()}
        return science,external,h,blobs

    def test_external_cache_has_independent_binding_and_all_five_inputs_are_local(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);science,external,h,blobs=self.fixture(root);out=root/"stage";out.mkdir()
            with patch.multiple(package,ROOT=root,SCIENTIFIC=science,EVALUATION_DAILY=external,EVALUATION_DAILY_SHA=h,SOURCE_BLOBS=blobs,SOURCE="a"*40):
                result=package.current_dataset_inputs(out)
            self.assertEqual(len(result["files"]),5)
            row=next(r for r in result["files"] if r["path"]=="daily_market_evaluation.csv")
            self.assertEqual(row["source_binding"],"independent_original_SHA_bound_by_committed_data_audit")
            self.assertEqual((out/"inputs/daily_market_evaluation.csv").read_bytes(),external.read_bytes())

    def test_changed_external_cache_cannot_bypass_committed_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);science,external,h,blobs=self.fixture(root);external.write_bytes(b"tampered\n")
            with patch.multiple(package,ROOT=root,SCIENTIFIC=science,EVALUATION_DAILY=external,EVALUATION_DAILY_SHA=h,SOURCE_BLOBS=blobs):
                with self.assertRaises(ValueError):package.current_dataset_inputs(root/"stage")

    def test_new_delivery_tables_take_precedence_without_altering_historical_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/"old/delivery";new=root/"new/delivery";old.mkdir(parents=True);new.mkdir(parents=True)
            (old/"appendix_B_current.csv").write_text('historical')
            (new/"appendix_B_current.csv").write_text('current')
            with patch.multiple(package,ROOT=root,CORRECTIONS=root/"old",SCIENTIFIC=root/"new"):
                self.assertEqual(package.current_delivery("appendix_B_current.csv","fallback"),"new/delivery/appendix_B_current.csv")
            self.assertEqual((old/"appendix_B_current.csv").read_text(),'historical')


class DocumentaryScopeTests(unittest.TestCase):
    def appendix_inputs(self):
        fields=["id","requirement","status","evidence","remaining_limit"]
        result=[]
        for folder,completed in [("inputs",14),("corrections/delivery",9)]:
            rows=[];used=0
            for number in range(1,22):
                state="no_aplica" if number in (14,15) else "completo" if used<completed else "parcial"
                if state=="completo":used+=1
                rows.append(dict(zip(fields,[f"B{number:02}","literal",""+state,"source","bounded scope"])))
            counts={state:sum(r["status"]==state for r in rows) for state in ["completo","parcial","no_aplica","no_cumple"]}
            result.extend([{"kind":"json_files","data":{"items":rows,"counts":counts},"file":Path(folder)/"appendix_B_current.json","path":folder+"/appendix_B_current.json"},
                           {"kind":"tables","fields":fields,"rows":[dict(r) for r in rows],"file":Path(folder)/"appendix_B_current.csv","path":folder+"/appendix_B_current.csv"}])
        return result

    def test_two_appendix_generations_validate_only_against_their_own_csv(self):
        review=reports.Review();reports.validate_appendix(self.appendix_inputs(),review)
        self.assertTrue(review.passed)
        pairs=[r for r in review.checks if r["control"]=="CSV y JSON del Apéndice B"]
        self.assertEqual(len(pairs),2)
        self.assertTrue(all(r["correcto"] for r in pairs))

    def test_mismatch_within_either_appendix_generation_is_rejected(self):
        for index in (1,3):
            with self.subTest(table=index):
                inputs=self.appendix_inputs();inputs[index]["rows"][0]["status"]="parcial"
                review=reports.Review();reports.validate_appendix(inputs,review)
                self.assertFalse(review.passed)
                failures=[r for r in review.checks if not r["correcto"]]
                self.assertEqual([r["fuente"] for r in failures],[inputs[index]["path"]])

    def test_presentation_checker_accepts_v2_and_v3_counts_without_claiming_oral_event(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"notas_orador.md").write_text('prepared, not performed')
            for count in (14,16):
                (root/"presentacion.html").write_text('<section class="slide">'*count)
                checker.validate_presentation(root,{"slides_prepared":count,"oral_defense_performed":False})
            with self.assertRaises(ValueError):
                checker.validate_presentation(root,{"slides_prepared":14,"oral_defense_performed":False})
            with self.assertRaises(ValueError):
                checker.validate_presentation(root,{"slides_prepared":16,"oral_defense_performed":True})

    def test_new81_predictor_study_does_not_inherit_old78_projection_row_count(self):
        path="scientific_closure/cpu_fixed_recipe_reproduction/result_v1/predictions_train.csv"
        item={"path":path,"kind":"tables","rows":[{"model":"fixture"}]*81,"fields":["model"]}
        common={"corrections_are_posthoc":True,"original_A2_replaced":False,"full_original_training_replay_claimed":False}
        current={**common,"corrections":[{"area":"cpu_fixed_recipe_reproduction","generation":"20261116","files":[{"path":path}]}]}
        review=reports.Review();reports.validate_corrections(current,[item],review)
        self.assertTrue(review.passed)
        legacy={**common,"corrections":[{"area":"frozen_inference","generation":"20261115","files":[{"path":path}]}]}
        review=reports.Review();reports.validate_corrections(legacy,[item],review)
        self.assertFalse(review.passed)

    def test_control_wrapper_import_needs_no_scientific_packages_or_execution(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"run.py";path.write_text(package.CONTROL_REPLAY_WRAPPER)
            spec=importlib.util.spec_from_file_location("_control_wrapper_contract",path)
            value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
            self.assertTrue(value.same(None,None))
            self.assertFalse(value.same(None,0.))
            self.assertFalse(value.same(.5,.6))
            self.assertNotIn('np',value.__dict__)
            self.assertFalse((Path(d)/'control_replay_report.json').exists())

    def test_adverse_scientific_result_is_preserved_while_integrity_can_pass(self):
        item={"path":"scientific_closure/controls/summary.json","kind":"json_files", "data":{"status":"FAIL","new_model_fits":13}}
        config={"corrections_are_posthoc":True,"original_A2_replaced":False,"full_original_training_replay_claimed":False,
                "corrections":[{"area":"controls","generation":"20261116","files":[{"path":item["path"]}]}]}
        review=reports.Review();reports.validate_corrections(config,[item],review)
        self.assertTrue(review.passed)
        self.assertIn('FAIL',json.dumps(review.summary))
        self.assertIn('20261116',json.dumps(review.summary))

    def test_current_replay_comparison_keeps_categorical_and_numeric_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runner=root/"run.py";runner.write_text(package.SCIENTIFIC_REPLAY_WRAPPER)
            spec=importlib.util.spec_from_file_location("_current_wrapper_test",runner);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
            a=root/"a.csv";b=root/"b.csv";a.write_text('model,p\nA,0.5\n');b.write_text(a.read_text())
            self.assertEqual(value.compare(a,b)["rows"],1)
            b.write_text('model,p\nA,0.6\n')
            with self.assertRaises(ValueError):value.compare(a,b)
            b.write_text('model,p\nB,0.5\n')
            with self.assertRaises(ValueError):value.compare(a,b)

    def test_nonembedded_current_trace_is_exported_with_generation_and_false_flags(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"scientific_closure/cpu_test_iut";source.mkdir(parents=True)
            (source/"summary.json").write_text('{"full_CPU_IUT_executed":true,"joint_CPU_improvements_Holm10":0,"original_parity_gate_passed":false}')
            (source/"trace.csv").write_text('value\n1\n')
            files=[{"path":p.relative_to(root).as_posix(),"render_in_report":p.name!='trace.csv'} for p in source.iterdir()]
            out=root/"output";out.mkdir()
            with patch.object(analysis,"BASE",root):records,html=analysis.recorded_corrections({"corrections":[{"area":"cpu_test_iut","generation":"20261116","files":files}]},out)
            self.assertTrue((out/"export/scientific_closure/cpu_test_iut/trace.csv").is_file())
            self.assertEqual(records[0]["generation"],"20261116")
            self.assertIn('joint_CPU_improvements_Holm10',html)
            self.assertIn('false',html)


if __name__=="__main__":unittest.main()
