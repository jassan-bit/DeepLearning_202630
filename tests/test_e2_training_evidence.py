"""Bind published integration evidence to source bytes and durable replay order."""
from pathlib import Path
import gzip,hashlib,json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_real_executor as dated
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"book_e2/_static/training_integration_20261109"


class TrainingEvidenceTests(unittest.TestCase):
    def test_source_and_compressed_receipts_match_the_published_snapshot(self):
        summary=json.loads((DATA/"summary.json").read_text())
        for path,sha in summary["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),sha,path)
        for name,item in summary["artifacts"].items():
            data=(DATA/name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(),item["sha256"])
            self.assertEqual(hashlib.sha256(gzip.decompress(data)).hexdigest(),item["uncompressed_sha256"])
        full=json.loads(gzip.decompress((DATA/"integration_full.json.gz").read_bytes()))
        rows=[r for r in full["training_runs"] if r["status"]=="completed"]
        self.assertEqual(len(rows),full["fitted_neural_runs"])
        self.assertEqual(len({r["run_id"] for r in rows}),len(rows))
        self.assertEqual(sum(r["resource_use"]["optimization_stages"] for r in rows),full["optimization_stages"])
        self.assertEqual(sum(r["resource_use"]["optimizer_steps"] for r in rows),full["optimizer_steps"])
        self.assertEqual(full["checkpoints_safely_reloaded"],full["checkpoint_count"])
        self.assertFalse(full["market_data_read"])
        self.assertFalse(full["full_protocol_search_executed"])

    def test_outer_observations_follow_completed_prediction_records_for_all_systems(self):
        rows=[json.loads(line) for line in gzip.decompress((DATA/"outer_events.jsonl.gz").read_bytes()).splitlines()]
        previous="0"*64
        predictions={}
        for index,row in enumerate(rows):
            token=row.pop("event_sha256")
            self.assertEqual(row["sequence"],index)
            self.assertEqual(row["previous_sha256"],previous)
            self.assertEqual(dated.canonical_hash(row),token)
            if row["kind"]=="predictions":
                self.assertEqual(len(row["systems"]),18)
                predictions[row["date"]]=token
            if row["kind"]=="observation":
                self.assertEqual(row["prediction_event_sha256"],predictions[row["date"]])
            previous=token
        self.assertEqual(len(predictions),3)


if __name__=="__main__":unittest.main()

