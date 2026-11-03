"""Reserve the agreed raw-test archive identities from acquisition metadata only.

No ZIP, market series, supervised test sample or model is opened by this script.
The review date is a documentary version label, not a new acquisition timestamp.
"""
from datetime import date
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json")
NAME = "test_raw_reserve_20250101_20260831.json"


def main():
    payload = (ROOT / SOURCE).read_bytes()
    manifest = json.loads(payload)
    archives = []
    for item in manifest["archives"]:
        first = date.fromisoformat(item["first_open_utc"][:10])
        last = date.fromisoformat(item["last_open_utc"][:10])
        if item["kind"] == "monthly" and first >= date(2025, 1, 1) and last <= date(2026, 8, 31):
            assert item["status"] == "verified"
            archives.append({key: item[key] for key in (
                "archive", "url", "checksum_url", "sha256", "bytes", "csv_rows",
                "first_open_utc", "last_open_utc", "relative_path")})
    assert len(archives) == 20
    assert archives[0]["first_open_utc"].startswith("2025-01-01")
    assert archives[-1]["last_open_utc"].startswith("2026-08-31")
    assert len({item["archive"] for item in archives}) == 20
    record = {
        "schema_version": 1,
        "status": "raw_test_reserve_manifest",
        "review_date": "2026-11-03",
        "source": "Binance Vision / Binance Spot BTCUSDT",
        "frequency": "1m",
        "test_start_day_utc": "2025-01-01",
        "test_end_day_utc_inclusive": "2026-08-31",
        "calendar_days": 608,
        "source_manifest": SOURCE.as_posix(),
        "source_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "construction": "Selection from the existing verified acquisition manifest only; no raw test archive was opened to create this file.",
        "supervised_sample_manifest_frozen": False,
        "scope": "Raw archive identity and planned calendar are reserved; sample eligibility, transformations and experimental execution remain separate E2 prerequisites.",
        "raw_test_archives_opened_in_this_review": 0,
        "archive_count": len(archives),
        "csv_rows_reported_by_acquisition_manifest": sum(item["csv_rows"] for item in archives),
        "archives": archives,
    }
    serialized = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    for directory in (ROOT / "data/manifests", ROOT / "book/_static/data"):
        (directory / NAME).write_text(serialized)
    print(f"Reserved {len(archives)} raw archive identities; zero test ZIP opened.")


if __name__ == "__main__":
    main()
