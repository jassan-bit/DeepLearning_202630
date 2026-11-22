"""Apply reviewed v2.1 fixes to generated v2; never edits source A."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'engine_data_v2.py'
s=P.read_text()
s=s.replace('import os\n','import os\nimport re\n',1)
s=s.replace('crypto-markets-bounded-archives-base-volume-v2','crypto-markets-bounded-archives-base-volume-v2.1')
pos=s.index('\ndef _atomic_new(')
s=s[:pos]+'''

def _canonical_binance_manifest_item(provider, symbol, row):
    require(provider in BINANCE_MARKETS and symbol in BINANCE_MARKETS[provider]["symbols"],
            "Manifest has unfrozen Binance provider/instrument")
    name = row.get("path")
    require(isinstance(name,str) and re.fullmatch(re.escape(symbol)+r"-1d-\\d{4}-\\d{2}(?:-\\d{2})?\\.zip",name),
            "Binance manifest instrument/filename differs")
    suffix = name[len(symbol)+4:-4]
    try:
        if len(suffix)==7:
            first = iso_date(suffix+"-01")
            last = first.replace(day=calendar.monthrange(first.year,first.month)[1])
            frequency = "monthly"
        else:
            first = last = iso_date(suffix)
            frequency = "daily"
    except (ValueError,TypeError):
        raise AdmissionError("Malformed Binance archive period") from None
    prefix = BINANCE_MARKETS[provider]["prefix"]
    expected = {"path":name,"url":f"https://data.binance.vision/data/{prefix}/{frequency}/klines/{symbol}/1d/{name}",
        "archive_frequency":frequency,"first_session":first.isoformat(),"last_session":last.isoformat(),
        "expected_bars":(last-first).days+1}
    require(row.get("url") == expected["url"], "Binance manifest provider/instrument/URL differs")
    for key in ("archive_frequency","first_session","last_session","expected_bars"):
        require(key not in row or row[key] == expected[key], "Binance manifest archive metadata differs: "+key)
    return expected


def _checked_binance_checksum(raw, name, expected_payload_sha=None):
    try:
        tokens = raw.decode("utf-8").split()
    except UnicodeError:
        raise AdmissionError("Invalid Binance checksum encoding") from None
    require(len(tokens)==2 and re.fullmatch(r"[0-9a-fA-F]{64}",tokens[0]) and tokens[1].lstrip("*")==name,
            "Binance checksum must bind the exact archive filename")
    claimed = tokens[0].lower()
    require(expected_payload_sha is None or claimed == expected_payload_sha,
            "Binance checksum payload hash differs")
    return claimed


def validate_binance_manifest(record, manifest_path=None):
    """Read-only canonical source binding, including source bytes if available.

    A self-hash alone does not authenticate a relabelled archive. Bind the
    provider, registered instrument, market namespace, 1d frequency, canonical
    archive filename, period and official URL. This is not external signature
    authentication: the original acquisition/recovery hashes remain necessary.
    Old source-A spot manifests keep their old shape and provenance.
    """
    require(isinstance(record,dict) and record.get("provider") in BINANCE_MARKETS,
            "Binance manifest provider required")
    provider, symbol = record["provider"], record.get("instrument")
    require(symbol in BINANCE_MARKETS[provider]["symbols"], "Unfrozen Binance manifest instrument")
    expected_market = BINANCE_MARKETS[provider]["market_type"]
    require(record.get("market_type",expected_market)==expected_market,
            "Binance manifest provider/market type differs")
    if "manifest_sha256" in record:
        require(record["manifest_sha256"] == identity({k:v for k,v in record.items() if k!="manifest_sha256"}),
                "Binance manifest identity differs")
    require(record.get("scope") in ("development_only","reserved_final"), "Binance manifest scope required")
    files = record.get("files")
    require(isinstance(files,list) and files, "Binance manifest needs source files")
    canonical = [_canonical_binance_manifest_item(provider,symbol,row) for row in files]
    require(len({r["path"] for r in canonical})==len(canonical), "Duplicate Binance archive path")
    ordered = sorted(canonical,key=lambda r:r["first_session"])
    require(all(a["last_session"] < b["first_session"] for a,b in zip(ordered,ordered[1:])),
            "Overlapping Binance archive periods")
    if record.get("acquisition_config") is not None:
        config=record["acquisition_config"]
        require(config.get("provider")==provider and config.get("instrument")==symbol,
                "Manifest source differs from acquisition config")
        require(canonical==binance_archive_plan(config), "Manifest archive plan differs from original acquisition config")
    for row in files:
        require(isinstance(row.get("sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",row["sha256"]),
                "Binance source SHA256 required")
        require(isinstance(row.get("provider_checksum_sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",row["provider_checksum_sha256"]),
                "Binance provider checksum companion SHA256 required")
        if manifest_path is not None:
            root=Path(manifest_path).parent
            path=_safe_path(root,row["path"])
            require(not path.is_symlink() and sha256_bytes(path.read_bytes())==row["sha256"],
                    "Binance source bytes differ from bound archive")
            checksum_path=_safe_path(root,row["path"]+".CHECKSUM")
            checksum=checksum_path.read_bytes()
            require(not checksum_path.is_symlink() and sha256_bytes(checksum)==row["provider_checksum_sha256"],
                    "Binance provider checksum companion bytes differ")
            _checked_binance_checksum(checksum,row["path"],row["sha256"])
    return {"status":"BINANCE_SOURCE_IDENTITY_VERIFIED","provider":provider,"instrument":symbol,
            "archives":len(canonical),"expected_raw_bars":sum(x["expected_bars"] for x in canonical),
            "source_bytes_verified":manifest_path is not None}
''' + s[pos:]
# Checksum sidecars must name the exact archive, including interrupted pairs.
old='''            tokens = checksum.read_text().split()
            require(len(tokens) >= 1 and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                    "Invalid stored provider checksum")
            if archive.exists():
                raw = archive.read_bytes()
                require(sha256_bytes(raw) == tokens[0].lower(), "Partial stored source checksum mismatch")'''
new='''            claimed = _checked_binance_checksum(checksum.read_bytes(),name)
            if archive.exists():
                raw = archive.read_bytes()
                require(sha256_bytes(raw) == claimed, "Partial stored source checksum mismatch")'''
assert old in s;s=s.replace(old,new)
old='''        tokens = checksum.decode().split()
        require(tokens and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                "Invalid upstream Binance checksum")
        claimed = tokens[0].lower()'''
assert old in s;s=s.replace(old,'''        claimed = _checked_binance_checksum(checksum,name)''')
# The config is itself hash-bound in pending metadata and then embedded once.
old='''    manifest["archive_policy"] = config.get("archive_policy", "bounded_monthly_with_daily_edges_v1")'''
s=s.replace(old,old+'\n    manifest["acquisition_config"] = config',1)
old='''    manifest["manifest_sha256"] = identity(manifest)
    _atomic_new(root/"raw_manifest.json",(json.dumps(manifest,indent=2)+"\\n").encode())'''
new='''    manifest["manifest_sha256"] = identity(manifest)
    validate_binance_manifest(manifest,root/"raw_manifest.json")
    _atomic_new(root/"raw_manifest.json",(json.dumps(manifest,indent=2)+"\\n").encode())'''
assert old in s;s=s.replace(old,new)
# Both preparation routes validate source identity before parsing labels.
old='''    require("test" not in split_config["raw_bounds"], "Development cannot prepare test")
    parsed = []'''
new='''    require("test" not in split_config["raw_bounds"], "Development cannot prepare test")
    if record["provider"] in BINANCE_MARKETS:
        validate_binance_manifest(record,manifest_path)
        require(record.get("rights")==rights and record.get("exposure")==exposure,
                "Development raw evidence differs from supplied evidence")
    parsed = []'''
assert old in s;s=s.replace(old,new)
old='''    parsed = []
    for row in record["files"]:
        payload = _safe_path(manifest_path.parent, row["path"]).read_bytes()
        require(sha256_bytes(payload) == row["sha256"], "Reserved raw byte hash mismatch")'''
new='''    if record["provider"] in BINANCE_MARKETS:
        validate_binance_manifest(record,manifest_path)
        require(record.get("rights")==rights and record.get("exposure")==exposure,
                "Reserved raw evidence differs from supplied evidence")
    parsed = []
    for row in record["files"]:
        payload = _safe_path(manifest_path.parent, row["path"]).read_bytes()
        require(sha256_bytes(payload) == row["sha256"], "Reserved raw byte hash mismatch")'''
assert old in s;s=s.replace(old,new)
# Preserve exact legacy source-A spot provenance on replay.
s=s.replace('''if record["provider"] in BINANCE_MARKETS else {})}, rights=rights, exposure=exposure,''',
'''if record["provider"] in BINANCE_MARKETS and record.get("adapter_version") else {})}, rights=rights, exposure=exposure,''')
# Validate original bundle's semantic identities before restoring metadata or network.
old='''    return bundle_raw, bundle, manifest_raw, original_raw, manifest, original'''
new='''    if manifest.get("provider") in BINANCE_MARKETS:
        validate_binance_manifest(manifest)
        require(original.get("provenance",{}).get("provider")==manifest["provider"] and
                original.get("provenance",{}).get("instrument")==manifest["instrument"],
                "Recovery manifest source differs from original admission")
    return bundle_raw, bundle, manifest_raw, original_raw, manifest, original'''
assert old in s;s=s.replace(old,new)
# CLI-only guard is owned by the Colab runtime package; no bypass or data reads.
old='''    args = parser.parse_args(argv)
    if args.command == "record-owner-evidence":'''
new='''    args = parser.parse_args(argv)
    if args.command == "prepare-final":
        from engine.confirmatory_gate import block_final_operation, ConfirmatoryGateError
        try:
            block_final_operation(Path(__file__).resolve().parents[1], "prepare-final-data")
        except ConfirmatoryGateError as exc:
            print(json.dumps({"status":"confirmation_not_enabled","operation":"prepare-final-data",
                "evaluation_executed":False,"R3_result":None,
                "gate_codes":getattr(exc,"gate_codes",[]),
                "invalid_metadata":getattr(exc,"invalid_metadata",None)}))
            raise SystemExit(2) from None
    if args.command == "record-owner-evidence":'''
assert old in s;s=s.replace(old,new)
compile(s,'engine_data_v2.py','exec');P.write_text(s)
prov=P.parent/'DATA_EXTENSION_PROVENANCE.json'
o=json.loads(prov.read_text());o['extension_sha256']=hashlib.sha256(P.read_bytes()).hexdigest();o['version']='crypto-markets-bounded-archives-base-volume-v2.1'
o['changes'].extend(['canonical semantic manifest source binding before prepare/final/recovery','checksum companion filename bound to archive','exact legacy source-A spot provenance retained during replay','CLI-only final operation blocked by packaged confirmatory_gate; low-level fixture functions unchanged'])
o['qa_corrections']=[{'counterexample':'self-hashed manifest relabelledBTCUSDT toETHUSDT while URLs and filenames remainedBTC','repair':'canonical provider/instrument/URL/file/date/frequency validator'}, {'counterexample':'source-A spot replay got new provenance fields','repair':'adapter-only provenance extras; original comparable check and original hashes preserved'}]
prov.write_text(json.dumps(o,indent=2)+'\n')
print(o['extension_sha256'])
