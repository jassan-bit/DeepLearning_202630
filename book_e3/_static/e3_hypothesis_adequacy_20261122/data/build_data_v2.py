"""Create a versioned data extension; never edits frozen source A."""
from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
candidates=[Path(os.environ['E3_SOURCE_A_DATA'])] if os.environ.get('E3_SOURCE_A_DATA') else []
if len(HERE.parents)>2:
    candidates.append(HERE.parents[2]/'experiments/e3_confirmatory_engine_20261121/engine/data.py')
candidates.append(Path('/workspace/e3_confirmatory_engine_20261121/engine/data.py'))
BASE=next((path for path in candidates if path.is_file()),None)
if BASE is None:raise RuntimeError('Set E3_SOURCE_A_DATA to the verified source-A engine/data.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='3851281f00fc7b43bfb681d10be988f0efc5fb3d879e6b4faab2fe4798a9d917'
s=BASE.read_text()
before=hashlib.sha256(BASE.read_bytes()).hexdigest()
start=s.index('def parse_binance_zip(')
end=s.index('\ndef expected_sessions(',start)
s=s[:start]+'''BINANCE_MARKETS = {
    "binance_vision": {"market_type": "spot", "prefix": "spot", "symbols": ("BTCUSDT",)},
    "binance_vision_um": {"market_type": "um", "prefix": "futures/um", "symbols": ("BTCUSDT", "ETHUSDT")},
    "binance_vision_cm": {"market_type": "cm", "prefix": "futures/cm", "symbols": ("BTCUSD_PERP",)},
}
BINANCE_DATA_ADAPTER_VERSION = "daily-crypto-markets-base-volume-v2"


def parse_binance_zip(raw, *, acquired_utc=None, market_type="spot"):
    """Daily close-price klines, with source-specific base-asset volume.

    Source: Binance public-data README pinned in qa/data/primary. USD-M and
    spot volume is column5; COIN-M column5 counts contracts, so column7 supplies
    base volume. Funding payments are not included in the close-price target.
    """
    require(market_type in ("spot", "um", "cm"), "Unsupported Binance market type")
    acquired = dt.datetime.fromisoformat(acquired_utc) if acquired_utc else dt.datetime.now(UTC)
    require(acquired.tzinfo is not None, "Acquisition timestamp must include UTC offset")
    rows, units = [], set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [p for p in archive.namelist() if p.endswith(".csv")]
        require(len(members) == 1, "Expected exactly one CSV in Binance archive")
        for index, r in enumerate(csv.reader(io.StringIO(archive.read(members[0]).decode("utf-8-sig")))):
            require(len(r) == 12, "Unexpected Binance kline schema")
            # Published futures archives can carry a header. Skip only a
            # recognized first-line schema; never discard malformed price rows.
            if index == 0 and r[0].strip().lower().replace(" ", "_") == "open_time":
                names = [x.strip().lower().replace(" ", "_") for x in r]
                require(names[:7] == ["open_time","open","high","low","close","volume","close_time"],
                        "Unexpected Binance header order")
                expected_volume = ({"base_volume","base_asset_volume"} if market_type == "cm" else
                                   {"quote_volume","quote_asset_volume"})
                require(names[7] in expected_volume, "Wrong market-volume header")
                continue
            start, close = int(r[0]), int(r[6])
            unit = 1_000_000 if start >= 100_000_000_000_000 else 1_000
            require(start >= 1_000_000_000_000, "Expected epoch ms/us, not seconds")
            require(market_type == "spot" or unit == 1_000, "Futures timestamp unit changed; review source before admission")
            units.add(unit)
            stamp = dt.datetime.fromtimestamp(start / unit, UTC)
            require(stamp.time() == dt.time(), "Daily Binance bar must start at UTC midnight")
            require(close == start + 86400 * unit - 1, "Wrong daily close time/unit")
            require(dt.datetime.fromtimestamp((close + 1) / unit, UTC) <= acquired,
                    "Incomplete/unclosed daily bar")
            volume = float(r[7] if market_type == "cm" else r[5])
            require(np.isfinite(volume) and volume >= 0, "Invalid base-asset volume")
            if market_type == "cm":
                contracts = float(r[5])
                require(np.isfinite(contracts) and contracts >= 0, "Invalid COIN-M contract count")
            rows.append((stamp.date().isoformat(), *(float(x) for x in r[1:5]), volume, 1.))
    return _rows_to_bars(rows, "binance_vision" if market_type == "spot" else "binance_vision_"+market_type,
        {"epoch_units_per_second": sorted(units), "timezone": "UTC", "market_type": market_type,
         "close": "next_UTC_midnight_minus_one_tick", "volume_units": "base_asset",
         "volume_column_zero_based": 7 if market_type == "cm" else 5,
         "parser_version": BINANCE_DATA_ADAPTER_VERSION})


def _binance_url(provider, symbol, date):
    require(provider in BINANCE_MARKETS and symbol in BINANCE_MARKETS[provider]["symbols"],
            "Unfrozen Binance provider/instrument")
    iso_date(date)
    name = f"{symbol}-1d-{date}.zip"
    prefix = BINANCE_MARKETS[provider]["prefix"]
    return name, f"https://data.binance.vision/data/{prefix}/daily/klines/{symbol}/1d/{name}"


def _atomic_new(path, raw):
    """Install immutable bytes, allowing exact reuse after interruption."""
    path = Path(path)
    require(not path.is_symlink(), "Acquisition symlink refused")
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, "Existing acquisition bytes differ; preserve and investigate")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        # There is one foreground collector. Refuse a raced different write.
        require(not path.exists(), "Acquisition target appeared during write")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pending_record(config, rights_receipt, exposure_receipt, timestamp):
    result = {"kind": "binance_daily_partial_acquisition_v2", "adapter_version": BINANCE_DATA_ADAPTER_VERSION,
              "config": config, "config_sha256": identity(config),
              "rights_sha256": identity(rights_receipt), "exposure_sha256": identity(exposure_receipt),
              "acquisition_started_utc": timestamp}
    result["pending_sha256"] = identity(result)
    return result


def inspect_binance_pending(config, output_dir, *, rights_receipt, exposure_receipt):
    """Pure read-only validation of interrupted raw acquisition, before retry.

    Does not download, write, or mark partial bytes as admitted. Missing ZIP or
    checksum companions may be completed by the activated collector; every
    present pair is hash-verified before any retry contacts the provider.
    """
    verify_evidence(rights_receipt, "approved"); verify_evidence(exposure_receipt, "audited")
    provider, symbol = config["provider"], config["instrument"]
    require(provider in BINANCE_MARKETS, "Pending inspection requires a Binance provider")
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    require(first <= last, "Invalid pending development range")
    root = Path(output_dir)
    target = root/"acquisition_pending.json"
    require(target.is_file() and not target.is_symlink(), "No authentic pending acquisition metadata; preserve partial directory")
    pending = json.loads(target.read_text())
    claimed = pending.pop("pending_sha256", None)
    require(claimed == identity(pending), "Pending metadata hash mismatch")
    expected = _pending_record(config, rights_receipt, exposure_receipt, pending.get("acquisition_started_utc"))
    expected.pop("pending_sha256")
    require(pending == expected, "Pending config/evidence/adapter differs; do not reuse under a new campaign")
    stamp = dt.datetime.fromisoformat(pending["acquisition_started_utc"])
    require(stamp.tzinfo is not None, "Pending acquisition timestamp missing timezone")
    allowed = {"acquisition_pending.json", "acquisition_attempts.jsonl", "raw_manifest.json", "reacquisition_events.jsonl"}
    complete, incomplete = 0, 0
    for offset in range((last-first).days+1):
        date = (first+dt.timedelta(days=offset)).isoformat()
        name, _ = _binance_url(provider,symbol,date)
        archive, checksum = root/name, root/(name+".CHECKSUM")
        allowed.update((name,name+".CHECKSUM"))
        require(not archive.is_symlink() and not checksum.is_symlink(), "Acquisition source symlink refused")
        if checksum.exists():
            tokens = checksum.read_text().split()
            require(len(tokens) >= 1 and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                    "Invalid stored provider checksum")
            if archive.exists():
                raw = archive.read_bytes()
                require(sha256_bytes(raw) == tokens[0].lower(), "Partial stored source checksum mismatch")
                bars = parse_binance_zip(raw, acquired_utc=pending["acquisition_started_utc"],
                                         market_type=BINANCE_MARKETS[provider]["market_type"])
                require(bars["dates"].tolist() == [date], "Daily archive contains a wrong date or count")
                complete += 1
        if archive.exists() != checksum.exists():
            incomplete += 1
    require(all(p.name in allowed and p.is_file() for p in root.iterdir()),
            "Unknown acquisition files retained; do not silently discard or overwrite")
    return {"status": "VALIDATED_PARTIAL_RAW_NOT_ADMITTED", "complete_pairs": complete,
            "incomplete_pairs": incomplete, "expected_pairs": (last-first).days+1,
            "acquisition_started_utc": pending["acquisition_started_utc"]}


def _collect_binance_daily(config, root, rights, exposure):
    provider, symbol = config["provider"], config["instrument"]
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    pending_path = root/"acquisition_pending.json"
    if not pending_path.exists():
        require(all(p.name == "acquisition_attempts.jsonl" and p.is_file() for p in root.iterdir()),
                "Unbound pre-existing acquisition files retained; use original recovery metadata")
        pending = _pending_record(config, rights, exposure, utc_now())
        _atomic_new(pending_path, (json.dumps(pending,indent=2)+"\\n").encode())
    state = inspect_binance_pending(config,root,rights_receipt=rights,exposure_receipt=exposure)
    manifest = {"provider":provider,"instrument":symbol,"scope":"development_only","files":[],
        "rights":rights,"exposure":exposure,"acquired_utc":state["acquisition_started_utc"],
        "acquisition_started_utc":state["acquisition_started_utc"],
        "acquired_utc_semantics":"First acquisition attempt; completed_utc and attempt ledgers report later actual events.",
        "market_type":BINANCE_MARKETS[provider]["market_type"],"adapter_version":BINANCE_DATA_ADAPTER_VERSION}
    downloaded = reused = 0
    total = (last-first).days+1
    for offset in range(total):
        date = (first+dt.timedelta(days=offset)).isoformat()
        name, url = _binance_url(provider,symbol,date)
        checksum_path, archive_path = root/(name+".CHECKSUM"), root/name
        checksum = checksum_path.read_bytes() if checksum_path.exists() else _network_get(url+".CHECKSUM")
        tokens = checksum.decode().split()
        require(tokens and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
                "Invalid upstream Binance checksum")
        claimed = tokens[0].lower()
        if archive_path.exists():
            payload = archive_path.read_bytes(); reused += 1
        else:
            payload = _network_get(url); downloaded += 1
        require(sha256_bytes(payload) == claimed, "Upstream Binance checksum mismatch")
        bars = parse_binance_zip(payload,acquired_utc=utc_now(),market_type=manifest["market_type"])
        require(bars["dates"].tolist() == [date], "Daily archive contains a wrong date or count")
        _atomic_new(archive_path,payload); _atomic_new(checksum_path,checksum)
        manifest["files"].append({"path":name,"url":url,"sha256":claimed,
            "provider_checksum_sha256":sha256_bytes(checksum)})
        if (offset+1) % 30 == 0 or offset+1 == total:
            print(json.dumps({"event":"development_acquisition_progress","provider":provider,"instrument":symbol,
                "completed_files":offset+1,"expected_files":total,"downloaded_payloads":downloaded,
                "reused_payloads":reused,"new_market_fits":0,"test_opened":False}),flush=True)
    manifest["completed_utc"] = utc_now()
    manifest["manifest_sha256"] = identity(manifest)
    _atomic_new(root/"raw_manifest.json",(json.dumps(manifest,indent=2)+"\\n").encode())
    return manifest


def _parse_binance_manifest_payload(raw, record):
    provider = record["provider"]
    expected = BINANCE_MARKETS[provider]["market_type"]
    require(record.get("market_type",expected) == expected, "Manifest provider/market type mismatch")
    return parse_binance_zip(raw,acquired_utc=record["acquired_utc"],market_type=expected)

''' + s[end:]
# Replace exactly one old spot collection branch; other providers untouched.
a=s.index('    if provider == "binance_vision":',s.index('def _collect_development('))
b=s.index('    elif provider == "nasdaq_wiki":',a)
s=s[:a]+'''    if provider in BINANCE_MARKETS:
        return _collect_binance_daily(config,root,rights_receipt,exposure_receipt)
''' + s[b:]
# Both preparation routes dispatch by the explicit provider identity.
old='''        if record["provider"] == "binance_vision":
            parsed.append(parse_binance_zip(payload, acquired_utc=record["acquired_utc"]))'''
new='''        if record["provider"] in BINANCE_MARKETS:
            parsed.append(_parse_binance_manifest_payload(payload,record))'''
assert s.count(old)==2,s.count(old)
s=s.replace(old,new)
# Metadata-only recovery recognizes providers, never aliases them to spot.
s=s.replace('hosts = {"binance_vision": "data.binance.vision", "nasdaq_wiki": "data.nasdaq.com",',
'''hosts = {"binance_vision": "data.binance.vision", "binance_vision_um": "data.binance.vision",
             "binance_vision_cm": "data.binance.vision", "nasdaq_wiki": "data.nasdaq.com",''')
old='''    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    key_name ='''
new='''    if provider in BINANCE_MARKETS:
        expected_prefix = "/data/"+BINANCE_MARKETS[provider]["prefix"]+"/daily/klines/"
        require(parsed.path.startswith(expected_prefix) and not parsed.query,
                "Binance recovery URL market or frequency differs from its provider")
    params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    key_name ='''
assert s.count(old)==1
s=s.replace(old,new)
old='''    attempt_log = manifest_path.parent/"acquisition_attempts.jsonl"
    if attempt_log.is_file():
        item = _encoded_metadata(attempt_log); item["relative_path"] = attempt_log.name
        companions.append(item)'''
new='''    for name in ("acquisition_attempts.jsonl", "acquisition_pending.json"):
        companion = manifest_path.parent/name
        if companion.is_file():
            item = _encoded_metadata(companion); item["relative_path"] = companion.name
            companions.append(item)'''
assert old in s
s=s.replace(old,new)
# Provenance records explicit adapter identity, without altering target/windows.
old='''        "split_config_sha256": identity(split_config)}, rights=rights, exposure=exposure,'''
new='''        "split_config_sha256": identity(split_config),
        **({"market_type":BINANCE_MARKETS[record["provider"]]["market_type"],
            "volume_units":"base_asset", "data_adapter_version":BINANCE_DATA_ADAPTER_VERSION}
           if record["provider"] in BINANCE_MARKETS else {})}, rights=rights, exposure=exposure,'''
assert s.count(old)==1
s=s.replace(old,new)
old='''        "split_config_sha256": identity(split_config), "protocol_sha256": protocol_hash}, rights=rights, exposure=exposure,'''
new='''        "split_config_sha256": identity(split_config), "protocol_sha256": protocol_hash,
        **({"market_type":BINANCE_MARKETS[record["provider"]]["market_type"],
            "volume_units":"base_asset", "data_adapter_version":BINANCE_DATA_ADAPTER_VERSION}
           if record["provider"] in BINANCE_MARKETS else {})}, rights=rights, exposure=exposure,'''
assert s.count(old)==1
s=s.replace(old,new)
s=s.replace('Market acquisition is opt-in; no market data are distributed with this package.',
'''Market acquisition is opt-in; no market data are distributed with this package.
Versioned extension of source A: Binance spot/USD-M/COIN-M base-volume decoding,
immutable resumable daily acquisition, and exact recovery. AI assistance declared.
No changes to the model, causal target, splits, statistical methods or seed bank.''')
compile(s,'engine_data_v2.py','exec')
(HERE/'engine_data_v2.py').write_text(s)
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==before
(HERE/'DATA_EXTENSION_PROVENANCE.json').write_text(json.dumps({'sourceA_file':'engine/data.py','sourceA_sha256':before,'extension_file':'engine_data_v2.py','extension_sha256':hashlib.sha256(s.encode()).hexdigest(),'version':'daily-crypto-markets-base-volume-v2','changes':['explicit provider identity for spot/USD-M/COIN-M','COIN-M base-volume column7, USD-M/spot column5','recognized first-line CSV headers','resumable daily acquisition with pending config/evidence identity and immutable source pairs','prepare/recovery/final parse by explicit market provider','recovery URL market binding and pending event snapshot'],'unchanged':['causal target/window/support constructors','model code and identity','seeds/statistical methods','owner evidence and final protocol requirements'],'market_downloads':0,'AI_assistance_declared':True},indent=2)+'\n')
print(json.dumps({'base_unchanged':True,'extension_sha256':hashlib.sha256(s.encode()).hexdigest(),'bytes':len(s.encode())}))
