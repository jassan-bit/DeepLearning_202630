from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent/'engine_data_v2.py'
s=P.read_text()
s=s.replace('import base64\n','import base64\nimport calendar\n',1)
s=s.replace('daily-crypto-markets-base-volume-v2','crypto-markets-bounded-archives-base-volume-v2')
a=s.index('def _binance_url(')
b=s.index('\ndef _atomic_new(',a)
s=s[:a]+'''def binance_archive_plan(config):
    """Metadata-only archive itinerary, bounded before observing any prices.

    Complete calendar months inside the development range use monthly 1d ZIPs.
    Partial edge months use daily 1d ZIPs, so acquisition never fetches bars
    before/after the approved range. Missing archives halt; no source substitution
    or selection by model performance. A legacy daily-only policy is explicit.
    """
    provider, symbol = config["provider"], config["instrument"]
    require(provider in BINANCE_MARKETS and symbol in BINANCE_MARKETS[provider]["symbols"],
            "Unfrozen Binance provider/instrument")
    first, last = iso_date(config["development_start"]), iso_date(config["development_end"])
    require(first <= last, "Inverted acquisition bounds")
    policy = config.get("archive_policy", "bounded_monthly_with_daily_edges_v1")
    require(policy in ("bounded_monthly_with_daily_edges_v1", "daily_only_v1"), "Unknown archive policy")
    prefix = BINANCE_MARKETS[provider]["prefix"]
    plan, current = [], first
+    while current <= last:
+        month_end = current.replace(day=calendar.monthrange(current.year,current.month)[1])
+        monthly = policy == "bounded_monthly_with_daily_edges_v1" and current.day == 1 and month_end <= last
+        end = month_end if monthly else current
+        frequency = "monthly" if monthly else "daily"
+        suffix = current.strftime("%Y-%m") if monthly else current.isoformat()
+        name = f"{symbol}-1d-{suffix}.zip"
+        plan.append({"path":name,"url":f"https://data.binance.vision/data/{prefix}/{frequency}/klines/{symbol}/1d/{name}",
+                     "archive_frequency":frequency,"first_session":current.isoformat(),"last_session":end.isoformat(),
+                     "expected_bars":(end-current).days+1})
+        current = end+dt.timedelta(days=1)
+    return plan
+
+
+def _validate_binance_archive(raw, item, provider, acquired_utc):
+    bars = parse_binance_zip(raw, acquired_utc=acquired_utc,market_type=BINANCE_MARKETS[provider]["market_type"])
+    require(bars["dates"].tolist() == expected_sessions(item["first_session"],item["last_session"],"24/7"),
+            "Archive contains unexpected/missing/out-of-range sessions")
+    return bars
+
+''' .replace('\n+','\n')+s[b:]
a=s.index('    allowed = {"acquisition_pending.json"',s.index('def inspect_binance_pending('))
b=s.index('\n\ndef _collect_binance_daily(',a)
s=s[:a]+'''    allowed = {"acquisition_pending.json", "acquisition_attempts.jsonl", "raw_manifest.json", "reacquisition_events.jsonl"}
+    complete, incomplete = 0, 0
+    plan = binance_archive_plan(config)
+    for item in plan:
+        name = item["path"]
+        archive, checksum = root/name, root/(name+".CHECKSUM")
+        allowed.update((name,name+".CHECKSUM"))
+        require(not archive.is_symlink() and not checksum.is_symlink(), "Acquisition source symlink refused")
+        if checksum.exists():
+            tokens = checksum.read_text().split()
+            require(len(tokens) >= 1 and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
+                    "Invalid stored provider checksum")
+            if archive.exists():
+                raw = archive.read_bytes()
+                require(sha256_bytes(raw) == tokens[0].lower(), "Partial stored source checksum mismatch")
+                _validate_binance_archive(raw,item,provider,pending["acquisition_started_utc"])
+                complete += 1
+        if archive.exists() != checksum.exists():
+            incomplete += 1
+    require(all(p.name in allowed and p.is_file() for p in root.iterdir()),
+            "Unknown acquisition files retained; do not silently discard or overwrite")
+    return {"status": "VALIDATED_PARTIAL_RAW_NOT_ADMITTED", "complete_pairs": complete,
+            "incomplete_pairs": incomplete, "expected_pairs": len(plan),
+            "expected_raw_bars":sum(item["expected_bars"] for item in plan),
+            "acquisition_started_utc": pending["acquisition_started_utc"]}
+''' .replace('\n+','\n')+s[b:]
a=s.index('    downloaded = reused = 0',s.index('def _collect_binance_daily('))
b=s.index('    manifest["completed_utc"] = utc_now()',a)
s=s[:a]+'''    manifest["archive_policy"] = config.get("archive_policy", "bounded_monthly_with_daily_edges_v1")
+    downloaded = reused = 0
+    plan = binance_archive_plan(config)
+    for offset, item in enumerate(plan):
+        name, url = item["path"], item["url"]
+        checksum_path, archive_path = root/(name+".CHECKSUM"), root/name
+        checksum = checksum_path.read_bytes() if checksum_path.exists() else _network_get(url+".CHECKSUM")
+        tokens = checksum.decode().split()
+        require(tokens and len(tokens[0]) == 64 and all(x in "0123456789abcdefABCDEF" for x in tokens[0]),
+                "Invalid upstream Binance checksum")
+        claimed = tokens[0].lower()
+        if archive_path.exists():
+            payload = archive_path.read_bytes(); reused += 1
+        else:
+            payload = _network_get(url); downloaded += 1
+        require(sha256_bytes(payload) == claimed, "Upstream Binance checksum mismatch")
+        _validate_binance_archive(payload,item,provider,utc_now())
+        _atomic_new(archive_path,payload); _atomic_new(checksum_path,checksum)
+        manifest["files"].append({**item,"sha256":claimed,
+            "provider_checksum_sha256":sha256_bytes(checksum)})
+        if (offset+1) % 12 == 0 or offset+1 == len(plan):
+            print(json.dumps({"event":"development_acquisition_progress","provider":provider,"instrument":symbol,
+                "completed_files":offset+1,"expected_files":len(plan),"downloaded_payloads":downloaded,
+                "reused_payloads":reused,"new_market_fits":0,"test_opened":False}),flush=True)
+''' .replace('\n+','\n')+s[b:]
old='''        expected_prefix = "/data/"+BINANCE_MARKETS[provider]["prefix"]+"/daily/klines/"
        require(parsed.path.startswith(expected_prefix) and not parsed.query,
                "Binance recovery URL market or frequency differs from its provider")'''
new='''        root_prefix = "/data/"+BINANCE_MARKETS[provider]["prefix"]
        require(any(parsed.path.startswith(root_prefix+"/"+frequency+"/klines/") for frequency in ("daily","monthly")) and not parsed.query,
                "Binance recovery URL market or frequency differs from its provider")'''
assert old in s
s=s.replace(old,new)
s=s.replace('BINANCE spot/USD-M/COIN-M base-volume decoding', 'Binance spot/USD-M/COIN-M base-volume decoding')
compile(s,'engine_data_v2.py','exec')
P.write_text(s)
prov=P.parent/'DATA_EXTENSION_PROVENANCE.json'
o=json.loads(prov.read_text());o['extension_sha256']=hashlib.sha256(P.read_bytes()).hexdigest();o['version']='crypto-markets-bounded-archives-base-volume-v2';o['changes'].extend(['bounded complete monthly archives with daily edges; no out-of-range bars acquired','metadata-only deterministic archive plan before prices; exact month/day recovery'])
prov.write_text(json.dumps(o,indent=2)+'\n')
print(o['extension_sha256'])
