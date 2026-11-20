# Two separate archival products

Prepared with AI assistance. Base source is the immutable project commit
`8d03a73d529eb3f9b2e89992b4488cfeb88a99e6`; a newer book is a different version.

## Executed local checks

The pinned43-chunk index describes8582 final logical files/4268147881 bytes for
the193-delta additive audit and259 files/2359027 bytes for the V3 pilot. The
terminal additive delta contains35 files and is not a cumulative snapshot.
The overlay index supplies each final logical path's last storage commit/hash.
The index manifest SHA256 is
`c2f231d140ab95513be7c0b7462202fe23b624b94de2225f8532986b8d25f272`.

A local filename/size/SHA256 search hashed1832 candidates/185382026 bytes and
found69 additive logical files/8276940 bytes and33 pilot files/984531 bytes.
These are actual locally verified bytes, not complete hydration. The missing
additive and pilot payloads total4261245437 bytes. The all8841 paths refer to
8831 distinct SHA256 objects/4270503439 bytes, excluding stage metadata.

## Product1: ephemeral complete historical snapshot

The production `--hydrate` mode obtains all8841 logical payloads and386
receipt/storage documents through public immutable raw URLs, verifies length,
SHA256 and declared Git blob SHA1, then materializes logical paths using hard
links to the content-addressed cache. It executes no model and never unpickles
a checkpoint. Existing verified objects can be reused after interruption.

The193 receipt/storage pairs add4750100 bytes. A failed request, hash, size or
budget check prevents a complete outcome. Retry is bounded to transient HTTP
or transport failures. The selected byte identity is fixed by the pinned index;
no new endpoints, files or results are selected because of outcomes.

This snapshot includes historical inputs and unreviewed checkpoints. It stays
in `ephemeral-cas/` and `ephemeral-snapshot/`, outside `public/`. It must not be
uploaded as an artifact or called a persistent DOI deposit. Downloading and
hashing it does not establish rights or reproduce training.

## Product2: source/evidence capsule with explicit exclusions

The runner reads the complete source tar stream from GitHub codeload at the
base commit. Every regular source file is inventoried with source hash, size,
origin and include/exclude decision. Nonregular entries are recorded/excluded.
The source stream itself is not an uploaded nested repository tar.

The scoped capsule retains source/code, documentation/configuration/locks,
selected generated aggregate evidence and the15 exact previously published
ONNX hashes (14 historical representatives plus experimental V3). Original
component notices remain. No blanket repository MIT/CC BY grant is invented.
The owner decision in the0.2.0 documentary ZIP applies only to its selected
own members; it is not extended to all scientific inputs or checkpoints.

Excluded files include raw data/input trees, NumPy/NPZ input matrices,
unreviewed historical checkpoints, nested archives/full papers, unreviewed
binary resources, dependency trees and full prediction panels. Notebook code
cells are preserved but outputs/execution counts/attachments are removed and
both original and archived hashes are recorded. Secret-like patterns cause
exclusion for review. Aggregate CSVs with OHLC headers are excluded.

All omissions are enumerated in repository/historical inventories; external
locators are not counted as included payload. The capsule is a source/evidence
review candidate, not the integral scientific master or a claim of clearance
for every possible redistribution. Its excluded inputs have exact locators
and hashes in the inventory, to be acquired under their own source conditions.

The exact deposited0.2.0 ZIP is preserved unchanged if encountered in the
source stream. The candidate0.3.0 is not written to Zenodo. Its manifest lists
every actually archived member/hash. Deterministic tar/gzip headers and sorted
paths permit rebuild. Verification extracts each regular file into an owned
temporary file, hashes it, removes that temporary file, and checks the complete
member set. This limits peak extraction space without substituting a listing
for extraction. SHA/byte equality is checked; no model inference is involved.

## CPU job contract

Python3 standard library only. Root owns the isolated Actions workflow.

```
python3 integral_capsule.py \
  --index-root frozen_reference/experiments/e3_completion_20261120/release/index \
  --output /tmp/e3-integral \
  --hydrate --workers 6 \
  --max-network-bytes 7000000000 --max-source-bytes 3000000000 \
  --max-artifact-bytes 190000000
```

Only `/tmp/e3-integral/public/` may be
uploaded. Never upload the entire output directory. The archive compressed
ceiling is190MB and all public output files together must remain below200MB.
Source decompression is capped at3GB and total counted network body bytes at
7GB. Six workers and a55-minute job budget are expected. Full run outcome is
the executed `execution_receipt.json`; this design document is not that outcome.

Human kit inspection/comparison fixtures are software checks only. External
human execution and training levelC remain distinct pending products.
