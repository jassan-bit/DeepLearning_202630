# Public documentary deposit verification

This is a read-only public check prepared with AI assistance after the owner
published https://zenodo.org/records/24356217. It downloads only the selected
916861-byte documentary ZIP, validates its reviewed SHA256 and all 65 member
hashes, and follows the DOI to the same public record.

The expected SHA256 is
`aa59b73311624f2a92b4497f98bfe182e3c2c320c0e0b092fefee5686bce8349`.

No Zenodo account, credentials, uploads, publication, model execution, fits,
calibration, or new scientific predictions are involved. No claim of R3,
integral archiving, legal review, or human training reproduction is made.
Metadata differences are reported separately from archive/DOI integrity.

The dedicated verification branch temporarily replaces the existing public
acceptance workflow with this GET-only check because that workflow already
supports a branch-create trigger. The default branch and deployed Render
source remain unchanged. This branch does not repeat public inference tests.

Run: `python3 verify_public_deposit.py /tmp/e3-zenodo-publication`

The completed run is Actions44825078081/job136595516673 at launcher commit
5fcb367c7645228ada68f1e3ad079735a1433f0e. Its passing `verification.json`,
exact public API response and provenance are preserved under `receipts/`.
The GitHub proof source is 0b39bca0b0c8f211661244b70cfe53397fb9693c. The
default branch's workflow is not replaced by this documentary packet.

Only rows15 and16 of the previous 17-requirement matrix are updated here;
the other15 rows and the historical Render packet remain unchanged. The
archive folder preserves the deposited ZIP and its prepublication descriptors
byte for byte. Reserved/unpublished markers inside that ZIP describe the
packaging instant; the external public receipt establishes the later deposit.
Its53 source copies refer to60a9d988, not the newly revised book.

Own selected code is MIT and own selected documentation/receipts/captures
CC BY4.0 by the owner's decision in OWNER_DECISION.json. Third-party exceptions
remain in RIGHTS.csv. No global repository license is inferred. Metadata
corrections remain pending in METADATA_CORRECTION.md; the public API still
reports the old title/date and lacks an explicit software version. The legacy
API reports CC BY4.0; the owner must review the full license list for MIT.

Post-publication metadata can be corrected via the existing Zenodo record's
Edit and Publish flow without changing the DOI or deposited ZIP.
