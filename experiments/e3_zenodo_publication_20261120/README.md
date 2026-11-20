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

The authoritative outcome is the resulting `verification.json`, not this
prepared procedure. Public receipts are printed in the job log and retained
as an Actions artifact even on failure. Post-publication metadata can be
corrected via the existing Zenodo record's Edit and Publish flow without
changing the DOI.
