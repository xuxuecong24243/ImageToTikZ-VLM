# Public historical regression fixtures

These files let a clean checkout run the historical Scene/TikZ checks without
publishing the private `outputs/` archive. The 56 original acceptance results,
49 full TikZ hashes, expected lengths, and badge/feedback assertions are unchanged.

Sixteen Scene copies contain redacted machine paths **only under `metadata`**.
Their canvas, styles, nodes, edges, labels, text, geometry and rendering remain
unchanged. `compatibility_report.json` records separate `original_sha256` and
`public_sha256`, exact redacted field pointers, and relative archive provenance.
The legacy `sha256` key continues to describe the original bytes; it must not be
used as the public-copy hash. Forty other Scene copies preserve the original bytes.

`native_exports_sha256.json` contains only the original package-export hash used
by tests, extracted from the recorded historical manifest. It is explicitly not
the full archive manifest or a record of API responses.

The original archive-byte check remains independent: it runs when all original
files are present locally, and explicitly skips when that private archive is
absent. Public fixtures still run all 56 validation conclusions and 49 exact TikZ
outputs; the compatibility suite is not skipped on a fresh checkout.

`middle_badge/` preserves an older, reviewed Scene used by a compatibility test.
It is test data, not the recommended Local recognizer and not an automatic OCR
result. `visual_feedback_raw.txt` is unchanged visible feedback for parser tests,
not a provider HTTP response or an instruction to apply its changes.

Inputs were screened for machine paths and common key/token indicators before
packaging. No images, PDFs, model weights, full raw API envelopes, or `.env` files
are included. Historical originals remain untouched in local `outputs/`.

`test_offline_visual_patch_inputs.py` separately uses temporary, explicitly
synthetic transaction/bounds fixtures for four loader-format tests. It exercises
the actual loader's file hashes and mismatch rejection, not an authentic model
request or historical PDF measurement. Its synthetic inputs are created at test
runtime; they are not stored here or presented as recognition results.
