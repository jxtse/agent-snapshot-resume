# Proxy parity evidence and changes

Scope: `fix/proxy-parity-gaps`, local offline work only. No commits, pushes,
sandboxes, model calls, or paid runs. Existing `.brief.md` was left untouched.
No Research Brain writeback because this task explicitly restricts writes to this worktree.

## Implemented

`parity.content_reference` now carries the session attachment record's **top-level
`rendered[].content`** into the attachment reference. Previously it used only the
structured `attachment` fields, losing the exact wording CC had already persisted.
`resume_proxy.rendered_text_problems` matches this complete, contiguous recorded
text once, at the existing attachment context position. CC's observed coalescing
into a system message removes the exact outer `<system-reminder>\n` and
`\n</system-reminder>` pair; both complete forms are supported. No fuzzy matching,
new general prose templates, byte-count tolerances, or equality exemptions were added.
The original position gate, request-byte check, tools/system checks and hint gates
remain in force. New content references require rebuilding bundles; old captures
are not retroactively declared valid or modified.

The raw resumed requests in the three reminder cases already contain the exact
original reminder payloads. `rewrite` preserves them; the defect was their
explanation in the parity gate, not absent wire content. No invented reminder
insertion is needed for these captures.

Small real session fixtures contain a `task_status` and an `edited_text_file`
record, without headers, credentials, or image blobs. Tests cover exact payloads,
wrapper coalescing, byte-preserving rewriting, changed text, duplication,
reordering, wrong position, image data mutation, exact companion markers, and
continued rejection of missing interrupted thinking.

## Per-case offline findings

Evidence root below: `/Volumes/Storage/hermes/workspace/asr-dsv4-o70-hint-20261113`.
For each fake trial, used `agent/resume/{bundle,proxy_spec}.json` and
`agent/resume-proxy/*/{first_request_raw,first_request_parity}.json`. The bundle's
`trial_dir` identifies the authoritative parent session and proxy logs. Rebuilt
content references from that session prefix (stripping prior resume scaffolding
for chained runs), then ran the pure rewrite/content checks locally.

| Case | Result |
| --- | --- |
| `jobs/r1-supraglacial-lake-classification-s225-fake-hinted/supraglacial-lake-classification__c8LoEuB` | PASS after rebuilding reference. Image dimension markers are persisted `isMeta=true, turnCompanion=true` user text records (first at zero-based session line 93). The current baseline already collects their exact digests in `meta_block_sha256`; old captured specs predate this. Nested image data remains bound by the tool-result digest. No blanket image-marker acceptance added. |
| `jobs/r1-symbolic-regression-s98-fake-hinted/symbolic-regression__Sbb9r7F` | Still REFUSED. The unexplained text is exactly `Your response above was cut off mid-stream. Resume directly from where it stops — no apology, no recap. If none of it survived, answer the request from the start.` It is a persisted companion record at session line 70, already handled by rebuilding the current meta reference. This exposes a second mismatch: reference block 37 is interrupted assistant thinking beginning `Real nested AUC ~0.59 vs null ~0.53.`; the resumed request omits it and resumes with the next thinking block beginning `Nested CV with univariate selection: real AUC ~0.594`. There is no exact original request body at this cut to prove that omission faithful. The non-tail thinking block is not ignored. |
| `jobs/r2b-virtual-baseline-localization-s711-fake-hinted/virtual-baseline-localization__QerPE5S` | PASS with new reference. Parent session line 276 persists `task_status` for `b15eutpj6` and exact rendered text. |
| `jobs/r2b-qsm-reconstruction-s758-fake-hinted` | PASS with new reference. `task_status` for `bcy7fgpjt` contains the exact original reminder, including the rendered command. |
| `jobs/r2b-dapi-he-alignment-s1014-fake-hinted/dapi-he-alignment__jrx3mVR` | PASS with new reference. Parent session lines 841 and 891 persist `task_status` and `edited_text_file`; the latter includes the exact changed-on-disk reminder and numbered snippet. |

The four passing captures already had `diffs={}` and `byte_diff=0` in their
recorded first-request parity reports. Offline rewriting does not change their
wire content; the rebuilt reference now explains the exact content. This is not
an independent byte-for-byte comparison with a stored original body: that body
was not logged. No new live fake run was started.

## Carried hints: not safely reconstructable from available logs

The initial hypothesis that the parent request body exists at every cut is false
for the supplied artifacts. `agent/model-proxy/requests-*.jsonl` stores metadata
(model, message count, byte count, usage, status, timestamps), **not messages or a
request body/hash**. Likewise resume-proxy `requests.jsonl` records actions and
counts, not full bodies. Equal byte counts cannot establish exact hint placement.

* Spin-glass parent: `jobs/r1b-spin-glass-groundstate-s1170-hinteda/spin-glass-groundstate__hnyYWt6`.
  The selected request index is 4, with 71 messages. The parent's stored first
  rewritten body has 59 messages; its three stored post-compaction bodies have 5
  each. The failing child's raw resume has only 15 messages. Its carried hint
  anchor is `call_00_z5u3lpL5HKTvfseNfu5T8176`.
* Navigation parent: `jobs/r1b-navigation-sensor-calibration-s1177-hintedb/navigation-sensor-calibration__fRax7c9`.
  Selected request index 32 has 116 messages. Stored first rewritten body has 20;
  both stored post-compaction bodies have 5. Child raw resume has 99. Its anchor
  is `call_00_pSoh13Hyow0L59K2WVBN9455`.

These are substantial lost-context failures, not just a misplaced hint. The
parent's stored first bodies contain summaries; the failed child rebuild lacks
both the required anchor and a usable summary. The available evidence does not
establish the exact parent's context at the new cut. `chain.py` and the carried
hint refusal remain unchanged. Moving hints to an arbitrary surviving user turn,
or borrowing a later compaction probe, would not satisfy the requested parity.

## diag-chipseq fallback assessment

`batch/r2/state.json:45-65` records the missing summary/preserved segment (54
messages vs 68). The cut bundle is
`batch/r2/units/diag-chipseq/bundles/diag-chipseq-s836-hint/bundle.json`.
Its parent is `jobs/r1-diag-chipseq-s738-hintedb/diag-chipseq__6AuNSQq`, selected
request index 17, 68 messages. The stored first rewritten body has 17 messages;
the only stored post-compaction body has 5. The model-proxy log again has only
metadata. Thus a generic exact-request reconstruction fallback cannot safely
repair this existing cut from the supplied evidence. It was not implemented.

An exact-body fallback would require a full effective request captured at the
identified cut, bound to the selected response/session, plus a stable prefix
anchor for subsequent requests and exact parity verification after removing only
the new hint. Saving every request body or reconstructing missing context from
unverified session heuristics is not a substitute for that evidence.

## Validation

Command: `env -u PYTHONPATH uv run --project . python -m pytest -q`.
Baseline: 329 passed, 1 skipped, 7 subtests passed.
Final: **338 passed, 1 skipped, 7 subtests passed** (10.93 s).
`git diff --check` also passed.
