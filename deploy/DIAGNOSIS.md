# Cloud checkpoint: bounded diagnosis

This retained result uses the **original small-cache configuration**. See the
[machine-aware configuration correction and longer same-VM comparison](RESOURCE_PROFILE.md)
for the follow-up. The measurements below and raw records are preserved.

This is an operator diagnostic on the existing D4ds_v4 (four vCPUs, 16 GiB), not a
benchmark study or a cloud speedup claim. Historical laptop evidence is unchanged.
The prompt was “What is 2 + 2? Give a short answer.” with exactly four requested
tokens, 512-token context, 3.5 GB planning budget and 1.2 GB starting cache.
The short token prefix is not a task-correctness evaluation.

| Threads | Mode | Import / load (s) | First prefill (s) | First decode (tok/s) | Warm prefill (s) | Warm decode (tok/s) |
|---|---|---:|---:|---:|---:|---:|
| 4 | LeanMoE | 2.17 / 6.87 | 44.35 | 0.243 | 43.24 | 0.312 |
| 4 | Windows paging | 1.05 / 2.02 | 49.92 | 1.988 | 6.63 | 4.018 |
| 2 | LeanMoE | 0.96 / 5.30 | 43.38 | 0.241 | 43.28 | 0.309 |
| 2 | Windows paging | 0.89 / 2.05 | 60.40 | 1.332 | 7.22 | 2.517 |

Exact values are in [four-thread measurements](../results/cloud_diagnosis_initial.json)
and [two-thread retest](../results/cloud_diagnosis_threads2.json). Each mode uses a
fresh child process, then repeats in the same loaded engine without clearing its
expert cache. Four-thread Numba dispatch compilation/cache loading took 2.80 s in
the first LeanMoE request and 1.31 s in the first paging request, then zero on repeats.
Those times overlap the phase times and must not be added again. Additional native
library initialization is not separately identified by that instrument.

Four-thread LeanMoE requests used about 71–75% of aggregate vCPU capacity and
recorded about 13.0–13.8 GB of system disk reads. Warm paging used about 95% and
recorded about 4.7 MB. Counters include other processes, and read time can overlap
CPU work or concurrent I/O; they do not establish isolated disk throughput or prove
a single cause. The retest adds per-token CPU/disk deltas to distinguish prefill
from decode. OS file-cache state was uncontrolled, never flushed, and preceding
manifest verification could warm it. Mode order and previous runs matter.

All requests generated the same four token IDs. The observed warm paging benefit
is consistent with file-cache residency on this larger-memory server. LeanMoE
remained slow with its model/cache retained, so unloading alone cannot explain
the measured throughput. A two-thread probe did not improve it. **Four threads
remain selected; no speculative performance fix or isolation relaxation was applied.**
Further experiments would be needed to separate storage latency, cache churn,
prefill routing and CPU/kernel effects. Neither mode's cloud-versus-laptop change
can be attributed to LeanMoE without a matched same-machine experiment.

The two diagnostic passes took approximately 180.5 and 189.1 seconds of measured
work, within the roughly fifteen-minute initial diagnosis window. The public
backend was reserved during each pass; authenticated HTTPS remained in place and
the backend was restarted by cleanup afterward.

The separate [live HTTPS verification](../results/cloud_live_verification.json)
also exercised real streaming, a live budget update, cancellation and sequential
comparison. That two-token comparison recorded LeanMoE prefill 43.27 s / decode
0.171 tokens/s and paging prefill 85.67 s / decode 1.751 tokens/s. It is a short
functional check with uncontrolled file-cache state, not another benchmark. Its
longer paging prefill illustrates why the warm-repeat figure cannot predict every
visitor's first-token latency. These checks preceded the pinned release and its
new `build.commit` field; the raw record therefore has no reported build hash.

## Store integrity

Both durable C: and fast D: copies passed SHA-256 verification against the VM
manifest. The supervisor selects D: for serving. Available disk inspection reports
NTFS on both drives, approximately 150 GiB D: capacity and 134 GiB free before the
diagnostic. See [integrity record](../results/cloud_store_integrity.json).

VM `experts.bin` and `resident.bin` hashes match the existing laptop manifest.
This checkpoint rehashed the VM copies, not the laptop binaries. Per-entry field
digests in both indexes match the laptop except `quant_error`, a diagnostic field;
offsets, shapes, sizes and record hashes match. Original stores and manifests were
not rewritten. The original model license and ExpertRelay attribution remain with
the deployment materials.

## Credential and operational remediation

The exposed proxy key was rotated in private configuration and both supervised
roles restarted. The old key was rejected with HTTP 401; new proxy access worked.
The exposed transfer URL's dedicated storage account was already absent from the
configured subscription, and its endpoint no longer resolved. No visitor/admin
password appeared in the cited output; those passwords were not changed. No
credentials are included here. The owner continues distributing access privately.

The release workflow also removes the old eight-day task execution limit, which
could otherwise terminate service before the approved deadline. Supervisors still
obey the private shutdown timestamp, and the Azure shutdown and $150 alert budget
are unchanged. The earlier $161 projection exceeds that alert threshold by $11;
disk/IP costs continue after deallocation.

The prepared pinned release passed 178 VM tests; three JavaScript checks were
skipped because Node.js is absent on the VM. The full local working tree passed
189 tests (including eight unfinished task-benchmark tests kept out of deployment).
The published deployment suite therefore contains 181 tests. Real Edge checks at
desktop/phone widths covered navigation, Cloud entry/local instructions, recorded
figures and layout, with no page errors or horizontal overflow. Bicep compilation
and PowerShell parsing also passed. A release-script fix handles native stdout
and stderr in private files and checks exit codes: Caddy's successful stderr
diagnostics otherwise became a terminating PowerShell 5.1 error under redirected
output. Both zero-exit/stderr and nonzero-exit handling were checked locally.
