# Machine-aware dashboard memory

The cloud dashboard inherited the laptop's 3.5 GB planning budget and 1.2 GB
expert-cache request. That was a configuration defect on a 16 GiB VM: paging
could retain far more weights. The original measurements remain in
[the initial diagnosis](DIAGNOSIS.md); their configuration is still available as
**Constrained adaptation experiment**, separately from **Automatic for this machine**.
No VM resize, OS-cache restriction or engine-kernel change was made.

## Allocation policy

[resources.py](../src/dashboard/resources.py) measures physical RAM total and
available, Windows commit headroom, process private commitment and working set.
Missing or inconsistent measurements refuse a new allocation with a retryable
message. Private commitment is never treated as physical residency or free RAM.

All GB figures are decimal; GiB is binary. The safety reserve is
`max(1 GB, 12% of measured physical RAM) + 256 MiB`. The last term allows for
measurement races; the reserve is left for Windows and other work, rather than
allocated to the expert cache. Runtime estimates separately cover maximum-context
KV, activations, scratch space and kernel/JIT allowances. Qwen's estimated future
runtime allowance is 433,669,632 bytes with the current 512-token context.

Before loading, choose whole expert records from the smaller of physical and
commit headroom, subtracting reserve, resident weights and future runtime needs.
Load the resident weights and only the minimum expert pool, then **measure again**.
For this model the resident allocation is 1,560,047,616 bytes, excluding the
mapped embedding; the minimum pool is 13 records of 8,671,232 bytes.
After resident loading, do not subtract resident weights again: their pages and
commitment are already reflected in the measured available RAM and commit
headroom. The empty bootstrap pool gets commit credit only, never assumed
physical-residency credit.

The second selection is:

```
physical room = available RAM - reserve - future runtime
commit room   = commit headroom - reserve - future runtime + bootstrap commit
cache bytes   = floor(min(physical room, commit room, all experts) / record) * record
```

Resize and remeasure once more; refuse and release the model if reserve has
disappeared. The reported automatic planning budget includes the larger of the
component estimate and measured existing process commitment plus future
allocations/cache. This is accounting metadata, not a Windows RAM cap.
Manual/constrained profiles also cap selection by their explicit cache request
and component-based planning budget. An unsafe manual request is refused rather
than silently misreported as its requested size.

Between tokens, a dashboard-only pressure guard measures headroom, shrinks on
pressure, or stops/releases the model if minimum safe operation is impossible.
Live growth uses only *additional* measured room; it never adds committed cache
capacity to free physical RAM. Checks cannot prevent another process allocating
memory during an indivisible forward pass. Measurements and estimates are not a
hard process or system memory limit.

Automatic runs load fresh and repeat selection on every model/mode switch.
Local manual runs may retain immutable model/cache resources after a measured
headroom check; fresh KV is created per request. Public requests always release
model, expert cache, KV and prefetch resources on completion, cancellation or
error. No resources or prompt state are retained across public visitors.
Existing ExpertRelay/CLI budget and adaptation behavior is unchanged.

The Settings selector distinguishes Automatic, Manual and Constrained; the last
retains 3.5 GB / 1.2 GB. Run details records the effective profile, planning budget,
actual allocated cache and reserve. Public manual requests retain the previous
4 GB budget / 1.3 GB cache admission caps. Automatic capacity is computed by the
server from trusted measurements; supplied numeric settings cannot override it.

## Bounded same-VM comparison

[Raw outcomes and samples](../results/resource_profile_comparison.json) preserve
all nine runs, including the three short diagnostic probes. Total parent elapsed
time was **588.80 seconds**, below the 850-second whole-experiment limit.
The active store was the existing NTFS temporary-SSD copy, checked by the backend
supervisor against its independently transferred manifest. The artifact includes
that manifest (including notices) and source hashes. Store verification and prior
reads can warm the OS cache; no forced flush was performed.

| Setting | All three methods |
|---|---|
| Machine | Existing Azure D4ds_v4, East US, Xeon Platinum 8272CL, 4 logical cores, 17,174,360,064 physical RAM bytes |
| Model/store | Same Qwen1.5-MoE-A2.7B-Chat int8 manifest; expert and resident binaries match the laptop |
| Compute | NumPy backend, fused int8 kernel; OpenBLAS 4, Numba 4 |
| Prompt | “What is 2 + 2? Give a short answer.”; exact templated token IDs in every record |
| Decoding | Greedy argmax; fixed 16-token output, EOS stopping disabled for this diagnostic; maximum context 512 |
| Process state | Fresh child process/model/KV per case, initially empty explicit cache; no shared visitor state |
| Order | Four-token probes: small, automatic, paging. Main repeat 1: automatic, paging, small. Repeat 2: paging, small, automatic |
| Timing | Load separate; prefill is first forward pass; decode is 15 later forwards per repeat, excluding load/prefill |
| Instrumentation | CPU and guest disk counters per step; private/working-set/available/commit samples every 0.1 s; source demand/prefetch/wait counters |

All six main output sequences match. This establishes execution consistency for
this workload, not general answer correctness. Main figures below combine 30
decode steps as total tokens / total forward time; probes are excluded.

| Measurement | Small-cache LeanMoE | Automatic LeanMoE | Windows paging |
|---|---:|---:|---:|
| Actual expert cache | 1.196630016 GB, 138 slots | 10.969108480 GB, 1,265 slots | OS managed |
| Occupancy at start / end | 0 / 138 | 0 / 1,265 | Not applicable |
| Decode tokens/s | 0.3036 | 1.9976 | 2.1980 |
| Repeat rates | 0.3036 / 0.3036 | 2.0069 / 1.9884 | 2.1587 / 2.2388 |
| Load seconds, range | 2.52–2.56 | 1.20–2.49 | 0.056–0.057 |
| First-token prefill seconds, range | 43.30–43.32 | 43.29–43.32 | 48.54–48.73 |
| Peak sampled private commitment, GB | 3.230 | 13.033 | 0.482 |
| Peak sampled working set, GB | 2.943 | 12.722 | 13.292 |
| Minimum sampled system available RAM, GB | 12.375 | 2.664 | 2.100 |
| Decode cache hits / expert uses | 312 / 2,880 (10.8%) | 2,586 / 2,880 (89.8%) | Not applicable |
| Decode demand bytes/token, MB | 552.36 | 35.84 | Not an explicit-source read counter |
| Decode prefetch bytes/token, MB | 263.03 | 80.06 | None |
| Decode forward blocked read/wait seconds/token | 2.995 | 0.232 | Source does not instrument page-fault waits |
| Decode guest disk reads, total GB | 24.462 | 3.477 | 1.786 |
| Decode CPU seconds / forward wall seconds | 304.89 / 98.81 | 53.73 / 15.02 | 37.28 / 13.65 |

Automatic sizing improved decode **6.58×** over the small cache. Paging remains
**10.0% faster** than automatic LeanMoE in these repeats, using a larger sampled
working set and much less private commitment. These are unequal-memory practical
configurations on the same machine, **not an equal-memory comparison**.
The historical 0.312 versus 4.018 comparison used a warm retained engine and only
three decode steps; do not splice its rates into this fresh-process experiment.

The remaining measured costs are expert misses/prefetch and substantial cold
prefill: automatic decode still reads about 116 MB/token through its explicit
source and spends 0.232 s/token blocked on expert availability. Its first prefill
reads about 10.67 GB into the empty cache and remains about 43.3 seconds.
Numba dispatch compilation/cache-loading was 0.27–0.30 seconds per main run,
overlapping forward timing, so it does not explain the roughly 43-second prefill.
No isolated claim that physical disk bandwidth is the bottleneck is supported:
guest disk service-time counters are small and differ from forward wait times.
Read/wait and CPU times overlap and must not be added together.

Limitations: one short prompt, two main repeats, only 15 decode steps per repeat,
uncontrolled OS file cache, guest-wide disk counters (not host physical media),
sampling can miss peaks, asynchronous prefetch can cross step boundaries.
Paging's source `bytes_read` counts mapped bytes accessed, not physical reads;
only guest disk counters are used for its I/O comparison. There is no cold-cache,
task-success, equal-memory, or broad speedup claim. A genuine constrained-hardware
comparison remains separate work; application budgets do not constrain OS cache.

The experiment ran the precommit candidate identified by its source hashes, based
on deployment 3fd7b20. Final changes add stream-error cleanup, UI/test coverage,
live available-RAM summaries and corrected *reported budget accounting*; the cache
selection formula used for these outcomes is unchanged. Deployed smoke checks
record final profile/capacity independently.

## Reproduction

Use an idle existing Windows machine in a maintenance window, keep authenticated
HTTPS protection, stop only the inference supervisor, and restore it in `finally`.
Do not run concurrently with dashboard inference. Supply the operator's private
store and manifest locations; never put credentials into commands or artifacts.

```powershell
& $PythonExe deploy/store_manifest.py verify --store-root $Store --manifest $Manifest
& $PythonExe -m deploy.profile_compare --store $Store --manifest $Manifest --out results/resource-profile-reproduction.json --tokens 16 --seconds 850
```

Every case has a fresh process; the parent terminates timed-out process trees and
preserves partial outcomes. Errors go to a private operator log. Do not publish
those logs without inspection. The full task-success study remains unfinished;
its existing artifact is a pilot. Cost controls, VM size and shutdown are unchanged.
