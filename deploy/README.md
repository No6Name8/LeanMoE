# Bounded public Windows demo

The cloud serves the existing Run / Compare / Evidence dashboard and the actual
ExpertRelay-derived LeanMoE engine with the Qwen int8 store. There is no third-party
inference API. Run and Compare share an Execution selector. On the local dashboard,
Cloud uses a fixed-origin, verified-HTTPS loopback relay with no visitor login.
The cloud's signed visitor cookie stays in the local relay memory,
separated by an HttpOnly loopback session cookie. The hosted dashboard selects
Cloud directly; Local requires a separate local installation. Cloud inference
runs on the server, never on the visitor's computer. Historical laptop evidence
is unchanged and separate from live measurements.
Cloud access is useful without a speedup claim.

## Existing configuration and costs

Azure East US, Standard_D4ds_v4: four vCPUs, 16 GiB RAM, NTFS temporary SSD.
The durable store is on C:, the serving copy on D:. Temporary storage can disappear
on deallocation, redeployment or host maintenance. The supervisor verifies C:,
copies to D:, then verifies D: before starting the localhost backend.

[Pricing reference](pricing-reference.json) records the official Azure Retail Prices
API rate: Windows VM $0.41/hour, 128 GiB P10 disk $19.71/month, public IP $0.005/hour.
Using 730 hours/month, 10 GB outbound at an assumed $0.087/GB and $0.10 automation
reserve, 48 running hours cost approximately $22.19; seven days approximately $75.23.
These exclude tax, extra storage/traffic and setup time. Run
`python deploy/estimate_costs.py --service-hours 48 --setup-hours 0 --teardown-hours 0`
for itemized assumptions; update prices before new spending.

The existing subscription alert threshold is $150. The earlier approximately $161
projection exceeds it by $11. Alerts are not spending caps and include other
subscription spending. The existing shutdown is November 14 at 17:00 UTC.
Neither setting is changed by publishing code. Disks/IP remain billable after
deallocation. Verify the schedule, linked runbook, VM-scoped managed identity and
actual deallocated power state. Only the owner may approve a different budget/deadline.

## Clean installation

Provisioning is a separate, paid action requiring account-owner approval. The
[Bicep template](main.bicep) and [provisioner](Provision-Azure.ps1) retain the approved
SKU, restricted operator /32, public 80/443 and VM-scoped shutdown identity.
The provisioner does nothing without `-Apply`. Keep parameter files outside Git.
It creates subscription-wide quarterly alerts; it does not enforce a spending cap.

1. Check out an approved **full commit hash** from GitHub on Windows Server.
   Obtain x64 Python 3.13 from [python.org](https://www.python.org/downloads/windows/),
   verifying its official signature, and Caddy from its
   [official release](https://github.com/caddyserver/caddy/releases), checking the
   published SHA-256. Do not download executables from mirrors.
2. Create the private site configuration from [site.example.json](site.example.json).
   Set `accessMode` to `public` for the no-login demo. Generate the backend proxy
   key using a cryptographic RNG, at least 32 characters. Protect the private
   directory for administrators, SYSTEM and read-only Local Service. Never commit
   secrets or print private files. Legacy `private` mode still supports individual
   visitor accounts and password hashes in `visitors.caddy`.
3. Privately transfer the store and [model notices](model-notices/). Generate a
   SHA-256 manifest with [store_manifest.py](store_manifest.py); transfer the
   manifest through an independent authenticated channel. Verify **every file**
   after transfer. No script automatically publishes weights. For rebuilt stores,
   preserve separate manifests: verify binary identity and compare index fields
   semantically; do not rewrite a source manifest to hide differences.
4. Run `Install-Guest.ps1 -PythonExe <verified-python.exe> -CaddyExe <caddy.exe>
   -CaddySha256 <published-sha256>`. Elevated VM administration is required.
   [Install-Prerequisites.ps1](Install-Prerequisites.ps1) downloads Microsoft's
   [official VC++ runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
   verifies Authenticode and the Microsoft publisher, and installs silently.
   It installs [certifi](https://github.com/certifi/python-certifi) from verified
   PyPI HTTPS and tests Python HTTPS trust. A required reboot stops installation.
   Never use unverified SSL contexts, `trusted-host`, or modified certificate bundles.
5. Installation creates the venv, runs tests/preflight and validates Caddy before
   starting Local Service supervisor tasks. Open only the already-approved 80/443
   guest firewall rules after checking readiness. Backend remains 127.0.0.1:8770;
   Caddy administration remains loopback. No Vercel layer is required.

Initial installation requires a clean Git checkout; production updates use the
pinned GitHub release workflow below. The supervisor reads the recorded commit
for a release or Git HEAD for a checkout. Keep the tested Python dependency
snapshot with the deployment record; broad local requirements alone are not a lock.

## Publish an approved commit

Commit deployment/access work separately from unfinished task benchmarks. Review
all pending paths and scan for secrets before staging. Push the commit first.
On the existing VM run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File deploy\Publish-Guest.ps1 -Commit <40-character-hash> -Apply
```

[Publish-Guest.ps1](Publish-Guest.ps1) downloads that immutable GitHub commit over
verified HTTPS into a new release directory, records archive SHA-256, tests it,
checks the model and proxy, then selects it for both tasks. The original checkout,
venv, stores and private configuration remain in place for recovery. It refuses
an existing release directory and a busy engine. Publishing briefly interrupts
availability; reserve a maintenance window. Verify unauthenticated HTTPS
`/api/status` reports `public_demo: true`, the selected `build.commit`, server
location and model availability; then test Run and Compare in separate browsers.
No deployment script changes the approved shutdown date or budget.

For the existing private deployment, back up the private `site.json`, set only
`accessMode` to `public`, and retain its old `visitorAccounts` for the pre-switch
readiness check. The new supervisor selects [Caddyfile.public](Caddyfile.public)
and the backend's fixed `public` identity. Publish the pinned release while idle;
verify the new page without Authorization headers. To roll back, restore the
private site configuration and prior release tasks together. Never remove the
backend proxy key or expose port 8770. The public URL can be used by anyone,
including people without access to the Azure account; the existing cost controls
and shutdown schedule remain important.

## Access protection and limits

Caddy admits public visitors over HTTPS but overwrites proxy identity headers
with a fixed public identity and the private backend key. The backend still binds
only to loopback and requires that key, the exact configured HTTPS Origin,
loopback Host and a signed Secure/HttpOnly/SameSite browser cookie. Separate
browser sessions cannot inspect or control each other's request. There is one
active job, no queue, three admissions/minute **globally**, 16 active
backend connection threads, a 16 KB body cap, 4,000 prompt characters, a 512-token
context and 96 output tokens/mode. Automatic allocation uses measured backend RAM
and commit headroom with a safety reserve. Explicit manual requests retain the
4 GB budget / 1.3 GB cache admission caps; a visitor cannot supply an automatic
capacity. Budgets are process planning targets, **not** caps on Windows file cache
or physical RAM. See [resource policy, raw comparison and reproduction](RESOURCE_PROFILE.md).

Cancellation is checked between forward passes. A ten-minute whole-job timer
requests cancellation; a stuck job exits the backend after eleven minutes so the
supervisor can restart it. Public job cleanup unloads the model and expert cache;
the process JIT cache and OS file cache can remain warm. Comparison runs both
modes sequentially on the same server, each with a fresh model load.

## Verification and bounded diagnosis

Run the full pytest suite from the release, then verify no-login HTTPS access,
direct-backend rejection, wrong-origin rejection, token/body limits, separate-
browser isolation, busy behavior, owner-only cancel/budget and final idle
cleanup. HTTP tests use tiny random weights; they do not prove real-model quality.

For initial performance diagnosis reserve the idle backend for maintenance and
run [diagnose.py](diagnose.py) against the configured fast store. Its parent bounds
both modes together to 850 seconds, terminates timed-out children and preserves
partial results. Use a wrapper with `finally` to restart the backend. It records
imports, load, Numba dispatch compilation/cache loading, prefill, decode, process
CPU and system disk counters. Each mode gets a fresh process and two fixed
four-token requests. The second request retains engine/cache state; Windows file
cache is **uncontrolled**, and manifest verification can warm it. No cache flush,
VM resize, purchased resource or isolation relaxation is required. System disk
counters include unrelated work and do not establish disk throughput causality.

The [task-success pilot](../results/task_benchmark_pilot_20261030-134509.json)
is a **three-task harness check**, both modes, first request plus repeat. The
[fixed specification and reproduction guide](../docs/TASK_BENCHMARK.md), runner,
scorer and pilot were committed separately from deployment/access changes at
`ba037e065ef0f851c9594a22ea0a2248b9c32b0a`. The full laptop task study
remains unfinished; these harness checks are not real-model task-success results
for all twelve tasks. Model store transfer and operator credentials remain private.

## Recovery, credential remediation and teardown

If publishing fails before switching, keep the old tasks. After a failed switch,
restore their saved actions from `C:\LeanMoE-private\previous-tasks.json`, stop only
the LeanMoE supervisors/listener processes and restart both tasks. Use the prior
release and dependency snapshot if dependencies changed. Do not overwrite a
working release or delete a store to recover. Inspect private Caddy/task logs;
never paste configurations, URLs with signed tokens, or raw commands containing keys.

Rotate a leaked proxy key in private configuration and any private operator copies,
then restart both owned roles while idle. Verify old key rejection and new HTTPS
access; existing browser cookies become invalid and are renewed on reload. Rotate
legacy private-mode visitor passwords if exposed. Revoke signed transfer URLs by removing their
stored access policy or rotating the signing storage key; deletion of a dedicated
transfer account also invalidates its endpoint. Confirm absence using Azure and
an endpoint check. The public demo needs no visitor password; operator and proxy
secrets stay private and must never appear in browser code or reports.

To stop compute, deallocate the existing VM through Azure and verify
`PowerState/deallocated`; an OS shutdown alone can keep compute billable. For
full teardown, first preserve needed evidence and private backups, then have the
owner explicitly approve deletion of the deployment resource group, OS disk,
public IP and any remaining transfer storage. Verify no billable resources remain.
Do not delete unrelated subscription resources or raise spending controls.
