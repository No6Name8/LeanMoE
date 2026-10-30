# Protected Windows deployment

The cloud serves the existing Run / Compare / Evidence dashboard and the actual
ExpertRelay-derived LeanMoE engine with the Qwen int8 store. There is no external
inference API. The Cloud tab links to the authenticated server; it does not run
remote inference on the visitor's computer. Historical laptop evidence is unchanged.
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
   Generate the proxy key using a cryptographic RNG, at least 32 characters.
   Create individual visitor passwords privately and bcrypt hashes using Caddy's
   `hash-password` **stdin**, never password command arguments. Save hashes in
   `C:\LeanMoE-private\visitors.caddy`; protect the directory for administrators,
   SYSTEM and read-only Local Service. Never commit secrets or print private files.
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
availability; reserve a maintenance window. Verify `/api/status` authenticated
`build.commit`, server location, model availability and browser behavior afterward.
No deployment script changes the approved shutdown date or budget.

## Access protection and limits

Caddy authenticates individual accounts and overwrites proxy identity headers.
The backend requires the proxy secret, exact configured HTTPS Origin, loopback
Host and a signed Secure/HttpOnly/SameSite browser cookie. Separate browser
sessions cannot inspect or control each other's request, even with a shared account.
There is one active job, no queue, three admissions/minute/account, 16 active
backend connection threads, a 16 KB body cap, 4,000 prompt characters, a 512-token
context and 96 output tokens/mode. Budgets are process planning targets (maximum
4 GB), **not** caps on Windows file cache or physical RAM.

Cancellation is checked between forward passes. A ten-minute whole-job timer
requests cancellation; a stuck job exits the backend after eleven minutes so the
supervisor can restart it. Public job cleanup unloads the model and expert cache;
the process JIT cache and OS file cache can remain warm. Comparison runs both
modes sequentially on the same server, each with a fresh model load.

## Verification and bounded diagnosis

Run the full pytest suite from the release, then verify HTTPS authentication,
wrong-origin rejection, token/body limits, same-account browser isolation,
cross-account isolation, busy behavior, owner-only cancel/budget and final idle
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

The existing task-success artifact is a **three-task pilot**, both modes, first
request plus repeat. The full task-success study remains unfinished. The pilot
and its runner/scorer are kept separate from deployment commits.

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
visitor passwords if exposed. Revoke signed transfer URLs by removing their
stored access policy or rotating the signing storage key; deletion of a dedicated
transfer account also invalidates its endpoint. Confirm absence using Azure and
an endpoint check. Unrelated administrator/visitor passwords need no rotation if
they were not exposed. Credentials must be delivered to visitors privately by
the owner; no credentials belong in the Cloud tab or reports.

To stop compute, deallocate the existing VM through Azure and verify
`PowerState/deallocated`; an OS shutdown alone can keep compute billable. For
full teardown, first preserve needed evidence and private backups, then have the
owner explicitly approve deletion of the deployment resource group, OS disk,
public IP and any remaining transfer storage. Verify no billable resources remain.
Do not delete unrelated subscription resources or raise spending controls.
