# Run only inside the approved, clean Windows VM, after transferring store/notices/manifest.
# Installs no cloud resources. Python and Caddy must be obtained from verified official releases.
param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [Parameter(Mandatory=$true)][string]$CaddyExe,
    [Parameter(Mandatory=$true)][string]$CaddySha256,
    [string]$RepoRoot='C:\LeanMoE'
)
$ErrorActionPreference = 'Stop'
& "$PSScriptRoot/Install-Prerequisites.ps1" -PythonExe $PythonExe
if ((Get-FileHash -LiteralPath $CaddyExe -Algorithm SHA256).Hash -ne $CaddySha256) { throw 'Caddy checksum mismatch' }
if ((Get-CimInstance Win32_OperatingSystem).Caption -notmatch 'Windows Server') { throw 'This script is for the approved VM, not the laptop.' }
if (-not (Test-Path 'C:\LeanMoE-private\site.json') -or -not (Test-Path 'C:\LeanMoE-private\visitors.caddy')) { throw 'Create the private site config and individual visitor password hashes first.' }
Set-Location -LiteralPath $RepoRoot
$cfg = Get-Content -LiteralPath 'C:\LeanMoE-private\site.json' -Raw | ConvertFrom-Json
# A git checkout must be clean; an archive must match the SHA-256 recorded when it was built.
if (Test-Path '.git') { if (git status --porcelain) { throw 'Install from a clean, approved checkout.' } }
else { throw 'Use a clean Git checkout for initial install; use Publish-Guest.ps1 for pinned GitHub archives.' }
New-Item -ItemType Directory -Force -Path 'C:\LeanMoE-tools','C:\LeanMoE-runtime','C:\LeanMoE-runtime\tmp','C:\LeanMoE-runtime\numba','D:\LeanMoE-models' | Out-Null
Copy-Item -LiteralPath $CaddyExe -Destination 'C:\LeanMoE-tools\caddy.exe'
# Private VM config contains only runtime secrets; never put Azure admin parameter files here.
icacls 'C:\LeanMoE-private' /inheritance:r /grant:r '*S-1-5-32-544:(OI)(CI)F' '*S-1-5-18:(OI)(CI)F' '*S-1-5-19:(OI)(CI)R' | Out-Null
icacls $RepoRoot /grant '*S-1-5-19:(OI)(CI)RX' | Out-Null
icacls 'C:\LeanMoE-tools' /grant '*S-1-5-19:(OI)(CI)RX' | Out-Null
icacls $cfg.durableStore /grant '*S-1-5-19:(OI)(CI)RX' | Out-Null
icacls 'C:\LeanMoE-runtime' /grant '*S-1-5-19:(OI)(CI)M' | Out-Null
icacls 'D:\LeanMoE-models' /grant '*S-1-5-19:(OI)(CI)M' | Out-Null
& $PythonExe -m venv .venv
if ($LASTEXITCODE) { throw 'venv failed' }
& '.\.venv\Scripts\python.exe' -m pip install -r deploy/requirements-cloud.txt
if ($LASTEXITCODE) { throw 'dependencies failed' }
& '.\.venv\Scripts\python.exe' -m pytest -q
if ($LASTEXITCODE) { throw 'tests failed' }
$cfg = Get-Content -LiteralPath 'C:\LeanMoE-private\site.json' -Raw | ConvertFrom-Json
& '.\.venv\Scripts\python.exe' deploy/store_manifest.py verify --store-root $cfg.durableStore --manifest $cfg.manifest
if ($LASTEXITCODE) { throw 'store integrity failed' }
$env:LEANMOE_MODELS_ROOT = Split-Path -Parent $cfg.durableStore
& '.\.venv\Scripts\python.exe' scripts/preflight.py --require-model
if ($LASTEXITCODE) { throw 'preflight failed' }
$env:LEANMOE_PROXY_SECRET=$cfg.proxySecret; $env:LEANMOE_PUBLIC_HOST=$cfg.publicHost; $env:LEANMOE_ACME_EMAIL=$cfg.acmeEmail
& 'C:\LeanMoE-tools\caddy.exe' validate --config deploy/Caddyfile --adapter caddyfile
if ($LASTEXITCODE) { throw 'proxy validation failed' }
$principal = New-ScheduledTaskPrincipal -UserId 'NT AUTHORITY\LOCAL SERVICE' -LogonType ServiceAccount -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
foreach ($role in @('backend','proxy')) {
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$RepoRoot\deploy\Run-Guest.ps1`" -Role $role -RepoRoot `"$RepoRoot`""
    Register-ScheduledTask -TaskName "LeanMoE-$role" -Action $action -Trigger (New-ScheduledTaskTrigger -AtStartup) -Principal $principal -Settings $settings | Out-Null
}
Start-ScheduledTask -TaskName 'LeanMoE-backend'
Start-ScheduledTask -TaskName 'LeanMoE-proxy'
Write-Output 'Tasks started. Verify readiness/TLS/visitor isolation before opening guest firewall ports 80/443. Do not expose 8770.'
