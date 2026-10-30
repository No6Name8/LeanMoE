# Publish an immutable GitHub commit on the existing VM; creates no Azure resources.
param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$Commit,
      [string]$PythonExe='C:\LeanMoE\.venv\Scripts\python.exe', [switch]$Apply)
$ErrorActionPreference = 'Stop'
if (-not $Apply) { Write-Output 'Plan only: download exact commit, verify/tests, then switch supervised roles.'; exit 0 }
if ((Get-CimInstance Win32_OperatingSystem).Caption -notmatch 'Windows Server') { throw 'Cloud guest only.' }
$cfg = Get-Content C:\LeanMoE-private\site.json -Raw | ConvertFrom-Json
if ([DateTimeOffset]::Parse($cfg.shutdownUtc) -le [DateTimeOffset]::UtcNow) { throw 'Approved deadline has passed.' }
$headers = @{ Host='127.0.0.1:8770'; 'X-LeanMoE-Proxy-Key'=$cfg.proxySecret; 'X-LeanMoE-Visitor'=$cfg.visitorAccounts[0] }
$status = Invoke-RestMethod http://127.0.0.1:8770/api/status -Headers $headers
if ($status.busy) { throw 'Engine busy; do not deploy over a visitor job.' }
$releaseBase = 'C:\LeanMoE-releases'
$release = [IO.Path]::GetFullPath((Join-Path $releaseBase ('LeanMoE-'+$Commit)))
if (-not $release.StartsWith($releaseBase+'\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid release destination.' }
if (Test-Path $release) { throw 'Release already exists; inspect it rather than overwrite it.' }
New-Item -ItemType Directory -Force -Path $releaseBase | Out-Null
$archive = Join-Path 'C:\LeanMoE-runtime' ($Commit+'.zip')
# GitHub HTTPS authenticates the origin and the URL pins the source commit.
Invoke-WebRequest ("https://github.com/No6Name8/LeanMoE/archive/$Commit.zip") -OutFile $archive -UseBasicParsing
$archiveHash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Expand-Archive -LiteralPath $archive -DestinationPath $releaseBase
if (-not (Test-Path (Join-Path $release 'src\dashboard\server.py'))) { throw 'Unexpected source archive layout.' }
Set-Content (Join-Path $release 'DEPLOYED_VERSION.txt') "source_commit: $Commit`narchive_sha256: $archiveHash" -Encoding ASCII
icacls $release /grant '*S-1-5-19:(OI)(CI)RX' | Out-Null
Set-Location -LiteralPath $release
. (Join-Path $release 'deploy\Guest-Tools.ps1')
Invoke-GuestTool $PythonExe @('-m','pip','install','--disable-pip-version-check','-r','deploy/requirements-cloud.txt') 'release-install'
Invoke-GuestTool $PythonExe @('-m','pytest','-q') 'release-tests'
$env:LEANMOE_MODELS_ROOT = Split-Path -Parent $cfg.fastStore
Invoke-GuestTool $PythonExe @('scripts/preflight.py','--require-model') 'release-preflight'
$env:LEANMOE_PROXY_SECRET=$cfg.proxySecret; $env:LEANMOE_PUBLIC_HOST=$cfg.publicHost; $env:LEANMOE_ACME_EMAIL=$cfg.acmeEmail
Invoke-GuestTool 'C:\LeanMoE-tools\caddy.exe' @('validate','--config','deploy/Caddyfile','--adapter','caddyfile') 'release-caddy'
# Recheck immediately before switching. Requests during this brief maintenance
# window can fail; the authenticated proxy never serves an unprotected backend.
if ((Invoke-RestMethod http://127.0.0.1:8770/api/status -Headers $headers).busy) { throw 'Engine became busy; retry later.' }
$backup = 'C:\LeanMoE-private\previous-tasks.json'
@('backend','proxy') | ForEach-Object { Get-ScheduledTask -TaskName ('LeanMoE-'+$_) | Select-Object TaskName,Actions } | ConvertTo-Json -Depth 8 | Set-Content $backup
foreach ($role in @('backend','proxy')) { Stop-ScheduledTask -TaskName ('LeanMoE-'+$role) }
Start-Sleep -Seconds 2
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 8770,80,443 | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
    $process = Get-Process -Id $_
    if ($process.ProcessName -notin 'python','python3','caddy') { throw 'Unexpected listener; refusing to stop it.' }
    Stop-Process -Id $process.Id -Force
}
foreach ($role in @('backend','proxy')) {
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$release\deploy\Run-Guest.ps1`" -Role $role -RepoRoot `"$release`" -PythonExe `"$PythonExe`""
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
    Set-ScheduledTask -TaskName ('LeanMoE-'+$role) -Action $action -Settings $settings | Out-Null
    Start-ScheduledTask -TaskName ('LeanMoE-'+$role)
}
Write-Output "Selected commit $Commit. Verify HTTPS status build.commit; use recovery instructions if readiness fails."
