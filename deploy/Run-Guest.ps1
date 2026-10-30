# Supervise only children this script starts. Both roles stop at the approved deadline.
param([ValidateSet('backend','proxy')][string]$Role, [string]$RepoRoot='C:\LeanMoE', [string]$PythonExe='C:\LeanMoE\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$cfg = Get-Content -LiteralPath 'C:\LeanMoE-private\site.json' -Raw | ConvertFrom-Json
if (($cfg | ConvertTo-Json -Depth 5) -match 'CHOOSE-|GENERATE_|SET_APPROVED|OPERATOR_EMAIL') { throw 'Unconfigured deployment; refusing to start.' }
$deadline = [DateTimeOffset]::Parse($cfg.shutdownUtc).ToUniversalTime()
$env:LEANMOE_PROXY_SECRET = $cfg.proxySecret
$env:LEANMOE_VISITOR_ACCOUNTS = $cfg.visitorAccounts -join ','
$env:LEANMOE_PUBLIC_HOST = $cfg.publicHost
$env:LEANMOE_ACME_EMAIL = $cfg.acmeEmail
$env:OPENBLAS_NUM_THREADS = '4'
$env:NUMBA_NUM_THREADS = '4'
$version = Join-Path $RepoRoot 'DEPLOYED_VERSION.txt'
if (Test-Path $version) {
    $record = Get-Content $version -Raw
    if ($record -match 'source_commit:\s*([0-9a-f]{40})') { $env:LEANMOE_DEPLOYED_COMMIT = $Matches[1] }
}
elseif (Test-Path (Join-Path $RepoRoot '.git')) {
    $commit = & git -C $RepoRoot rev-parse HEAD
    if ($LASTEXITCODE -eq 0 -and $commit -match '^[0-9a-f]{40}$') { $env:LEANMOE_DEPLOYED_COMMIT = $commit }
}
$bundle = & $PythonExe -m certifi
if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $bundle)) { throw 'Verified certificate bundle unavailable; run Install-Prerequisites.ps1.' }
$env:SSL_CERT_FILE = $bundle
$env:REQUESTS_CA_BUNDLE = $bundle
$env:NUMBA_CACHE_DIR = 'C:\LeanMoE-runtime\numba'
$env:XDG_DATA_HOME = 'C:\LeanMoE-runtime\caddy'
$env:TEMP = 'C:\LeanMoE-runtime\tmp'
$env:TMP = $env:TEMP
Set-Location -LiteralPath $RepoRoot
$child = $null
try {
    if ($Role -eq 'backend') {
        $durable = [IO.Path]::GetFullPath($cfg.durableStore)
        $fast = [IO.Path]::GetFullPath($cfg.fastStore)
        if (-not $durable.StartsWith('C:\LeanMoE-models\', [StringComparison]::OrdinalIgnoreCase) -or -not $fast.StartsWith('D:\LeanMoE-models\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected store roots; refusing to copy.' }
        if ((Get-Volume -DriveLetter D).FileSystem -ne 'NTFS') { throw 'The fast store must be on the Windows NTFS temporary disk. Never format a disk automatically.' }
        & $PythonExe deploy/store_manifest.py verify --store-root $durable --manifest $cfg.manifest
        if ($LASTEXITCODE) { throw 'Durable store verification failed' }
        robocopy $durable $fast /E /R:2 /W:2 /COPY:DAT /DCOPY:DAT /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw 'Fast-store copy failed' }
        & $PythonExe deploy/store_manifest.py verify --store-root $fast --manifest $cfg.manifest
        if ($LASTEXITCODE) { throw 'Fast store verification failed' }
        $env:LEANMOE_MODELS_ROOT = Split-Path -Parent $fast
    }
    $restarts = 0
    while ([DateTimeOffset]::UtcNow -lt $deadline -and $restarts -lt 5) {
        if ($Role -eq 'backend') {
            $child = Start-Process -FilePath $PythonExe -ArgumentList '-m','src.dashboard','--execution-location','server','--public-origin',("https://"+$cfg.publicHost) -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
        } else {
            $child = Start-Process -FilePath 'C:\LeanMoE-tools\caddy.exe' -ArgumentList 'run','--config',"$RepoRoot\deploy\Caddyfile",'--adapter','caddyfile' -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
        }
        while (-not $child.WaitForExit(1000) -and [DateTimeOffset]::UtcNow -lt $deadline) { }
        if (-not $child.HasExited) { taskkill.exe /PID $child.Id /T /F | Out-Null }
        $restarts++
        Start-Sleep -Seconds 2
    }
} finally {
    if ($child -and -not $child.HasExited) { taskkill.exe /PID $child.Id /T /F | Out-Null }
}
