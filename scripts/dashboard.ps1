# Start the LeanMoE dashboard on this computer only (http://127.0.0.1:8770).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dashboard.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\dashboard.ps1 -Port 8771 -NoOpen
#
# Uses the .venv that scripts\setup.ps1 creates (note the dot). Stop with Ctrl+C.
# For the real model, set $env:LEANMOE_MODELS_ROOT in the same window before starting.

param(
    [int]$Port = 8770,
    [ValidateSet("local", "server")]
    [string]$ExecutionLocation = "local",
    [switch]$NoOpen
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "No .venv in $RepoRoot. Run the setup first:" -ForegroundColor Red
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1"
    exit 1
}
$dashArgs = @("-m", "src.dashboard", "--port", "$Port", "--execution-location", $ExecutionLocation)
if (-not $NoOpen) { $dashArgs += "--open" }
& $VenvPython @dashArgs
exit $LASTEXITCODE
