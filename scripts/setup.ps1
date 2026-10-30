# LeanMoE setup: create .venv in the repository, install dependencies, run preflight.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1 -WithToy      (also install torch for the toy fallback)
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1 -Python "C:\path\to\python.exe"
#
# Safe to rerun: an existing .venv is reused and packages already installed are kept.
# It never downloads model files and never changes system settings. "-ExecutionPolicy Bypass"
# above applies only to that one PowerShell process.

param(
    [switch]$WithToy,
    [string]$Python = ""
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "SETUP FAILED: $Message" -ForegroundColor Red
    exit 1
}

Write-Host "LeanMoE setup in $RepoRoot"

if ($env:OS -ne "Windows_NT") {
    Fail "LeanMoE runs on Windows only (its engine uses Win32 unbuffered file reads)."
}

# Find a Python 3.10-3.13 interpreter (3.13 is the tested version).
$candidates = @()
if ($Python -ne "") { $candidates += ,@($Python) }
foreach ($v in @("3.13", "3.12", "3.11", "3.10")) { $candidates += ,@("py", "-$v") }
$candidates += ,@("python")

$chosen = $null
foreach ($c in $candidates) {
    $exe = $c[0]
    $pre = @()
    if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $ver = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $ver) { continue }
    $ver = "$ver".Trim()
    if (@("3.10", "3.11", "3.12", "3.13") -contains $ver) {
        $chosen = @{ Exe = $exe; Pre = $pre; Version = $ver }
        break
    }
}
if (-not $chosen) {
    Fail "No Python 3.10-3.13 found. Install Python 3.13 from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH' or use the 'py' launcher), then rerun this script."
}
Write-Host "Using Python $($chosen.Version): $($chosen.Exe) $($chosen.Pre -join ' ')"
if ($chosen.Version -ne "3.13") {
    Write-Host "NOTE: LeanMoE was tested with Python 3.13; $($chosen.Version) is untested." -ForegroundColor Yellow
}

if ($WithToy -and $RepoRoot.Length -gt 80) {
    Write-Host "NOTE: the repository path is long ($($RepoRoot.Length) characters). torch has deeply nested files; if its install fails with a path-length error, clone into a shorter folder such as C:\LeanMoE." -ForegroundColor Yellow
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Host "Reusing existing virtual environment .venv"
} else {
    Write-Host "Creating virtual environment .venv"
    & $chosen.Exe @($chosen.Pre) -m venv .venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Fail "Could not create .venv. Delete the .venv folder if it exists and rerun."
    }
}

Write-Host "Installing dependencies (requirements-dev.txt: engine packages and pytest)"
& $VenvPython -m pip install --disable-pip-version-check -q -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) {
    Fail "pip could not install requirements-dev.txt (see the messages above). Check the internet connection and rerun."
}
if ($WithToy) {
    Write-Host "Installing the optional toy dependency (requirements-toy.txt: torch, large download)"
    & $VenvPython -m pip install --disable-pip-version-check -q -r requirements-toy.txt
    if ($LASTEXITCODE -ne 0) {
        Fail "pip could not install requirements-toy.txt. If the error mentions a path length, clone into a shorter folder."
    }
}

Write-Host ""
& $VenvPython scripts\preflight.py
$pre = $LASTEXITCODE
Write-Host ""
if ($pre -ne 0) {
    Fail "Preflight found a problem (see PROBLEM lines above)."
}
Write-Host "Setup complete. Next:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m pytest"
Write-Host "  .\.venv\Scripts\python.exe eval\compare_artifacts.py results\adaptive_demo_20261030-060626.json results\adaptive_demo_20261030-060728.json"
exit 0
