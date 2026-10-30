# Run elevated on the VM. Downloads use normal Windows HTTPS verification.
param([Parameter(Mandatory=$true)][string]$PythonExe)
$ErrorActionPreference = 'Stop'
if ((Get-CimInstance Win32_OperatingSystem).Caption -notmatch 'Windows Server') { throw 'Cloud guest only.' }
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$installer = Join-Path $env:TEMP 'leanmoe-vc-redist-x64.exe'
Invoke-WebRequest 'https://aka.ms/vc14/vc_redist.x64.exe' -OutFile $installer -UseBasicParsing
$signature = Get-AuthenticodeSignature -LiteralPath $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') { throw 'Microsoft installer signature verification failed.' }
$run = Start-Process -FilePath $installer -ArgumentList '/install','/quiet','/norestart' -WindowStyle Hidden -Wait -PassThru
if ($run.ExitCode -eq 3010) { throw 'Runtime installed; reboot required before continuing.' }
if ($run.ExitCode -notin 0,1638) { throw "VC++ runtime installation failed: $($run.ExitCode)" }
# pip verifies PyPI TLS with its normal trust configuration; never use trusted-host.
& $PythonExe -m pip install --disable-pip-version-check --index-url https://pypi.org/simple certifi==2026.10.23
if ($LASTEXITCODE) { throw 'Certificate bundle installation failed.' }
$bundle = & $PythonExe -m certifi
if ($LASTEXITCODE -or -not (Test-Path -LiteralPath $bundle)) { throw 'Certificate bundle missing.' }
$env:SSL_CERT_FILE = $bundle
$env:REQUESTS_CA_BUNDLE = $bundle
& $PythonExe -c "import ssl,urllib.request; c=ssl.create_default_context(); r=urllib.request.urlopen('https://huggingface.co',context=c,timeout=20); assert r.status==200; print('Verified Python HTTPS trust')"
if ($LASTEXITCODE) { throw 'Certificate verification failed; do not bypass it.' }
