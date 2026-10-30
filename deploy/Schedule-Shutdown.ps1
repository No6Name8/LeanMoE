param([string]$Subscription, [string]$ResourceGroup, [string]$VmName, [DateTimeOffset]$ShutdownUtc)
$ErrorActionPreference = 'Stop'
$root = "https://management.azure.com/subscriptions/$Subscription/resourceGroups/$ResourceGroup"
$account = "$root/providers/Microsoft.Automation/automationAccounts/$VmName-stop"
$vm = "/subscriptions/$Subscription/resourceGroups/$ResourceGroup/providers/Microsoft.Compute/virtualMachines/$VmName"
$api = '?api-version=2024-10-23'
function Put-Json($Url, $Value) {
    $file = [IO.Path]::GetTempFileName()
    try {
        $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $file -Encoding UTF8
        az rest --method put --url $Url --body "@$file" --headers Content-Type=application/json --output none
        if ($LASTEXITCODE) { throw 'Azure shutdown configuration failed' }
    } finally { Remove-Item -LiteralPath $file -ErrorAction SilentlyContinue }
}
Put-Json "$account/runbooks/deallocate$api" @{ location='eastus'; properties=@{ runbookType='PowerShell'; logProgress=$false; logVerbose=$false; draft=@{} } }
az rest --method put --url "$account/runbooks/deallocate/draft/content$api" --body "@$PSScriptRoot/deallocate.ps1" --headers Content-Type=text/powershell --output none
if ($LASTEXITCODE) { throw 'Runbook upload failed' }
az rest --method post --url "$account/runbooks/deallocate/publish$api" --output none
if ($LASTEXITCODE) { throw 'Runbook publish failed' }
Put-Json "$account/schedules/approved-deadline$api" @{ properties=@{ frequency='OneTime'; startTime=$ShutdownUtc.ToUniversalTime().ToString('o'); timeZone='Etc/UTC' } }
$job = [Guid]::NewGuid().ToString()
Put-Json "$account/jobSchedules/$job$api" @{ properties=@{ runbook=@{name='deallocate'}; schedule=@{name='approved-deadline'}; parameters=@{VmResourceId=$vm} } }
