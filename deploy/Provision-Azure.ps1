# STAGE 2 ONLY. No action without -Apply and a private parameter file reflecting approval.
param(
    [Parameter(Mandatory=$true)][string]$Subscription,
    [Parameter(Mandatory=$true)][string]$ResourceGroup,
    [Parameter(Mandatory=$true)][string]$ParametersFile,
    [Parameter(Mandatory=$true)][DateTimeOffset]$ShutdownUtc,
    [Parameter(Mandatory=$true)][string]$AlertEmail,
    [int]$BudgetUsd = 150,
    [switch]$Apply
)
$ErrorActionPreference = 'Stop'
if (-not $Apply) { Write-Output 'Plan only. No resources created. Review deploy/README.md and obtain account/cost/deadline approval.'; exit 0 }
if ($ShutdownUtc -le [DateTimeOffset]::UtcNow.AddHours(1)) { throw 'Shutdown deadline must be approved and at least one hour in the future.' }
$p = Get-Content -LiteralPath $ParametersFile -Raw | ConvertFrom-Json
if ($p.parameters.operatorCidr.value -notmatch '^\d{1,3}(\.\d{1,3}){3}/32$') { throw 'Use the operator IPv4 /32, never public RDP/SSH.' }
az account set --subscription $Subscription
if ($LASTEXITCODE) { throw 'Azure account access failed' }
$p = Get-Content -LiteralPath $ParametersFile -Raw | ConvertFrom-Json
$vmSize = $p.parameters.vmSize.value
if (-not $vmSize) { throw 'Set vmSize explicitly in the parameter file.' }
$sku = az vm list-skus --location eastus --resource-type virtualMachines --size $vmSize --all -o json | ConvertFrom-Json
if ($LASTEXITCODE -or -not $sku -or @(($sku | Where-Object name -eq $vmSize)[0].restrictions).Count) { throw 'SKU unavailable/restricted. Do not silently substitute region or size.' }
# Free-trial subscriptions have small vCPU quotas that can't be raised without upgrading. The VM needs 4 free
# vCPUs in both the regional total and its size family; check before creating anything.
$usage = az vm list-usage --location eastus -o json | ConvertFrom-Json
if ($LASTEXITCODE -or -not $usage) { throw 'Could not read vCPU quotas.' }
foreach ($q in @(@{ Label='Total Regional vCPUs'; Match={ $_.name.value -eq 'cores' } }, @{ Label="$(($sku | Where-Object name -eq $vmSize)[0].family) vCPUs"; Match={ $_.name.value -eq ($sku | Where-Object name -eq $vmSize)[0].family } })) {
    $u = @($usage | Where-Object $q.Match)[0]
    if (-not $u) { throw "Quota '$($q.Label)' not found. Check Usage + quotas in the portal; do not substitute a size silently." }
    if (([int]$u.limit - [int]$u.currentValue) -lt 4) { throw "Quota '$($q.Label)': $($u.currentValue) of $($u.limit) used; $vmSize needs 4. Do not substitute a size silently." }
}
$shutdownTag = 'approvedShutdown=' + $ShutdownUtc.ToUniversalTime().ToString('o')
az group create --name $ResourceGroup --location eastus --tags project=LeanMoE $shutdownTag --output none
if ($LASTEXITCODE) { throw 'Resource group creation failed' }
az deployment group create --resource-group $ResourceGroup --template-file "$PSScriptRoot/main.bicep" --parameters "@$ParametersFile" --output none
if ($LASTEXITCODE) { throw 'Provisioning failed. Inspect/delete partially created resources; do not blindly retry.' }
# Subscription-wide cost budget with e-mail alerts. Quarterly, so costs that span two calendar months
# accumulate instead of resetting on the 1st. Alerts only: a budget never stops resources.
$budgetBody = @{ properties = @{
    category = 'Cost'; amount = $BudgetUsd; timeGrain = 'Quarterly'
    timePeriod = @{ startDate = '2026-10-01T00:00:00Z'; endDate = '2027-09-30T00:00:00Z' }
    notifications = @{
        actual50 = @{ enabled = $true; operator = 'GreaterThanOrEqualTo'; threshold = 50; thresholdType = 'Actual'; contactEmails = @($AlertEmail) }
        actual80 = @{ enabled = $true; operator = 'GreaterThanOrEqualTo'; threshold = 80; thresholdType = 'Actual'; contactEmails = @($AlertEmail) }
        actual100 = @{ enabled = $true; operator = 'GreaterThanOrEqualTo'; threshold = 100; thresholdType = 'Actual'; contactEmails = @($AlertEmail) }
        forecast100 = @{ enabled = $true; operator = 'GreaterThanOrEqualTo'; threshold = 100; thresholdType = 'Forecasted'; contactEmails = @($AlertEmail) }
    } } }
$budgetFile = [IO.Path]::GetTempFileName()
try {
    $budgetBody | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $budgetFile -Encoding UTF8
    az rest --method put --url "https://management.azure.com/subscriptions/$Subscription/providers/Microsoft.Consumption/budgets/leanmoe-subscription-budget?api-version=2024-08-01" --body "@$budgetFile" --headers Content-Type=application/json --output none
    if ($LASTEXITCODE) { throw 'Budget creation failed. Deallocate the VM before doing further work.' }
} finally { Remove-Item -LiteralPath $budgetFile -ErrorAction SilentlyContinue }
& "$PSScriptRoot/Schedule-Shutdown.ps1" -Subscription $Subscription -ResourceGroup $ResourceGroup -VmName $p.parameters.vmName.value -ShutdownUtc $ShutdownUtc
if ($LASTEXITCODE) { throw 'Shutdown scheduling failed. Deallocate immediately before doing further work.' }
Write-Output 'VM provisioned. Keep guest firewall 80/443 closed until backend tests, store verification and proxy validation pass.'
