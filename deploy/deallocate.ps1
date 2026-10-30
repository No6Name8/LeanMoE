# Azure Automation runbook: managed identity is scoped to this VM only.
param([Parameter(Mandatory=$true)][string]$VmResourceId)
$ErrorActionPreference = 'Stop'
$headers = @{ 'X-IDENTITY-HEADER' = $env:IDENTITY_HEADER; 'Metadata' = 'True' }
$token = Invoke-RestMethod -Uri ($env:IDENTITY_ENDPOINT + '?resource=https://management.azure.com/') -Headers $headers
$auth = @{ Authorization = 'Bearer ' + $token.access_token }
Invoke-RestMethod -Method Post -Uri ('https://management.azure.com' + $VmResourceId + '/deallocate?api-version=2024-07-01') -Headers $auth
Write-Output 'Deallocation requested. Operator must confirm PowerState/deallocated; disks and reserved IP remain billable.'
