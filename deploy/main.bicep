targetScope = 'resourceGroup'

param location string = 'eastus'
param vmName string = 'leanmoe-demo-1'
@description('Approved deployment: 4 vCPU, 16 GB RAM and local temporary SSD. Recheck account/region availability before provisioning.')
param vmSize string = 'Standard_D4ds_v4'
param dnsLabel string
param adminUsername string
@secure()
param adminPassword string
@description('Only the operator public IPv4 /32; never 0.0.0.0/0')
param operatorCidr string

resource nsg 'Microsoft.Network/networkSecurityGroups@2024-05-01' = {
  name: '${vmName}-nsg'
  location: location
  properties: {
    securityRules: [
      {
        name: 'https-and-acme'
        properties: {
          priority: 100
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['80', '443']
        }
      }
      {
        name: 'operator-only'
        properties: {
          priority: 110
          direction: 'Inbound'
          access: 'Allow'
          protocol: 'Tcp'
          sourceAddressPrefix: operatorCidr
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRanges: ['22', '3389']
        }
      }
    ]
  }
}
resource vnet 'Microsoft.Network/virtualNetworks@2024-05-01' = {
  name: '${vmName}-vnet'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.42.0.0/16'] }
    subnets: [{ name: 'app', properties: { addressPrefix: '10.42.0.0/24', networkSecurityGroup: { id: nsg.id } } }]
  }
}
resource ip 'Microsoft.Network/publicIPAddresses@2024-05-01' = {
  name: '${vmName}-ip'
  location: location
  sku: { name: 'Standard' }
  properties: { publicIPAllocationMethod: 'Static', dnsSettings: { domainNameLabel: dnsLabel } }
}
resource nic 'Microsoft.Network/networkInterfaces@2024-05-01' = {
  name: '${vmName}-nic'
  location: location
  properties: {
    ipConfigurations: [{ name: 'primary', properties: {
      privateIPAllocationMethod: 'Dynamic'
      subnet: { id: '${vnet.id}/subnets/app' }
      publicIPAddress: { id: ip.id }
    } }]
  }
}
resource vm 'Microsoft.Compute/virtualMachines@2024-07-01' = {
  name: vmName
  location: location
  properties: {
    hardwareProfile: { vmSize: vmSize }
    storageProfile: {
      imageReference: { publisher: 'MicrosoftWindowsServer', offer: 'WindowsServer', sku: '2022-datacenter-azure-edition', version: 'latest' }
      osDisk: { createOption: 'FromImage', diskSizeGB: 128, managedDisk: { storageAccountType: 'Premium_LRS' }, deleteOption: 'Delete' }
    }
    osProfile: { computerName: vmName, adminUsername: adminUsername, adminPassword: adminPassword }
    networkProfile: { networkInterfaces: [{ id: nic.id, properties: { primary: true } }] }
  }
}
resource automation 'Microsoft.Automation/automationAccounts@2024-10-23' = {
  name: '${vmName}-stop'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: { sku: { name: 'Basic' }, publicNetworkAccess: true }
}
resource stopRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vm.id, automation.id, 'stop')
  scope: vm
  properties: {
    principalId: automation.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '9980e02c-c2be-4d73-94e8-173b1dc7cf3c')
  }
}
// The cost budget and its alerts are created at subscription scope by Provision-Azure.ps1.
output publicHost string = ip.properties.dnsSettings.fqdn
output vmId string = vm.id
output automationName string = automation.name
