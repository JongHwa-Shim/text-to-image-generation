# Worker Node Initial Setup Script (Run once on each worker)

param(
    [string]$MasterIP = "",
    [switch]$Help
)

# Color output functions
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    $colors = @{ "Red" = "Red"; "Green" = "Green"; "Yellow" = "Yellow"; "Blue" = "Blue"; "White" = "White" }
    Write-Host $Message -ForegroundColor $colors[$Color]
}

function Log-Info { param([string]$Message); Write-ColorOutput "[INFO] $Message" "Blue" }
function Log-Warn { param([string]$Message); Write-ColorOutput "[WARN] $Message" "Yellow" }
function Log-Error { param([string]$Message); Write-ColorOutput "[ERROR] $Message" "Red" }
function Log-Success { param([string]$Message); Write-ColorOutput "[SUCCESS] $Message" "Green" }

function Show-Help {
    Write-Host @"
Worker Node Initial Setup Script

This script needs to be run only once on each worker node.
After that, all deployment and training will be handled automatically from the master node.

Usage:
    .\scripts\worker_setup_en.ps1 [-MasterIP IP_ADDRESS]

Parameters:
    -MasterIP IP        Master node IP address (for trust relationship)
    -Help              Show this help message

Examples:
    .\scripts\worker_setup_en.ps1 -MasterIP "192.168.1.100"
    .\scripts\worker_setup_en.ps1  # Interactive mode
"@
}

function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Enable-PowerShellRemoting {
    Log-Info "Setting up PowerShell Remoting..."
    
    try {
        # Enable PowerShell Remoting
        $result = Enable-PSRemoting -Force -SkipNetworkProfileCheck 2>&1
        
        # Configure WinRM
        winrm quickconfig -q 2>&1 | Out-Null
        Set-WSManInstance -ResourceURI winrm/config/service -ValueSet @{EnableCompatibilityHttpListener=$true} -ErrorAction SilentlyContinue
        
        Log-Success "PowerShell Remoting enabled successfully"
        return $true
    }
    catch {
        Log-Error "PowerShell Remoting setup failed: $($_.Exception.Message)"
        return $false
    }
}

function Set-FirewallRules {
    Log-Info "Configuring firewall rules..."
    
    try {
        # Allow PyTorch Distributed port (29500)
        $ruleName = "PyTorch Distributed Training"
        $existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
        
        if ($existingRule) {
            Log-Info "Firewall rule already exists, updating..."
            Set-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort 29500 -Action Allow
        } else {
            Log-Info "Creating new firewall rule..."
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort 29500 -Action Allow | Out-Null
        }
        
        # Allow PowerShell Remoting
        Enable-NetFirewallRule -DisplayGroup "Windows Remote Management" -ErrorAction SilentlyContinue
        
        Log-Success "Firewall rules configured successfully"
        return $true
    }
    catch {
        Log-Error "Firewall configuration failed: $($_.Exception.Message)"
        return $false
    }
}

function Set-TrustedHosts {
    param([string]$MasterIP)
    
    if ([string]::IsNullOrEmpty($MasterIP)) {
        Log-Warn "No master IP provided, skipping trusted hosts configuration"
        return $true
    }
    
    Log-Info "Setting up trusted hosts for IP: $MasterIP"
    
    try {
        $currentTrustedHosts = (Get-Item WSMan:\localhost\Client\TrustedHosts).Value
        
        if ([string]::IsNullOrEmpty($currentTrustedHosts) -or $currentTrustedHosts -eq "*") {
            Set-Item WSMan:\localhost\Client\TrustedHosts -Value $MasterIP -Force
        } else {
            if ($currentTrustedHosts -notlike "*$MasterIP*") {
                $newTrustedHosts = "$currentTrustedHosts,$MasterIP"
                Set-Item WSMan:\localhost\Client\TrustedHosts -Value $newTrustedHosts -Force
            }
        }
        
        Log-Success "Trusted hosts configured successfully"
        return $true
    }
    catch {
        Log-Error "Trusted hosts configuration failed: $($_.Exception.Message)"
        return $false
    }
}

function Optimize-PowerSettings {
    Log-Info "Optimizing power settings for distributed training..."
    
    try {
        # Set to High Performance power plan
        $highPerfGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
        powercfg /setactive $highPerfGuid 2>&1 | Out-Null
        
        # Disable USB selective suspend
        powercfg /change usb-selective-suspend-setting 0 2>&1 | Out-Null
        
        # Set disk timeout to never
        powercfg /change disk-timeout-ac 0 2>&1 | Out-Null
        powercfg /change disk-timeout-dc 0 2>&1 | Out-Null
        
        Log-Success "Power settings optimized successfully"
        return $true
    }
    catch {
        Log-Warn "Power settings optimization failed, continuing anyway..."
        return $true
    }
}

function Test-GPUAvailability {
    Log-Info "Checking GPU availability..."
    
    try {
        $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" -or $_.Name -like "*AMD*" -or $_.Name -like "*Intel*" }
        $gpuCount = $gpuInfo.Count
        
        if ($gpuCount -gt 0) {
            Log-Success "Found $gpuCount GPU(s):"
            foreach ($gpu in $gpuInfo) {
                Log-Info "  - $($gpu.Name)"
            }
        } else {
            Log-Warn "No dedicated GPUs found, will use CPU for training"
        }
        
        return $gpuCount
    }
    catch {
        Log-Warn "GPU detection failed: $($_.Exception.Message)"
        return 0
    }
}

function Test-PythonInstallation {
    Log-Info "Checking Python installation..."
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Log-Success "Python found: $pythonVersion"
            return $true
        } else {
            Log-Warn "Python not found in PATH, will need to be installed"
            return $false
        }
    }
    catch {
        Log-Warn "Python check failed, will need to be installed"
        return $false
    }
}

function Test-NetworkConnectivity {
    param([string]$MasterIP)
    
    if ([string]::IsNullOrEmpty($MasterIP)) {
        return $true
    }
    
    Log-Info "Testing network connectivity to master: $MasterIP"
    
    try {
        $pingResult = Test-Connection -ComputerName $MasterIP -Count 2 -Quiet
        if ($pingResult) {
            Log-Success "Network connectivity to master OK"
            
            # Test specific port 29500
            $tcpTest = Test-NetConnection -ComputerName $MasterIP -Port 29500 -WarningAction SilentlyContinue
            if ($tcpTest.TcpTestSucceeded) {
                Log-Success "Port 29500 connectivity OK"
            } else {
                Log-Warn "Port 29500 not accessible (may be normal if master is not running)"
            }
        } else {
            Log-Error "Cannot reach master node at $MasterIP"
            return $false
        }
        
        return $true
    }
    catch {
        Log-Error "Network connectivity test failed: $($_.Exception.Message)"
        return $false
    }
}

function Get-SystemInfo {
    Log-Info "Gathering system information..."
    
    try {
        $computerInfo = Get-ComputerInfo
        $systemInfo = @{
            ComputerName = $env:COMPUTERNAME
            OperatingSystem = $computerInfo.WindowsProductName
            TotalPhysicalMemory = [math]::Round($computerInfo.TotalPhysicalMemory / 1GB, 2)
            ProcessorName = $computerInfo.CsProcessors[0].Name
            IPAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" -or $_.IPAddress -like "172.*" }).IPAddress[0]
        }
        
        Log-Success "System information gathered"
        return $systemInfo
    }
    catch {
        Log-Error "Failed to gather system information: $($_.Exception.Message)"
        return @{}
    }
}

function Save-WorkerInfo {
    param($SystemInfo, $GPUCount)
    
    Log-Info "Saving worker information..."
    
    try {
        $workerInfo = @{
            ComputerName = $SystemInfo.ComputerName
            IPAddress = $SystemInfo.IPAddress
            OperatingSystem = $SystemInfo.OperatingSystem
            TotalMemoryGB = $SystemInfo.TotalPhysicalMemory
            ProcessorName = $SystemInfo.ProcessorName
            GPUCount = $GPUCount
            SetupDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            Status = "Ready"
        }
        
        $workerInfo | ConvertTo-Json | Out-File "worker_info.json" -Encoding UTF8
        Log-Success "Worker information saved to worker_info.json"
    }
    catch {
        Log-Error "Failed to save worker information: $($_.Exception.Message)"
    }
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

# Check admin rights
if (-not (Test-AdminRights)) {
    Log-Error "This script must be run as Administrator"
    Log-Info "Please right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Distributed Training Worker Setup  " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Get master IP if not provided
if ([string]::IsNullOrEmpty($MasterIP)) {
    Write-Host "You can skip this by pressing Enter (optional setting)"
    $MasterIP = Read-Host "Enter master node IP address (optional, press Enter to skip)"
}

# System information gathering
$systemInfo = Get-SystemInfo
if ($systemInfo.Count -gt 0) {
    Write-Host ""
    Log-Info "Worker Node: $($systemInfo.ComputerName) ($($systemInfo.IPAddress))"
    Log-Info "OS: $($systemInfo.OperatingSystem)"
    Log-Info "Memory: $($systemInfo.TotalPhysicalMemory) GB"
    Log-Info "CPU: $($systemInfo.ProcessorName)"
}

# Setup steps
$setupSteps = @(
    @{ Name = "PowerShell Remoting"; Function = { Enable-PowerShellRemoting } },
    @{ Name = "Firewall Rules"; Function = { Set-FirewallRules } },
    @{ Name = "Trusted Hosts"; Function = { Set-TrustedHosts -MasterIP $MasterIP } },
    @{ Name = "Power Settings"; Function = { Optimize-PowerSettings } }
)

Write-Host ""
Log-Info "Starting worker setup process..."
$allSuccess = $true

foreach ($step in $setupSteps) {
    Write-Host ""
    Log-Info "Executing: $($step.Name)..."
    
    try {
        $result = & $step.Function
        if (-not $result) {
            Log-Error "$($step.Name) failed"
            $allSuccess = $false
        }
    }
    catch {
        Log-Error "$($step.Name) failed with exception: $($_.Exception.Message)"
        $allSuccess = $false
    }
}

# GPU and Python check
Write-Host ""
$gpuCount = Test-GPUAvailability
$pythonOK = Test-PythonInstallation

# Network connectivity test
if (-not [string]::IsNullOrEmpty($MasterIP)) {
    Write-Host ""
    Test-NetworkConnectivity -MasterIP $MasterIP | Out-Null
}

# Save worker information
Write-Host ""
Save-WorkerInfo -SystemInfo $systemInfo -GPUCount $gpuCount

# Final results
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "           Setup Results              " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

if ($allSuccess) {
    Log-Success "Worker node setup completed successfully!"
    Log-Success "Worker: $($systemInfo.ComputerName) ($($systemInfo.IPAddress))"
    
    if ($gpuCount -gt 0) {
        Log-Success "GPUs available: $gpuCount"
    } else {
        Log-Warn "No GPUs detected - will use CPU training"
    }
    
    if ($pythonOK) {
        Log-Success "Python installation: OK"
    } else {
        Log-Warn "Python not found - will be installed during deployment"
    }
    
    Write-Host ""
    Log-Info "Next steps:"
    Log-Info "1. Run this script on all other worker nodes"
    Log-Info "2. From master node, run: .\scripts\master_deploy.ps1"
    Log-Info "3. Distributed training will start automatically"
    
} else {
    Log-Error "Worker setup completed with some failures"
    Log-Info "Please review the errors above and retry if necessary"
    Log-Info "Some features may not work properly during distributed training"
}

Write-Host ""
Write-Host "Worker setup script finished." -ForegroundColor Cyan
