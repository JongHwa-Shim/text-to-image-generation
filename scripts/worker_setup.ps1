# 워커 노드 초기 설정 스크립트 (각 워커에서 1회 실행)

param(
    [string]$MasterIP = "",
    [switch]$Help
)

# 색상 함수
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
워커 노드 초기 설정 스크립트

이 스크립트는 각 워커 노드에서 1회만 실행하면 됩니다.
이후 모든 배포와 훈련은 마스터 노드에서 자동으로 처리됩니다.

사용법:
    .\scripts\worker_setup.ps1 [-MasterIP IP주소]

매개변수:
    -MasterIP IP        마스터 노드의 IP 주소 (신뢰 관계 설정용)
    -Help              이 도움말 표시

예시:
    .\scripts\worker_setup.ps1 -MasterIP "192.168.1.100"
    .\scripts\worker_setup.ps1  # 대화형 모드
"@
}

function Test-AdminRights {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Enable-PowerShellRemoting {
    Log-Info "PowerShell Remoting 설정 중..."
    
    try {
        # PowerShell Remoting 활성화
        Enable-PSRemoting -Force -SkipNetworkProfileCheck | Out-Null
        
        # WinRM 서비스 설정
        Set-Service WinRM -StartupType Automatic
        Start-Service WinRM
        
        # 기본 방화벽 규칙 확인
        Enable-PSRemoting -Force | Out-Null
        
        Log-Success "PowerShell Remoting 활성화 완료"
        return $true
    } catch {
        Log-Error "PowerShell Remoting 설정 실패: $($_.Exception.Message)"
        return $false
    }
}

function Set-TrustedHosts {
    param([string]$MasterIP)
    
    Log-Info "신뢰할 수 있는 호스트 설정 중..."
    
    try {
        if ([string]::IsNullOrEmpty($MasterIP)) {
            # 모든 호스트 신뢰 (개발 환경용)
            Set-Item WSMan:\localhost\Client\TrustedHosts -Value "*" -Force
            Log-Warn "모든 호스트를 신뢰하도록 설정됨 (보안 주의)"
        } else {
            # 특정 마스터 IP만 신뢰
            $currentHosts = Get-Item WSMan:\localhost\Client\TrustedHosts
            if ($currentHosts.Value -eq "" -or $currentHosts.Value -eq $null) {
                Set-Item WSMan:\localhost\Client\TrustedHosts -Value $MasterIP -Force
            } else {
                $hosts = $currentHosts.Value + "," + $MasterIP
                Set-Item WSMan:\localhost\Client\TrustedHosts -Value $hosts -Force
            }
            Log-Success "마스터 노드 $MasterIP 를 신뢰할 수 있는 호스트로 추가"
        }
        return $true
    } catch {
        Log-Error "신뢰할 수 있는 호스트 설정 실패: $($_.Exception.Message)"
        return $false
    }
}

function Set-ExecutionPolicy {
    Log-Info "실행 정책 설정 중..."
    
    try {
        Set-ExecutionPolicy RemoteSigned -Scope LocalMachine -Force
        Log-Success "실행 정책을 RemoteSigned로 설정 완료"
        return $true
    } catch {
        Log-Error "실행 정책 설정 실패: $($_.Exception.Message)"
        return $false
    }
}

function Set-FirewallRules {
    Log-Info "방화벽 규칙 설정 중..."
    
    try {
        # 분산 학습용 포트 열기
        $ports = @(29500, 29501, 29502, 29503, 29504)
        foreach ($port in $ports) {
            $ruleName = "Distributed ML Training - Port $port"
            
            # 기존 규칙 삭제 (있는 경우)
            Remove-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
            
            # 새 규칙 추가
            New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $port -Action Allow | Out-Null
        }
        
        # PowerShell Remoting 포트
        $psRuleName = "Windows Remote Management (PowerShell Remoting)"
        if (-not (Get-NetFirewallRule -DisplayName "*$psRuleName*" -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $psRuleName -Direction Inbound -Protocol TCP -LocalPort 5985,5986 -Action Allow | Out-Null
        }
        
        Log-Success "방화벽 규칙 설정 완료"
        return $true
    } catch {
        Log-Error "방화벽 규칙 설정 실패: $($_.Exception.Message)"
        return $false
    }
}

function Test-GPUAvailability {
    Log-Info "GPU 확인 중..."
    
    try {
        $gpuInfo = nvidia-smi --query-gpu=count,name --format=csv,noheader,nounits 2>$null
        if ($gpuInfo) {
            $gpuCount = ($gpuInfo -split "`n").Count
            Log-Success "GPU $gpuCount개 감지됨"
            $gpuInfo -split "`n" | ForEach-Object { Log-Info "  GPU: $_" }
            return $gpuCount
        } else {
            Log-Warn "GPU를 찾을 수 없습니다. nvidia-smi가 설치되어 있는지 확인하세요."
            return 0
        }
    } catch {
        Log-Warn "GPU 상태 확인 실패. NVIDIA 드라이버가 설치되어 있는지 확인하세요."
        return 0
    }
}

function Test-PythonInstallation {
    Log-Info "Python 설치 확인 중..."
    
    try {
        $pythonVersion = python --version 2>$null
        if ($pythonVersion) {
            Log-Success "Python 설치됨: $pythonVersion"
            return $true
        } else {
            Log-Error "Python이 설치되지 않았습니다"
            Log-Info "Python 3.9+ 설치가 필요합니다: https://python.org"
            return $false
        }
    } catch {
        Log-Error "Python 설치 확인 실패"
        return $false
    }
}

function Set-PowerSettings {
    Log-Info "전원 설정 최적화 중..."
    
    try {
        # 고성능 전원 계획 설정
        powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c | Out-Null
        
        # 절전 모드 비활성화
        powercfg /change monitor-timeout-ac 0 | Out-Null
        powercfg /change standby-timeout-ac 0 | Out-Null
        powercfg /change hibernate-timeout-ac 0 | Out-Null
        
        Log-Success "전원 설정 최적화 완료"
        return $true
    } catch {
        Log-Warn "전원 설정 최적화 실패: $($_.Exception.Message)"
        return $false
    }
}

function Get-SystemInfo {
    Log-Info "시스템 정보 수집 중..."
    
    $info = @{
        ComputerName = $env:COMPUTERNAME
        IPAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*"} | Select-Object -First 1).IPAddress
        OS = (Get-WmiObject Win32_OperatingSystem).Caption
        TotalRAM = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        CPU = (Get-WmiObject Win32_Processor).Name
    }
    
    Log-Info "시스템 정보:"
    Log-Info "  컴퓨터명: $($info.ComputerName)"
    Log-Info "  IP 주소: $($info.IPAddress)"
    Log-Info "  운영체제: $($info.OS)"
    Log-Info "  RAM: $($info.TotalRAM) GB"
    Log-Info "  CPU: $($info.CPU)"
    
    return $info
}

function Test-NetworkConnectivity {
    param([string]$MasterIP)
    
    if ([string]::IsNullOrEmpty($MasterIP)) {
        return $true
    }
    
    Log-Info "마스터 노드 연결 테스트 중..."
    
    try {
        $pingResult = Test-Connection -ComputerName $MasterIP -Count 2 -Quiet
        if ($pingResult) {
            Log-Success "마스터 노드 $MasterIP 연결 확인"
            return $true
        } else {
            Log-Error "마스터 노드 $MasterIP 연결 실패"
            return $false
        }
    } catch {
        Log-Error "네트워크 연결 테스트 실패: $($_.Exception.Message)"
        return $false
    }
}

function Save-WorkerInfo {
    param([hashtable]$SystemInfo, [int]$GPUCount)
    
    $workerInfo = @{
        ComputerName = $SystemInfo.ComputerName
        IPAddress = $SystemInfo.IPAddress
        OS = $SystemInfo.OS
        TotalRAM = $SystemInfo.TotalRAM
        CPU = $SystemInfo.CPU
        GPUCount = $GPUCount
        SetupDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Status = "Ready"
    }
    
    # JSON 파일로 저장 (나중에 마스터에서 수집용)
    $workerInfoPath = "$env:TEMP\worker_info.json"
    $workerInfo | ConvertTo-Json | Out-File -FilePath $workerInfoPath -Encoding UTF8
    
    Log-Success "워커 정보 저장됨: $workerInfoPath"
    Log-Info "이 파일을 마스터 노드에 전달하면 자동 인식됩니다"
}

function Main {
    if ($Help) {
        Show-Help
        exit 0
    }
    
    Write-Host @"
🚀 분산 머신러닝 워커 노드 설정
=====================================
이 스크립트는 워커 노드를 분산 학습에 참여할 수 있도록 설정합니다.
각 워커 노드에서 1회만 실행하면 됩니다.

"@
    
    # 관리자 권한 확인
    if (-not (Test-AdminRights)) {
        Log-Error "이 스크립트는 관리자 권한으로 실행해야 합니다"
        Log-Info "PowerShell을 관리자 권한으로 다시 실행하세요"
        Read-Host "계속하려면 Enter를 누르세요"
        exit 1
    }
    
    # 마스터 IP 입력 (제공되지 않은 경우)
    if ([string]::IsNullOrEmpty($MasterIP)) {
        Write-Host ""
        $MasterIP = Read-Host "마스터 노드 IP 주소를 입력하세요 (선택사항, Enter로 건너뛰기)"
    }
    
    # 시스템 정보 수집
    $systemInfo = Get-SystemInfo
    
    # 설정 단계별 실행
    $steps = @(
        @{ Name = "PowerShell Remoting 활성화"; Action = { Enable-PowerShellRemoting } },
        @{ Name = "신뢰할 수 있는 호스트 설정"; Action = { Set-TrustedHosts -MasterIP $MasterIP } },
        @{ Name = "실행 정책 설정"; Action = { Set-ExecutionPolicy } },
        @{ Name = "방화벽 규칙 설정"; Action = { Set-FirewallRules } },
        @{ Name = "전원 설정 최적화"; Action = { Set-PowerSettings } }
    )
    
    $success = $true
    foreach ($step in $steps) {
        Write-Host ""
        $result = & $step.Action
        if (-not $result) {
            $success = $false
            Log-Error "$($step.Name) 실패"
        }
    }
    
    # GPU 및 Python 확인
    Write-Host ""
    $gpuCount = Test-GPUAvailability
    $pythonOK = Test-PythonInstallation
    
    # 네트워크 연결 테스트
    if (-not [string]::IsNullOrEmpty($MasterIP)) {
        Write-Host ""
        Test-NetworkConnectivity -MasterIP $MasterIP | Out-Null
    }
    
    # 워커 정보 저장
    Write-Host ""
    Save-WorkerInfo -SystemInfo $systemInfo -GPUCount $gpuCount
    
    # 결과 출력
    Write-Host ""
    Write-Host "=====================================
설정 완료 요약" -ForegroundColor Cyan
    
    if ($success -and $pythonOK) {
        Log-Success "✅ 워커 노드 설정이 성공적으로 완료되었습니다!"
        Write-Host ""
        Log-Info "다음 단계:"
        Log-Info "1. 이 설정을 모든 워커 노드에서 반복하세요"
        Log-Info "2. 마스터 노드에서 배포 스크립트를 실행하세요"
        Log-Info "3. 마스터 노드가 자동으로 코드 배포 및 훈련을 시작합니다"
        
        if (-not [string]::IsNullOrEmpty($MasterIP)) {
            Write-Host ""
            Log-Info "마스터 노드 접속 테스트:"
            Log-Info "  Test-WSMan -ComputerName $($systemInfo.IPAddress)"
        }
    } else {
        Log-Error "❌ 일부 설정이 실패했습니다"
        Log-Info "위의 오류 메시지를 확인하고 수동으로 해결하세요"
    }
    
    Write-Host ""
    Read-Host "완료. Enter를 누르면 종료합니다"
}

Main
