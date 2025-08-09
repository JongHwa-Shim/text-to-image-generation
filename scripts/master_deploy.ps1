# 마스터 노드 자동 배포 및 분산 학습 실행 스크립트

param(
    [string]$ConfigFile = "configs/train_config.yaml",
    [string]$WorkersFile = "",
    [string[]]$WorkerIPs = @(),
    [string]$Username = "",
    [string]$Password = "",
    [string]$ProjectPath = "C:\ml-project",
    [int]$Port = 29500,
    [switch]$SkipDeploy,
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
마스터 노드 자동 배포 및 분산 학습 스크립트

사용법:
    .\scripts\master_deploy.ps1 [매개변수]

매개변수:
    -ConfigFile FILE        훈련 설정 파일 (기본값: configs/train_config.yaml)
    -WorkersFile FILE       워커 목록 JSON 파일
    -WorkerIPs IP[]         워커 노드 IP 배열
    -Username USER          원격 접속 사용자명
    -Password PASS          원격 접속 비밀번호
    -ProjectPath PATH       워커에서 프로젝트 경로 (기본값: C:\ml-project)
    -Port PORT              통신 포트 (기본값: 29500)
    -SkipDeploy            코드 배포 건너뛰기 (이미 배포된 경우)
    -Help                  이 도움말 표시

예시:
    # 대화형 모드
    .\scripts\master_deploy.ps1

    # 워커 IP 직접 지정
    .\scripts\master_deploy.ps1 -WorkerIPs @("192.168.1.101", "192.168.1.102")

    # 워커 목록 파일 사용
    .\scripts\master_deploy.ps1 -WorkersFile "workers.json"

    # 배포 없이 훈련만
    .\scripts\master_deploy.ps1 -SkipDeploy -WorkerIPs @("192.168.1.101")
"@
}

function Get-Credentials {
    param([string]$Username, [string]$Password)
    
    if ([string]::IsNullOrEmpty($Username)) {
        $Username = Read-Host "워커 노드 접속 사용자명"
    }
    
    if ([string]::IsNullOrEmpty($Password)) {
        $securePassword = Read-Host "비밀번호" -AsSecureString
    } else {
        $securePassword = ConvertTo-SecureString $Password -AsPlainText -Force
    }
    
    return New-Object System.Management.Automation.PSCredential($Username, $securePassword)
}

function Get-WorkerList {
    param([string]$WorkersFile, [string[]]$WorkerIPs)
    
    $workers = @()
    
    # 파일에서 워커 목록 로드
    if (-not [string]::IsNullOrEmpty($WorkersFile) -and (Test-Path $WorkersFile)) {
        Log-Info "워커 목록 파일에서 로드 중: $WorkersFile"
        try {
            $workersData = Get-Content $WorkersFile | ConvertFrom-Json
            foreach ($worker in $workersData) {
                $workers += @{
                    IP = $worker.IPAddress
                    Name = $worker.ComputerName
                    GPUCount = $worker.GPUCount
                }
            }
        } catch {
            Log-Error "워커 파일 읽기 실패: $($_.Exception.Message)"
        }
    }
    
    # 직접 지정된 IP 추가
    foreach ($ip in $WorkerIPs) {
        $workers += @{
            IP = $ip
            Name = "Worker-$ip"
            GPUCount = 4  # 기본값
        }
    }
    
    # 대화형 입력
    if ($workers.Count -eq 0) {
        Log-Info "워커 노드를 대화형으로 입력하세요 (빈 값 입력 시 종료)"
        do {
            $ip = Read-Host "워커 노드 IP 주소"
            if (-not [string]::IsNullOrEmpty($ip)) {
                $workers += @{
                    IP = $ip
                    Name = "Worker-$ip"
                    GPUCount = 4
                }
            }
        } while (-not [string]::IsNullOrEmpty($ip))
    }
    
    return $workers
}

function Test-WorkerConnectivity {
    param([array]$Workers, [pscredential]$Credential)
    
    Log-Info "워커 노드 연결 테스트 중..."
    $connectedWorkers = @()
    
    foreach ($worker in $Workers) {
        try {
            Log-Info "  $($worker.IP) 연결 테스트 중..."
            $session = New-PSSession -ComputerName $worker.IP -Credential $Credential -ErrorAction Stop
            
            # 기본 정보 수집
            $info = Invoke-Command -Session $session -ScriptBlock {
                @{
                    ComputerName = $env:COMPUTERNAME
                    OS = (Get-WmiObject Win32_OperatingSystem).Caption
                    FreeSpace = [math]::Round((Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB, 1)
                }
            }
            
            $worker.Session = $session
            $worker.Name = $info.ComputerName
            $worker.OS = $info.OS
            $worker.FreeSpace = $info.FreeSpace
            
            Log-Success "  ✅ $($worker.IP) ($($info.ComputerName)) 연결됨"
            $connectedWorkers += $worker
            
        } catch {
            Log-Error "  ❌ $($worker.IP) 연결 실패: $($_.Exception.Message)"
        }
    }
    
    if ($connectedWorkers.Count -eq 0) {
        Log-Error "연결 가능한 워커 노드가 없습니다"
        exit 1
    }
    
    Log-Success "$($connectedWorkers.Count)개 워커 노드 연결 확인"
    return $connectedWorkers
}

function Deploy-ProjectToWorkers {
    param([array]$Workers, [string]$ProjectPath)
    
    Log-Info "프로젝트 코드 및 데이터 배포 중..."
    
    $projectRoot = Get-Location
    $excludePatterns = @("*.git*", "*.venv*", "*__pycache__*", "*.pyc", "*logs*", "*outputs*", "*checkpoints*")
    
    foreach ($worker in $Workers) {
        Log-Info "  $($worker.Name) ($($worker.IP))에 배포 중..."
        
        try {
            # 대상 디렉토리 생성
            Invoke-Command -Session $worker.Session -ScriptBlock {
                param($Path)
                if (Test-Path $Path) {
                    Remove-Item $Path -Recurse -Force
                }
                New-Item -ItemType Directory -Path $Path -Force | Out-Null
            } -ArgumentList $ProjectPath
            
            # 파일 복사 (PowerShell 세션 사용)
            $filesToCopy = Get-ChildItem -Path $projectRoot -Recurse | Where-Object {
                $exclude = $false
                foreach ($pattern in $excludePatterns) {
                    if ($_.FullName -like "*$pattern*") {
                        $exclude = $true
                        break
                    }
                }
                return -not $exclude -and -not $_.PSIsContainer
            }
            
            foreach ($file in $filesToCopy) {
                $relativePath = $file.FullName.Substring($projectRoot.Path.Length + 1)
                $targetPath = Join-Path $ProjectPath $relativePath
                $targetDir = Split-Path $targetPath -Parent
                
                # 대상 디렉토리 생성
                Invoke-Command -Session $worker.Session -ScriptBlock {
                    param($Dir)
                    if (-not (Test-Path $Dir)) {
                        New-Item -ItemType Directory -Path $Dir -Force | Out-Null
                    }
                } -ArgumentList $targetDir
                
                # 파일 복사
                Copy-Item -Path $file.FullName -ToSession $worker.Session -Destination $targetPath -Force
            }
            
            Log-Success "  ✅ $($worker.Name) 배포 완료"
            
        } catch {
            Log-Error "  ❌ $($worker.Name) 배포 실패: $($_.Exception.Message)"
            throw
        }
    }
}

function Install-Dependencies {
    param([array]$Workers, [string]$ProjectPath)
    
    Log-Info "의존성 설치 중..."
    
    foreach ($worker in $Workers) {
        Log-Info "  $($worker.Name)에서 의존성 설치 중..."
        
        try {
            $result = Invoke-Command -Session $worker.Session -ScriptBlock {
                param($Path)
                cd $Path
                
                # uv 설치 확인
                try {
                    uv --version | Out-Null
                } catch {
                    pip install uv
                }
                
                # 의존성 설치
                uv sync
                
                return "SUCCESS"
            } -ArgumentList $ProjectPath
            
            if ($result -eq "SUCCESS") {
                Log-Success "  ✅ $($worker.Name) 의존성 설치 완료"
            }
            
        } catch {
            Log-Error "  ❌ $($worker.Name) 의존성 설치 실패: $($_.Exception.Message)"
            throw
        }
    }
}

function Generate-AccelerateConfigs {
    param([array]$Workers, [string]$MasterIP, [int]$Port, [string]$ProjectPath)
    
    Log-Info "Accelerate 설정 파일 생성 중..."
    
    $totalGPUs = 4  # 마스터 노드 GPU 수 (가정)
    foreach ($worker in $Workers) {
        $totalGPUs += $worker.GPUCount
    }
    
    $totalNodes = 1 + $Workers.Count
    
    # 마스터 노드 설정
    $masterConfig = @"
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_process_ip: $MasterIP
main_process_port: $Port
main_training_function: main
mixed_precision: fp16
num_machines: $totalNodes
num_processes: $totalGPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"@
    
    $configDir = "configs\accelerate"
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }
    
    $masterConfigPath = "$configDir\master_config.yaml"
    $masterConfig | Out-File -FilePath $masterConfigPath -Encoding UTF8
    Log-Success "마스터 설정 파일 생성: $masterConfigPath"
    
    # 워커 노드 설정
    for ($i = 0; $i -lt $Workers.Count; $i++) {
        $worker = $Workers[$i]
        $rank = $i + 1
        
        $workerConfig = @"
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: $rank
main_process_ip: $MasterIP
main_process_port: $Port
main_training_function: main
mixed_precision: fp16
num_machines: $totalNodes
num_processes: $totalGPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"@
        
        # 워커에 설정 파일 전송
        try {
            Invoke-Command -Session $worker.Session -ScriptBlock {
                param($Config, $Path)
                $configDir = "$Path\configs\accelerate"
                if (-not (Test-Path $configDir)) {
                    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
                }
                $Config | Out-File -FilePath "$configDir\worker_config.yaml" -Encoding UTF8
            } -ArgumentList $workerConfig, $ProjectPath
            
            Log-Success "워커 $rank 설정 파일 전송 완료"
            
        } catch {
            Log-Error "워커 $rank 설정 파일 전송 실패: $($_.Exception.Message)"
            throw
        }
    }
    
    return $masterConfigPath
}

function Start-DistributedTraining {
    param([array]$Workers, [string]$MasterConfigPath, [string]$ConfigFile, [string]$ProjectPath)
    
    Log-Info "분산 학습 시작..."
    
    # 워커 노드들 먼저 시작 (백그라운드)
    $workerJobs = @()
    foreach ($worker in $Workers) {
        Log-Info "  워커 $($worker.Name) 훈련 시작..."
        
        $job = Invoke-Command -Session $worker.Session -AsJob -ScriptBlock {
            param($Path, $ConfigFile)
            cd $Path
            .\.venv\Scripts\activate.ps1
            accelerate launch --config_file configs\accelerate\worker_config.yaml scripts\train.py --config $ConfigFile
        } -ArgumentList $ProjectPath, $ConfigFile
        
        $workerJobs += $job
        Start-Sleep -Seconds 2  # 워커 간 시작 간격
    }
    
    # 마스터 노드 시작 (메인 프로세스)
    Log-Success "모든 워커 노드 시작됨. 마스터 노드에서 훈련 시작..."
    Start-Sleep -Seconds 5  # 워커들이 준비될 시간
    
    try {
        .\.venv\Scripts\activate.ps1
        accelerate launch --config_file $MasterConfigPath scripts\train.py --config $ConfigFile
        
        Log-Success "훈련 완료!"
        
    } catch {
        Log-Error "훈련 중 오류 발생: $($_.Exception.Message)"
        
        # 워커 작업 중단
        Log-Info "워커 노드 작업 중단 중..."
        foreach ($job in $workerJobs) {
            Stop-Job $job -ErrorAction SilentlyContinue
        }
        
        throw
    } finally {
        # 세션 정리
        foreach ($worker in $Workers) {
            Remove-PSSession $worker.Session -ErrorAction SilentlyContinue
        }
    }
}

function Get-LocalIP {
    try {
        $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notlike "*Loopback*"} | Select-Object -First 1).IPAddress
        return $ip
    } catch {
        return "127.0.0.1"
    }
}

function Main {
    if ($Help) {
        Show-Help
        exit 0
    }
    
    Write-Host @"
🚀 분산 머신러닝 마스터 노드 배포
======================================
이 스크립트는 마스터 노드에서 모든 워커에 자동 배포하고
분산 학습을 시작합니다.

"@ -ForegroundColor Cyan
    
    # 기본 설정
    $masterIP = Get-LocalIP
    Log-Info "마스터 노드 IP: $masterIP"
    Log-Info "프로젝트 루트: $(Get-Location)"
    
    # 워커 목록 수집
    $workers = Get-WorkerList -WorkersFile $WorkersFile -WorkerIPs $WorkerIPs
    if ($workers.Count -eq 0) {
        Log-Error "워커 노드가 지정되지 않았습니다"
        exit 1
    }
    
    Log-Info "워커 노드 $($workers.Count)개:"
    foreach ($worker in $workers) {
        Log-Info "  - $($worker.IP)"
    }
    
    # 자격 증명 획득
    $credential = Get-Credentials -Username $Username -Password $Password
    
    # 워커 연결 테스트
    $connectedWorkers = Test-WorkerConnectivity -Workers $workers -Credential $credential
    
    if (-not $SkipDeploy) {
        # 프로젝트 배포
        Deploy-ProjectToWorkers -Workers $connectedWorkers -ProjectPath $ProjectPath
        
        # 의존성 설치
        Install-Dependencies -Workers $connectedWorkers -ProjectPath $ProjectPath
    } else {
        Log-Info "코드 배포를 건너뜁니다"
    }
    
    # Accelerate 설정 생성
    $masterConfigPath = Generate-AccelerateConfigs -Workers $connectedWorkers -MasterIP $masterIP -Port $Port -ProjectPath $ProjectPath
    
    # 분산 학습 시작
    Write-Host ""
    $confirm = Read-Host "분산 학습을 시작하시겠습니까? (y/N)"
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        Start-DistributedTraining -Workers $connectedWorkers -MasterConfigPath $masterConfigPath -ConfigFile $ConfigFile -ProjectPath $ProjectPath
    } else {
        Log-Info "준비 완료. 수동으로 훈련을 시작하세요:"
        Log-Info "  accelerate launch --config_file $masterConfigPath scripts\train.py --config $ConfigFile"
    }
}

Main
