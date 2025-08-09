# 🚀 분산 학습 완전 가이드 (Windows 2단계 방식)

## 📋 목차
1. [기본 개념](#기본-개념)
2. [새로운 2단계 방식](#새로운-2단계-방식)
3. [워커 노드 설정](#워커-노드-설정)
4. [마스터 노드 자동화](#마스터-노드-자동화)
5. [문제 해결](#문제-해결)

---

## 🧠 기본 개념

### **분산 학습 유형**
- **단일 GPU**: 1개 노드, 1개 GPU
- **단일 노드 다중 GPU**: 1개 노드, 여러 GPU
- **다중 노드 다중 GPU**: 여러 노드, 각각 여러 GPU

### **우리 프로젝트의 분산 학습 아키텍처**
```
마스터 노드 (Rank 0)          워커 노드 1 (Rank 1)
┌─────────────────┐          ┌─────────────────┐
│ GPU 0 (주 GPU)  │          │ GPU 0           │
│ GPU 1           │   <---> │ GPU 1           │
│ GPU 2           │          │ GPU 2           │
│ GPU 3           │          │ GPU 3           │
└─────────────────┘          └─────────────────┘
```

---

## 🔄 새로운 2단계 방식

### **혁신적인 개선사항**
기존의 복잡한 수동 설정을 **2단계 자동화**로 대폭 단순화:

#### **🔄 Before (기존 방식)**
```
모든 노드에서 해야 할 일:
1. 프로젝트 클론 ❌
2. Python 환경 설정 ❌ 
3. uv 설치 및 의존성 설치 ❌
4. 네트워크 설정 ❌
5. Accelerate 설정 ❌
6. 수동 실행 ❌
총 소요시간: 30분 × 노드 수
```

#### **✅ After (새로운 방식)**
```
워커 노드 (1회, 5분):
1. worker_setup.ps1 실행 ✅

마스터 노드 (1분):
1. master_deploy.ps1 실행 ✅

총 소요시간: 6분 (노드 수 무관)
```

### **핵심 원리**
- **워커 노드**: PowerShell Remoting 기반 신뢰 관계만 구축
- **마스터 노드**: 원격으로 모든 배포 및 설정 자동화

---

## 🛠️ 환경 구성

### **1. 하드웨어 요구사항**
```
최소 권장:
- 각 노드: GPU 2개 이상 (8GB+ VRAM)
- 네트워크: 10Gbps 이상
- RAM: 32GB+ 권장

최적 권장:
- 각 노드: GPU 4개 이상 (16GB+ VRAM)
- 네트워크: InfiniBand 또는 100Gbps Ethernet
- RAM: 64GB+ 권장
```

### **2. 소프트웨어 요구사항**
- Python 3.9+
- PyTorch 2.0+ (GPU 지원)
- Accelerate 0.20+
- CUDA 11.8+ / ROCm (AMD GPU)

---

## 🔧 워커 노드 설정

### **Step 1: 워커 노드 준비사항**

각 워커 노드에서 **최소한의 준비사항**:

#### **필수 설치사항**
- **Windows 10/11** (PowerShell 5.1+)
- **GPU 드라이버** (NVIDIA/AMD 최신 버전)
- **Python 3.9+** (python.org에서 다운로드)

#### **선택사항** (성능 향상용)
- **Visual Studio Redistributable** (C++ 라이브러리)
- **CUDA Toolkit** (개발용, 런타임은 자동 설치됨)

> **중요**: 프로젝트 코드, uv, 의존성 등은 **설치하지 마세요**! 
> 마스터에서 자동으로 배포됩니다.

### **Step 2: 워커 자동 설정 실행**

#### **2-1. 관리자 권한으로 PowerShell 실행**
1. **시작 메뉴** → **PowerShell** 우클릭 → **관리자 권한으로 실행**
2. UAC 창에서 **예** 클릭

#### **2-2. 워커 설정 스크립트 실행**
```powershell
# 프로젝트를 임시로 다운로드 (설정용)
git clone <프로젝트-URL> temp-setup
cd temp-setup

# 워커 설정 실행
.\scripts\worker_setup.ps1

# 마스터 IP 알고 있다면
.\scripts\worker_setup.ps1 -MasterIP "192.168.1.100"
```

#### **2-3. 자동 처리되는 내용**
스크립트가 자동으로 처리하는 작업들:
- ✅ PowerShell Remoting 활성화
- ✅ 방화벽 포트 열기 (29500-29504, 5985-5986)
- ✅ 신뢰할 수 있는 호스트 설정
- ✅ 실행 정책 조정
- ✅ 전원 설정 최적화 (절전 모드 비활성화)
- ✅ 시스템 정보 수집 및 저장

#### **2-4. 설정 완료 확인**
```powershell
# 설정 확인 명령들
Get-Service WinRM                    # WinRM 서비스 실행 확인
Get-Item WSMan:\localhost\Client\TrustedHosts  # 신뢰 호스트 확인
Test-WSMan                          # PowerShell Remoting 테스트
```

---

## 🎛️ 마스터 노드 자동화

### **Step 1: 마스터 노드 준비**

#### **1-1. 프로젝트 환경 구성**
```powershell
# 프로젝트 클론 및 환경 설정
git clone <프로젝트-URL>
cd text-to-image-generation

# uv 설치 및 의존성 설치
pip install uv
uv sync

# 가상환경 활성화
.\.venv\Scripts\activate
```

#### **1-2. 데이터 준비**
```powershell
# 훈련 데이터를 ./data/ 디렉토리에 배치
# 구조:
# ./data/train/input_text/    (텍스트 파일들)
# ./data/train/label_floorplan/ (이미지 파일들)
```

### **Step 2: 자동 배포 및 훈련**

#### **2-1. 워커 목록 준비 (3가지 방법)**

**방법 A: 대화형 입력 (가장 간단)**
```powershell
.\scripts\master_deploy.ps1
# 스크립트가 워커 IP를 대화형으로 요청
```

**방법 B: 직접 IP 지정**
```powershell
.\scripts\master_deploy.ps1 -WorkerIPs @("192.168.1.101", "192.168.1.102", "192.168.1.103")
```

**방법 C: JSON 파일 사용**
```powershell
# workers.json 파일 생성
@"
[
  {"IPAddress": "192.168.1.101", "ComputerName": "Worker1", "GPUCount": 4},
  {"IPAddress": "192.168.1.102", "ComputerName": "Worker2", "GPUCount": 4}
]
"@ | Out-File workers.json

.\scripts\master_deploy.ps1 -WorkersFile "workers.json"
```

#### **2-2. 자동 처리되는 전체 과정**

마스터 배포 스크립트가 자동으로 수행하는 작업들:

1. **연결 테스트** 🔗
   - 모든 워커 노드 PowerShell Remoting 연결 확인
   - 시스템 정보 수집 (GPU, RAM, OS 등)

2. **프로젝트 배포** 📦
   - 마스터의 전체 프로젝트를 모든 워커에 복사
   - 자동 제외: `.git`, `.venv`, `__pycache__`, `logs`, `checkpoints`

3. **환경 구성** ⚙️
   - 각 워커에서 uv 설치 (없는 경우)
   - 프로젝트 의존성 자동 설치 (`uv sync`)
   - 가상환경 설정

4. **설정 파일 생성** 📄
   - 각 노드별 Accelerate 설정 파일 자동 생성
   - 노드 순위(rank), IP, 포트 자동 할당

5. **분산 학습 시작** 🚀
   - 워커 노드들 순차적으로 훈련 시작 (백그라운드)
   - 마스터 노드에서 주 훈련 프로세스 실행

### **Step 3: 고급 옵션들**

#### **3-1. 기존 배포 환경 재사용**
```powershell
# 이미 배포된 환경에서 훈련만 다시 실행
.\scripts\master_deploy.ps1 -SkipDeploy -WorkerIPs @("192.168.1.101")
```

#### **3-2. 커스텀 설정**
```powershell
# 커스텀 설정으로 실행
.\scripts\master_deploy.ps1 `
  -ConfigFile "configs/custom_config.yaml" `
  -ProjectPath "D:\ml-project" `
  -Port 29501 `
  -WorkerIPs @("192.168.1.101", "192.168.1.102")
```

**마스터 노드 (192.168.1.100):**
```bash
accelerate config
```
```
# 설정 답변 예시:
- In which compute environment are you running? multi-node multi-GPU
- How many machines will you use? 2
- What is the rank of this machine? 0
- What is the IP address of the machine with rank 0? 192.168.1.100
- What is the port you will use to communicate? 29400
- How many GPU(s) should be used for distributed training? 4
- Mixed precision: fp16
```

**워커 노드 1 (192.168.1.101):**
```bash
accelerate config
```
```
# 설정 답변 예시:
- In which compute environment are you running? multi-node multi-GPU
- How many machines will you use? 2
- What is the rank of this machine? 1
- What is the IP address of the machine with rank 0? 192.168.1.100
- What is the port you will use to communicate? 29400
- How many GPU(s) should be used for distributed training? 4
- Mixed precision: fp16
```

#### **3-2. 설정 파일 확인**
```bash
# 설정 파일 위치 확인
accelerate env
cat ~/.cache/huggingface/accelerate/default_config.yaml
```

### **Step 4: 프로젝트 설정 수정**

#### **4-1. train_config.yaml 수정**
```yaml
training:
  batch_size: 8  # 노드당 배치 크기 (총 배치 = 8 × 2노드 = 16)
  num_workers: 4  # 충분한 워커 수
  mixed_precision: "fp16"
  
  # 분산 학습 전용 설정
  gradient_accumulation_steps: 2  # 메모리 절약
  save_every: 500  # 자주 저장
  
wandb:
  enabled: true
  project: "floorplan-distributed"
```

---

## 🚀 실행 방법

### **방법 1: 동시 실행 (권장)**

#### **1-1. 마스터 노드에서 실행**
```bash
# 터미널 1 (마스터 노드)
cd /path/to/text-to-image-generation
./.venv/Scripts/activate
accelerate launch scripts/train.py --config configs/train_config.yaml
```

#### **1-2. 워커 노드에서 실행 (30초 이내)**
```bash
# 터미널 2 (워커 노드 1)
cd /path/to/text-to-image-generation
./.venv/Scripts/activate
accelerate launch scripts/train.py --config configs/train_config.yaml
```

### **방법 2: SSH를 통한 자동 실행**

#### **2-1. 실행 스크립트 생성**
```bash
# scripts/distributed_launch.sh
#!/bin/bash

MASTER_NODE="192.168.1.100"
WORKER_NODES=("192.168.1.101" "192.168.1.102")
PROJECT_PATH="/path/to/text-to-image-generation"

# 마스터 노드에서 실행
echo "Starting training on master node..."
cd $PROJECT_PATH
./.venv/Scripts/activate
accelerate launch scripts/train.py --config configs/train_config.yaml &
MASTER_PID=$!

# 워커 노드들에서 실행
for worker in "${WORKER_NODES[@]}"; do
    echo "Starting training on worker node: $worker"
    ssh $worker "cd $PROJECT_PATH && ./.venv/Scripts/activate && accelerate launch scripts/train.py --config configs/train_config.yaml" &
done

# 모든 프로세스 완료 대기
wait $MASTER_PID
echo "Distributed training completed!"
```

#### **2-2. 실행 권한 부여 및 실행**
```bash
chmod +x scripts/distributed_launch.sh
./scripts/distributed_launch.sh
```

---

## 📊 모니터링

### **1. 실시간 로그 확인**
```bash
# 마스터 노드 로그
tail -f logs/train_*.log

# GPU 사용률 모니터링
watch -n 1 nvidia-smi
```

### **2. WandB 대시보드**
- 브라우저에서 https://wandb.ai 접속
- 프로젝트 "floorplan-distributed" 확인
- 다중 노드 메트릭 모니터링

### **3. 네트워크 통신 확인**
```bash
# 포트 연결 상태 확인
netstat -tulpn | grep 29400

# 노드 간 ping 테스트
ping 192.168.1.101
ping 192.168.1.102
```

---

## 🔧 성능 최적화

### **1. 배치 크기 튜닝**
```python
# 총 효과적 배치 크기 계산
total_batch_size = batch_size × num_nodes × num_gpus_per_node × gradient_accumulation_steps
# 예: 4 × 2 × 4 × 2 = 64
```

### **2. 네트워크 최적화**
```bash
# Linux에서 네트워크 버퍼 크기 증가
echo 'net.core.rmem_max = 134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 134217728' >> /etc/sysctl.conf
sysctl -p
```

### **3. 데이터 로딩 최적화**
```yaml
training:
  num_workers: 8  # CPU 코어 수에 맞춰 조정
  pin_memory: true
  persistent_workers: true
```

---

## 🚨 문제 해결

### **자주 발생하는 문제들**

#### **1. "Connection refused" 오류**
```bash
# 해결 방법:
1. 방화벽 포트 확인
2. IP 주소 정확성 확인
3. 노드 간 ping 테스트
```

#### **2. "CUDA out of memory" 오류**
```yaml
# train_config.yaml에서 배치 크기 줄이기
training:
  batch_size: 2  # 4에서 2로 감소
  gradient_accumulation_steps: 4  # 2에서 4로 증가
```

#### **3. "Timeout" 오류**
```yaml
# 더 긴 타임아웃 설정
accelerate_config:
  timeout: 1800  # 30분
```

#### **4. 데이터 불일치 오류**
```bash
# 모든 노드에서 데이터 해시 확인
find ./data -type f -name "*.png" | head -5 | xargs md5sum
```

### **로그 분석**

#### **정상 시작 로그**
```
Initializing distributed training...
Rank 0/1 initialized
All processes joined successfully
Training started with 8 GPUs across 2 nodes
```

#### **오류 로그 패턴**
```
NCCL timeout → 네트워크 문제
CUDA OOM → 메모리 부족  
Process died → 노드 연결 끊김
```

---

## 📈 예상 성능

### **성능 향상 예측**
```
단일 GPU (baseline):     100% (1시간)
단일 노드 4 GPU:         ~350% (17분)
2노드 8 GPU:            ~650% (9분)
4노드 16 GPU:           ~1200% (5분)
```

### **확장성 고려사항**
- **선형 확장 한계**: 8노드 이상에서는 통신 오버헤드 증가
- **네트워크 병목**: InfiniBand 사용 시 더 좋은 성능
- **데이터 I/O**: SSD RAID 또는 분산 파일시스템 권장

---

## 🎯 실전 팁

### **1. 단계적 확장**
```
1단계: 단일 노드 다중 GPU로 시작
2단계: 2노드로 확장하여 안정성 확인  
3단계: 더 많은 노드로 확장
```

### **2. 체크포인트 전략**
```yaml
training:
  save_every: 250  # 자주 저장
  max_checkpoints: 10  # 충분한 백업
```

### **3. 모니터링 필수 항목**
- GPU 사용률 (모든 노드 95%+ 유지)
- 네트워크 대역폭 사용률
- 데이터 로딩 속도
- 메모리 사용량

이 가이드를 따라하시면 안정적이고 효율적인 분산 학습 환경을 구축할 수 있습니다! 🚀
