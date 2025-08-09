# 🚀 분산 학습 빠른 실행 가이드

## 🎯 한 눈에 보는 실행 방법

### **🖥️ 단일 노드 다중 GPU (가장 간단)**
```powershell
# Windows - 자동 설정으로 즉시 시작
.\scripts\launch_distributed.ps1 -Auto
```

### **🌐 다중 노드 설정 (2단계 방식)**

## **📋 새로운 2단계 방식**
1. **워커 노드**: 각 워커에서 1회 설정 (5분)
2. **마스터 노드**: 자동 배포 및 훈련 시작 (1분)

---

### **1️⃣ 단계: 워커 노드 설정 (각 워커에서 1회)**

#### **워커 노드 준비사항:**
- Windows 10/11
- GPU 드라이버 설치됨
- Python 3.9+ 설치됨

#### **워커 설정 실행:**
```powershell
# 각 워커 컴퓨터에서 관리자 권한으로 실행
.\scripts\worker_setup.ps1

# 또는 마스터 IP와 함께
.\scripts\worker_setup.ps1 -MasterIP "192.168.1.100"
```

**이 스크립트가 자동으로 처리하는 것들:**
- ✅ PowerShell Remoting 활성화
- ✅ 방화벽 포트 열기 (29500-29504)
- ✅ 네트워크 신뢰 관계 설정
- ✅ 전원 설정 최적화
- ✅ GPU 및 시스템 정보 수집

---

### **2️⃣ 단계: 마스터 노드에서 자동 배포**

#### **마스터 노드에서 실행:**
```powershell
# 대화형 모드 (권장)
.\scripts\master_deploy.ps1

# 워커 IP 직접 지정
.\scripts\master_deploy.ps1 -WorkerIPs @("192.168.1.101", "192.168.1.102")

# 이미 배포된 경우 (훈련만)
.\scripts\master_deploy.ps1 -SkipDeploy -WorkerIPs @("192.168.1.101")
```

**이 스크립트가 자동으로 처리하는 것들:**
- ✅ 모든 워커에 코드 및 데이터 복사
- ✅ 각 워커에서 uv 설치 및 의존성 설치
- ✅ Accelerate 설정 파일 자동 생성
- ✅ 분산 학습 자동 시작

---

## 📊 성능 예상

| 설정 | 예상 속도 | 실행 방법 |
|------|-----------|-----------|
| 1 GPU | 1x (기준) | `python scripts/train.py` |
| 4 GPU (단일 노드) | 3.5x | `.\scripts\launch_distributed.ps1 -Auto` |
| 8 GPU (2 노드) | 6.5x | `.\scripts\master_deploy.ps1` |
| 16 GPU (4 노드) | 12x | `.\scripts\master_deploy.ps1` |

---

## 🔍 상태 확인

### **실행 중 모니터링**
```powershell
# GPU 사용률 확인
nvidia-smi

# 네트워크 연결 확인  
netstat -an | findstr 29500

# 프로세스 확인
tasklist | findstr python

# PowerShell Remoting 테스트
Test-WSMan -ComputerName "워커IP"
```

### **정상 작동 신호**
```
✅ 정상 로그:
[INFO] Initializing distributed training...
[INFO] Rank 0/1 initialized
[INFO] All processes joined successfully
[INFO] Training started with 8 GPUs across 2 nodes

✅ GPU 사용률: 모든 GPU가 95%+ 사용률
✅ 네트워크: 포트 29500에 연결 상태 확인
```

---

## 🚨 문제 해결

### **자주 발생하는 문제**

#### **❌ "PowerShell Remoting 연결 실패"**
```powershell
# 해결 방법:
# 1. 워커에서 Remoting 활성화 확인
Enable-PSRemoting -Force

# 2. 신뢰할 수 있는 호스트 설정 확인
Get-Item WSMan:\localhost\Client\TrustedHosts

# 3. 방화벽 확인 (포트 5985, 5986)
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*Remote*"}
```

#### **❌ "CUDA out of memory"**
```yaml
# configs/train_config.yaml에서 배치 크기 줄이기
training:
  batch_size: 2  # 4에서 2로 감소
  gradient_accumulation_steps: 4  # 효과적 배치 크기 유지
```

#### **❌ "Access Denied"**
```powershell
# 해결 방법:
# 1. 관리자 권한으로 실행
# 2. 같은 사용자 계정으로 모든 노드 로그인
# 3. UAC 설정 확인
```

---

## 💡 성능 최적화 팁

### **1. 배치 크기 최적화**
```yaml
# GPU 메모리에 맞춰 조정
training:
  batch_size: 4      # 노드당 배치
  # 총 효과적 배치 = 4 × 2노드 × 4GPU × 2누적 = 64
  gradient_accumulation_steps: 2
```

### **2. 데이터 로딩 최적화**
```yaml
training:
  num_workers: 8     # CPU 코어 수
  pin_memory: true   # GPU 전송 속도 향상
```

### **3. 네트워크 최적화**
- **유선 연결 사용** (Wi-Fi 보다 안정적)
- **같은 스위치/라우터** 연결 (지연시간 최소화)
- **10Gbps+ 네트워크** 권장

---

## 📋 체크리스트

### **시작 전 확인사항**
- [ ] 모든 노드에 동일한 코드 버전
- [ ] 모든 노드에 동일한 데이터
- [ ] 방화벽 포트 29500 열림
- [ ] 네트워크 연결 확인 (ping 테스트)
- [ ] GPU 드라이버 최신 버전

### **실행 중 확인사항**
- [ ] 모든 GPU 사용률 95%+
- [ ] 네트워크 포트 연결됨
- [ ] 로스 값 정상적으로 감소
- [ ] WandB에서 모든 노드 메트릭 확인

---

## 🎓 추가 자료

- **상세 가이드**: `docs/distributed_training_guide.md`
- **초간단 가이드**: `DISTRIBUTED_QUICK_START.md`
- **Accelerate 문서**: https://huggingface.co/docs/accelerate

---

## 🆘 도움 요청

문제가 해결되지 않을 때:

1. **로그 확인**: `logs/` 폴더의 최신 로그 파일
2. **환경 정보**: `accelerate env` 출력
3. **GPU 정보**: `nvidia-smi` 출력
4. **네트워크 정보**: `ipconfig` (Windows) / `ifconfig` (Linux) 출력

이 정보와 함께 문의하시면 빠른 해결이 가능합니다! 🚀
