# ⚡ 분산 학습 초간단 시작 가이드

## 🎯 2단계로 끝내는 분산 학습

### **1️⃣ 워커 노드 설정 (각 워커에서 1회, 5분)**

#### **준비사항 체크**
- [ ] Windows 10/11
- [ ] GPU 드라이버 설치됨  
- [ ] Python 3.9+ 설치됨

#### **실행**
```powershell
# 관리자 권한 PowerShell에서
git clone <프로젝트-URL> temp-setup
cd temp-setup
.\scripts\worker_setup.ps1 -MasterIP "마스터IP"
```

**완료 표시**: "✅ 워커 노드 설정이 성공적으로 완료되었습니다!"

---

### **2️⃣ 마스터 노드 배포 (마스터에서 1회, 1분)**

#### **준비**
```powershell
# 프로젝트 환경 준비
git clone <프로젝트-URL>
cd text-to-image-generation
pip install uv && uv sync
.\.venv\Scripts\activate
```

#### **실행**
```powershell
# 자동 배포 및 훈련 시작
.\scripts\master_deploy.ps1 -WorkerIPs @("워커1IP", "워커2IP")

# 또는 대화형 모드
.\scripts\master_deploy.ps1
```

**완료**: 모든 노드에서 자동으로 분산 학습 시작! 🚀

---

## 📊 예상 결과

| 노드 구성 | 총 설정 시간 | 성능 향상 |
|-----------|--------------|-----------|
| 2 노드 8 GPU | 7분 | 6.5배 |
| 4 노드 16 GPU | 11분 | 12배 |

---

## 🚨 문제 발생 시

### **워커 설정 실패**
```powershell
# 관리자 권한으로 다시 실행
Enable-PSRemoting -Force
.\scripts\worker_setup.ps1
```

### **마스터 연결 실패**  
```powershell
# 연결 테스트
Test-WSMan -ComputerName "워커IP"

# 신뢰 호스트 확인
Get-Item WSMan:\localhost\Client\TrustedHosts
```

### **CUDA 메모리 부족**
```yaml
# configs/train_config.yaml
training:
  batch_size: 2  # 줄이기
  gradient_accumulation_steps: 4  # 늘리기
```

---

## 🎓 추가 도움말

- **상세 기술 가이드**: `docs/distributed_training_guide.md`
- **빠른 참조 가이드**: `README_DISTRIBUTED.md`
- **텍스트 청킹 테스트**: `scripts/test_token_limits.py`
- **문제 해결**: 위 문서들의 "문제 해결" 섹션

이제 몇 개 명령어로 강력한 분산 학습 환경을 구축할 수 있습니다! 🎉
