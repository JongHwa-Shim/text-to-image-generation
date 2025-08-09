# 텍스트 조건 평면도 생성 모델 - 완전 가이드

> **새로운 세션을 위한 종합 설명서**  
> 이 문서는 프로젝트의 모든 기술적 세부사항, 구현 내용, 발생했던 문제들과 해결방법을 포함합니다.

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 기술 스택](#2-핵심-기술-스택)
3. [프로젝트 아키텍처](#3-프로젝트-아키텍처)
4. [주요 구현 사항](#4-주요-구현-사항)
5. [데이터 처리 파이프라인](#5-데이터-처리-파이프라인)
6. [모델 구조](#6-모델-구조)
7. [훈련 시스템](#7-훈련-시스템)
8. [추론 시스템](#8-추론-시스템)
9. [분산 학습 시스템](#9-분산-학습-시스템)
10. [텍스트 처리 고급 기능](#10-텍스트-처리-고급-기능)
11. [발생한 문제들과 해결방법](#11-발생한-문제들과-해결방법)
12. [파일별 상세 설명](#12-파일별-상세-설명)
13. [설정 및 실행 방법](#13-설정-및-실행-방법)
14. [개발 히스토리](#14-개발-히스토리)

---

## 1. 프로젝트 개요

### 1.1 목표
- **입력**: 구조화된 텍스트 (방의 개수/종류, 연결관계, 위치관계)
- **출력**: 256x256 RGB 평면도 이미지
- **특징**: LoRA 파인튜닝을 통한 Stable Diffusion 기반 생성 모델

### 1.2 핵심 혁신사항
1. **Attention 기반 텍스트 청킹**: CLIP 77토큰 제한 극복
2. **2단계 분산학습**: Windows 환경 최적화 자동화 시스템
3. **스마트 데이터 증강**: 부분 조건 입력 지원

### 1.3 입력 텍스트 형식
```
SJH-Style FloorPlan Generation

[Number and Type of Rooms]
The floorplan have 1 living room, 1 master room, 1 kitchen, 1 bathroom

[Connection Between Rooms]
living room #1 and master room #1 are connected.
living room #1 and kitchen #1 are connected.

[Positional Relationship Between Rooms]
master room #1 is left-below living room #1.
kitchen #1 is above living room #1.
```

### 1.4 출력 이미지 형식
- **해상도**: 256x256x3 (RGB)
- **색상 매핑**: 각 방 유형별 고유 RGB 값
```python
room_colors = {
    'living_room': [255, 255, 220],    # 연한 노란색
    'master_room': [0, 255, 0],        # 초록색
    'bedroom': [30, 140, 50],          # 어두운 초록색
    'kitchen': [190, 90, 90],          # 갈색
    'bathroom': [66, 78, 255],         # 파란색
    'balcony': [50, 180, 255],         # 하늘색
    'wall': [95, 95, 95],              # 회색
    'door': [255, 255, 0],             # 노란색
    'window': [125, 190, 190]          # 청록색
}
```

---

## 2. 핵심 기술 스택

### 2.1 기반 기술
- **Python**: 3.9+
- **PyTorch**: 딥러닝 프레임워크
- **Stable Diffusion v1.5**: 기반 생성 모델
- **LoRA (Low-Rank Adaptation)**: 효율적 파인튜닝
- **CLIP**: 텍스트-이미지 인코딩

### 2.2 라이브러리
- **Hugging Face Accelerate**: 분산 학습 및 혼합 정밀도
- **Diffusers**: Stable Diffusion 파이프라인
- **PEFT**: LoRA 구현
- **Weights & Biases**: 실험 추적
- **OpenCV**: 이미지 후처리
- **uv**: Python 패키지 관리

### 2.3 개발 도구
- **PowerShell**: Windows 분산학습 자동화
- **YAML**: 설정 파일 형식
- **Git**: 버전 관리

---

## 3. 프로젝트 아키텍처

### 3.1 전체 파이프라인
```
텍스트 입력 → 전처리/청킹 → CLIP 인코딩 → U-Net 디퓨전 → VAE 디코딩 → 후처리 → 최종 이미지
```

### 3.2 모듈 구조
```
src/
├── data/           # 데이터 처리
├── models/         # 모델 정의
├── training/       # 훈련 시스템
├── inference/      # 추론 시스템
└── utils/          # 유틸리티
```

### 3.3 주요 컴포넌트
1. **FloorPlanDataset**: 데이터 로딩 및 증강
2. **FloorPlanDiffusionModel**: LoRA 적용된 Stable Diffusion
3. **FloorPlanTrainer**: Accelerate 기반 통합 훈련기
4. **AccelerateFloorPlanGenerator**: 추론 엔진
5. **FloorPlanPostProcessor**: 이미지 후처리

---

## 4. 주요 구현 사항

### 4.1 LoRA 파인튜닝
```python
# LoRA 설정
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,          # alpha
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1
)
```

### 4.2 VAE 인코딩/디코딩
- **인코더**: 256×256×3 → 32×32×4 (16배 압축)
- **디코더**: 32×32×4 → 256×256×3
- **목적**: 메모리 효율성 및 학습 안정성

### 4.3 혼합 정밀도 훈련
```yaml
mixed_precision: fp16  # 메모리 절약
```

### 4.4 그래디언트 누적
```yaml
gradient_accumulation_steps: 4  # 효과적 배치 크기 증가
```

---

## 5. 데이터 처리 파이프라인

### 5.1 데이터 구조
```
data/train/
├── input_text/        # 텍스트 파일 (n.txt)
└── label_floorplan/   # 이미지 파일 (n.png)
```

### 5.2 데이터 증강 전략

#### 5.2.1 무작위 마스킹
```python
# 각 섹션에서 20-50% 항목 무작위 제거
room_mask_prob: 0.3
connection_mask_prob: 0.3
position_mask_prob: 0.3
```

#### 5.2.2 순서 변경
```python
# 연결관계: "A and B connected" ↔ "B and A connected"
# 위치관계: "A is left-below B" ↔ "B is right-above A"
swap_prob: 0.5
```

#### 5.2.3 위치 관계 대칭성
```python
position_mappings = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "left-below": "right-above",
    "left-above": "right-below"
}
```

### 5.3 토큰화 과정
```python
# CLIP 토큰화 (기본 77토큰 제한)
tokenizer(text, max_length=77, truncation=True, return_tensors="pt")
```

---

## 6. 모델 구조

### 6.1 Stable Diffusion v1.5 구성
```
1. Text Encoder: CLIP ViT-L/14 (768dim 임베딩)
2. U-Net: 860M parameters (노이즈 예측)
3. VAE: 84M parameters (이미지 ↔ 잠재공간 변환)
```

### 6.2 LoRA 적용 대상
```python
# U-Net
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

# Text Encoder  
target_modules = ["q_proj", "v_proj"]
```

### 6.3 손실 함수
```python
# MSE Loss (노이즈 예측)
loss = F.mse_loss(noise_pred, noise)
```

---

## 7. 훈련 시스템

### 7.1 FloorPlanTrainer 클래스
- **기반**: Hugging Face Accelerate
- **기능**: 분산학습, 혼합정밀도, 체크포인트 관리
- **로깅**: Weights & Biases 연동

### 7.2 훈련 과정
```python
1. VAE 인코딩: 이미지 → 잠재 공간
2. 노이즈 추가: 순방향 디퓨전
3. 노이즈 예측: U-Net 추론
4. 손실 계산: MSE Loss
5. 역전파: LoRA 파라미터만 업데이트
```

### 7.3 주요 하이퍼파라미터
```yaml
learning_rate: 1e-4
batch_size: 2
num_epochs: 100
optimizer: AdamW
scheduler_type: cosine
warmup_steps: 1000
```

---

## 8. 추론 시스템

### 8.1 AccelerateFloorPlanGenerator
- **입력**: 텍스트 조건, 체크포인트 경로
- **출력**: 생성된 평면도 이미지
- **기능**: 단일/배치/대화형 생성

### 8.2 생성 과정
```python
1. 텍스트 인코딩: CLIP → 임베딩
2. 잠재 노이즈: 랜덤 초기화
3. 디노이징: U-Net 반복 실행
4. VAE 디코딩: 잠재 → 이미지
5. 후처리: 색상 정규화
```

### 8.3 생성 파라미터
```python
num_inference_steps: 50      # 추론 스텝 수
guidance_scale: 7.5          # 텍스트 가이던스 강도
seed: Optional[int]          # 재현성을 위한 시드
```

---

## 9. 분산 학습 시스템

### 9.1 2단계 방식 (Windows 최적화)

#### 9.1.1 1단계: 워커 설정 (scripts/worker_setup.ps1)
```powershell
# 각 워커 노드에서 1회 실행
.\scripts\worker_setup.ps1 -MasterIP "마스터_IP"
```

**수행 작업:**
- PowerShell Remoting 활성화
- 방화벽 설정 (포트 29500)
- 신뢰할 수 있는 호스트 설정
- 전원 관리 최적화

#### 9.1.2 2단계: 마스터 배포 (scripts/master_deploy.ps1)
```powershell
# 마스터 노드에서 실행
.\scripts\master_deploy.ps1
```

**수행 작업:**
- 워커 IP 자동 탐지/입력
- 코드 및 데이터 복사 (SMB/SCP)
- uv 설치 및 의존성 설치
- Accelerate 설정 파일 생성
- 분산 훈련 실행

### 9.2 Accelerate 설정

#### 9.2.1 단일 GPU
```yaml
# configs/accelerate/single_gpu.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: fp16
num_processes: 1
```

#### 9.2.2 동적 멀티노드 설정
```yaml
# 런타임에 자동 생성
distributed_type: MULTI_GPU
num_machines: <워커수+1>
num_processes: <총GPU수>
machine_rank: <노드별개별설정>
main_process_ip: <마스터IP>
main_process_port: 29500
```

---

## 10. 텍스트 처리 고급 기능

### 10.1 CLIP 77토큰 제한 문제
**문제**: 구조화된 긴 텍스트가 77토큰을 초과
**해결**: 3가지 방법 제공

### 10.2 방법 1: 스마트 자르기 (기본값)
```python
# 중요도 기반 정보 보존
priorities = [
    "Number and Type of Rooms",     # 높음
    "Connection Between Rooms",     # 중간
    "Positional Relationship"       # 낮음
]
```

### 10.3 방법 2: Attention 기반 텍스트 청킹
```python
class CLIPTextChunker:
    def chunk_text(self, text, max_length=77):
        # 텍스트를 청크로 분할
        
class CLIPTextEmbeddingAggregator:
    def __init__(self, aggregation_method="attention"):  # 기본값
        self.aggregation_method = aggregation_method
    
    def aggregate_embeddings(self, embeddings, chunk_lengths):
        if self.aggregation_method == "attention":
            return self._attention_aggregation(embeddings, chunk_lengths)
```

#### 10.3.1 Attention 집계 원리
```python
def _attention_aggregation(self, embeddings, chunk_lengths):
    # 1. 각 청크 임베딩에 대해 attention 점수 계산
    attention_scores = self._compute_attention_scores(embeddings)
    
    # 2. Softmax로 정규화
    attention_weights = F.softmax(attention_scores, dim=0)
    
    # 3. 가중 평균으로 최종 임베딩 생성
    final_embedding = torch.sum(attention_weights * embeddings, dim=0)
    return final_embedding
```

### 10.4 방법 3: 확장된 CLIP (개념적)
```python
# src/models/extended_clip.py
# 위치 임베딩 확장으로 더 긴 시퀀스 처리
class ExtendedCLIPTextModel:
    def extend_position_embeddings(self, new_max_length):
        # 기존 위치 임베딩 보간/복제
```

### 10.5 성능 비교
| 방법 | 정보 보존율 | 처리 시간 | 품질 | 메모리 |
|------|-------------|-----------|------|--------|
| 기본 자르기 | 30% | 1x | 낮음 | 낮음 |
| 스마트 자르기 | 80% | 1x | 중간 | 낮음 |
| 텍스트 청킹 | 100% | 5x | 높음 | 중간 |

---

## 11. 발생한 문제들과 해결방법

### 11.1 초기 설정 문제들

#### 11.1.1 uv 의존성 충돌
**문제**: `flake8`의 Python 버전 요구사항 충돌
**해결**: `pyproject.toml`에서 `requires-python = ">=3.9"`로 수정

#### 11.1.2 hatchling 빌드 오류
**문제**: 패키지 파일을 찾을 수 없음
**해결**: `pyproject.toml`에 `packages = ["src"]` 추가

### 11.2 모듈 임포트 문제들

#### 11.2.1 FloorPlanDataLoader 임포트 오류
**문제**: `ImportError: cannot import name 'FloorPlanDataLoader'`
**해결**: `src/data/__init__.py`의 `__all__`에 추가

#### 11.2.2 순환 임포트 문제
**문제**: `FloorPlanGenerator` 순환 임포트
**해결**: `src/inference/__init__.py`에서 직접 임포트 제거

### 11.3 LoRA 설정 문제들

#### 11.3.1 잘못된 task_type
**문제**: `AttributeError: 'LoraModel' object has no attribute 'prepare_inputs_for_generation'`
**해결**: `LoraConfig`에서 `task_type="CAUSAL_LM"` 제거

#### 11.3.2 파라미터 접근 오류
**문제**: `AttributeError: 'StableDiffusionPipeline' object has no attribute 'named_parameters'`
**해결**: `pipeline.unet.named_parameters()` 직접 접근

### 11.4 데이터 처리 문제들

#### 11.4.1 배치 키 불일치
**문제**: `KeyError: 'pixel_values'`
**해결**: 데이터셋에서 `'image'`, `'text'` 키 사용하도록 수정

#### 11.4.2 Windows 멀티프로세싱 오류
**문제**: `RuntimeError: An attempt has been made to start a new process`
**해결**: `num_workers: 0` 설정 (Windows 전용)

### 11.5 모델 차원 불일치 문제들

#### 11.5.1 U-Net 입력 채널 오류
**문제**: `expected input[4, 3, 256, 256] to have 4 channels, but got 3 channels`
**해결**: VAE 인코딩으로 RGB → 잠재공간 변환 추가

#### 11.5.2 학습률 타입 오류
**문제**: `TypeError: '<=' not supported between instances of 'float' and 'str'`
**해결**: YAML에서 `1e-4` → `0.0001` 변경

### 11.6 Accelerate 체크포인트 문제들

#### 11.6.1 체크포인트 구조 불일치
**문제**: LoRA 개별 파일 vs Accelerate 통합 체크포인트
**해결**: `AccelerateFloorPlanGenerator`에서 `FloorPlanTrainer` 통해 로딩

#### 11.6.2 이미지 변환 타입 오류
**문제**: `AttributeError: 'list' object has no attribute 'squeeze'`
**해결**: 파이프라인 출력 형식 처리 로직 개선

### 11.7 텍스트 처리 문제들

#### 11.7.1 numpy 배열 메서드 오류
**문제**: `AttributeError: 'numpy.ndarray' object has no attribute 'dim'`
**해결**: `image.dim()` → `image.ndim` 변경

#### 11.7.2 어텐션 차원 불일치
**문제**: `IndexError: tuple index out of range`
**해결**: 스칼라 attention_scores 처리 로직 추가

---

## 12. 파일별 상세 설명

### 12.1 핵심 설정 파일

#### pyproject.toml
```toml
[project]
name = "floorplan-diffusion"
version = "0.1.0"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "diffusers>=0.21.0",
    "transformers>=4.30.0",
    "accelerate>=0.20.0",
    "peft>=0.4.0",
    # ... 기타 의존성
]

[tool.hatch.build.targets.wheel]
packages = ["src"]
```

#### configs/train_config.yaml
```yaml
training:
  model_name: "runwayml/stable-diffusion-v1-5"
  lora_rank: 16
  lora_alpha: 32
  learning_rate: 1e-4
  batch_size: 2
  num_epochs: 100
  mixed_precision: "fp16"

data:
  train_text_dir: "./data/train/input_text"
  train_image_dir: "./data/train/label_floorplan"
  num_workers: 0  # Windows 호환성
```

### 12.2 핵심 소스 코드

#### src/data/dataset.py
```python
class FloorPlanDataset(Dataset):
    def __init__(self, text_dir, image_dir, tokenizer, transform=None):
        self.text_files = sorted(glob.glob(os.path.join(text_dir, "*.txt")))
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        # 텍스트 로딩 및 토큰화
        with open(self.text_files[idx], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 이미지 로딩 및 전처리
        image = Image.open(self.image_files[idx]).convert('RGB')
        
        return {
            'text': text,
            'image': self.transform(image) if self.transform else image
        }
```

#### src/data/text_chunking.py
```python
class CLIPTextEmbeddingAggregator:
    def __init__(self, aggregation_method="attention"):  # 기본값: attention
        self.aggregation_method = aggregation_method
    
    def _attention_aggregation(self, embeddings, chunk_lengths):
        # Query: 전체 임베딩의 평균
        query = torch.mean(embeddings, dim=0, keepdim=True)
        
        # Key = Value: 각 청크 임베딩
        keys = values = embeddings
        
        # Attention 점수 계산
        attention_scores = torch.matmul(query, keys.transpose(0, 1))
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 가중 평균으로 최종 임베딩 생성
        final_embedding = torch.matmul(attention_weights, values)
        return final_embedding.squeeze(0)
```

#### src/models/lora_wrapper.py
```python
class LoRAWrapper:
    def __init__(self, pipeline, lora_rank=16, lora_alpha=32):
        # U-Net LoRA 설정
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
        )
        self.pipeline.unet = get_peft_model(self.pipeline.unet, unet_lora_config)
        
        # Text Encoder LoRA 설정
        text_encoder_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
        )
        self.pipeline.text_encoder = get_peft_model(self.pipeline.text_encoder, text_encoder_lora_config)
```

#### src/training/trainer.py
```python
class FloorPlanTrainer:
    def __init__(self, config):
        self._setup_accelerator()
        self.model = FloorPlanDiffusionModel(config)
        self._prepare_accelerate()
    
    def _setup_accelerator(self):
        self.accelerator = Accelerator(
            mixed_precision=self.config.get('mixed_precision', 'fp16'),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            project_config=ProjectConfiguration(project_dir="./checkpoints"),
        )
    
    def _train_step_accelerate(self, batch):
        # VAE 인코딩: RGB → 잠재공간
        with torch.no_grad():
            latents = self.model.pipeline.vae.encode(batch['image']).latent_dist.sample()
            latents = latents * 0.18215
        
        # 노이즈 추가
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 텍스트 인코딩
        text_inputs = self.tokenizer(batch['text'], max_length=77, truncation=True, return_tensors="pt")
        encoder_hidden_states = self.model.pipeline.text_encoder(text_inputs.input_ids.to(latents.device))[0]
        
        # U-Net 예측
        noise_pred = self.model.pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # 손실 계산
        loss = F.mse_loss(noise_pred, noise)
        return loss
```

#### src/inference/generator.py
```python
class AccelerateFloorPlanGenerator:
    def __init__(self, checkpoint_path, config_path):
        self.config = self._load_config(config_path)
        self.trainer = FloorPlanTrainer(self.config)
        self.trainer.load_checkpoint(checkpoint_path)
        
    def generate_single(self, text, num_inference_steps=50, guidance_scale=7.5, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # 파이프라인 생성
        result = self.trainer.model.pipeline(
            text,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_dict=True
        )
        
        # 이미지 추출 및 변환
        if isinstance(result, dict) and 'images' in result:
            image = result['images'][0]
        else:
            image = result[0] if isinstance(result, list) else result
        
        # PIL.Image → numpy 변환
        if hasattr(image, 'convert'):
            image = np.array(image.convert('RGB'))
        
        return image
```

### 12.3 분산학습 스크립트

#### scripts/worker_setup.ps1
```powershell
param(
    [Parameter(Mandatory=$true)]
    [string]$MasterIP
)

# PowerShell Remoting 활성화
Enable-PSRemoting -Force -SkipNetworkProfileCheck

# 방화벽 규칙 추가
New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -Protocol TCP -LocalPort 29500 -Action Allow

# 신뢰할 수 있는 호스트 설정
Set-Item WSMan:\localhost\Client\TrustedHosts -Value $MasterIP -Force

# 전원 관리 최적화
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # 고성능

Write-Host "워커 노드 설정 완료: $env:COMPUTERNAME"
```

#### scripts/master_deploy.ps1
```powershell
# 워커 IP 수집
$WorkerIPs = @()
if ($WorkerIPs.Count -eq 0) {
    # 대화형 입력
    do {
        $ip = Read-Host "워커 노드 IP 입력 (완료시 Enter)"
        if ($ip) { $WorkerIPs += $ip }
    } while ($ip)
}

# 각 워커에 배포
foreach ($workerIP in $WorkerIPs) {
    # 코드 복사
    Copy-Item -Path ".\*" -Destination "\\$workerIP\C$\distributed_training\" -Recurse -Force
    
    # 원격 실행
    Invoke-Command -ComputerName $workerIP -ScriptBlock {
        cd C:\distributed_training
        
        # uv 설치
        if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
            pip install uv
        }
        
        # 의존성 설치
        uv sync
    }
}

# Accelerate 설정 생성 및 분산 훈련 시작
$masterIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*"}).IPAddress
$numMachines = $WorkerIPs.Count + 1
$numProcesses = $numMachines * 2  # 노드당 2 GPU 가정

# 마스터 노드 훈련 시작
accelerate launch --config_file "configs/accelerate/master_config.yaml" scripts/train.py
```

### 12.4 테스트 및 유틸리티

#### scripts/test_token_limits.py
```python
def main():
    # 텍스트 청킹 테스트
    chunker = CLIPTextChunker()
    aggregator = CLIPTextEmbeddingAggregator()  # 기본값: attention
    
    # 성능 벤치마크
    methods = ["truncate", "chunk", "smart_truncate"]
    for method in methods:
        start_time = time.time()
        result = process_text(sample_text, method)
        elapsed = time.time() - start_time
        print(f"{method}: {elapsed:.3f}초")
```

---

## 13. 설정 및 실행 방법

### 13.1 초기 설정
```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd text-to-image-generation

# 2. uv 설치 및 의존성 설치
pip install uv
uv sync

# 3. 가상환경 활성화 (Windows)
.\.venv\Scripts\activate
```

### 13.2 단일 GPU 훈련
```bash
python scripts/train.py --config configs/train_config.yaml
```

### 13.3 멀티노드 분산훈련 (Windows)
```powershell
# 1단계: 각 워커에서 1회 실행
.\scripts\worker_setup.ps1 -MasterIP "마스터_IP"

# 2단계: 마스터에서 실행
.\scripts\master_deploy.ps1
```

### 13.4 평면도 생성
```bash
# 단일 생성
python scripts/generate.py --checkpoint checkpoints/checkpoints --text "SJH-Style FloorPlan Generation [Number and Type of Rooms] The floorplan have 1 living room, 1 kitchen"

# 대화형 생성
python scripts/generate.py --checkpoint checkpoints/checkpoints --interactive
```

### 13.5 텍스트 청킹 테스트
```bash
python scripts/test_token_limits.py
```

---

## 14. 개발 히스토리

### 14.1 Phase 1: 기본 구현 (초기)
- Stable Diffusion + LoRA 기본 파인튜닝 구현
- 데이터셋 및 기본 훈련 파이프라인 구축
- 초기 데이터 증강 전략 개발

### 14.2 Phase 2: Accelerate 전환
- `torch.distributed` → Hugging Face Accelerate 마이그레이션
- 통합 훈련기 (`FloorPlanTrainer`) 개발
- 분산학습 지원 구축

### 14.3 Phase 3: 추론 시스템 구축
- `AccelerateFloorPlanGenerator` 개발
- Accelerate 체크포인트 호환성 구현
- 후처리 파이프라인 구축

### 14.4 Phase 4: 텍스트 처리 고도화
- CLIP 77토큰 제한 문제 해결
- Attention 기반 텍스트 청킹 구현
- 3가지 텍스트 처리 방법 제공

### 14.5 Phase 5: 분산학습 자동화
- Windows 환경 최적화
- 2단계 분산학습 방식 개발
- PowerShell 기반 자동화 스크립트

### 14.6 Phase 6: 프로젝트 정리
- 미사용 파일 정리
- 문서 체계 정비
- 종합 가이드 작성

---

## 🎯 핵심 혁신사항 요약

1. **Attention 기반 텍스트 청킹**: CLIP 77토큰 제한을 우아하게 해결
2. **2단계 분산학습**: Windows 환경에 최적화된 자동화 시스템
3. **통합 훈련 시스템**: Accelerate 기반 단일 스크립트로 모든 환경 지원
4. **지능적 데이터 증강**: 실제 사용 패턴을 반영한 부분 조건 처리

---

**이 문서는 프로젝트의 모든 기술적 세부사항을 포함하며, 새로운 세션에서 즉시 개발을 이어갈 수 있도록 구성되었습니다.** 🚀
