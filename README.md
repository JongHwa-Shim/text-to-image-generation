# 텍스트 조건 평면도 생성 모델

이 프로젝트는 텍스트 조건을 입력받아 건물 평면도를 생성하는 디퓨전 모델을 구현합니다. LoRA(Low-Rank Adaptation)를 사용한 Stable Diffusion 파인튜닝을 기반으로 하며, Hugging Face Accelerate를 통한 분산 학습을 지원합니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [동작 원리](#동작-원리)
- [기반 모델 정보](#기반-모델-정보)
- [데이터 처리](#데이터-처리)
- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [설정 파일](#설정-파일)
- [분산 학습](#분산-학습)
- [결과 예시](#결과-예시)
- [문제 해결](#문제-해결)

## 🏗️ 프로젝트 개요

### 목표
- **입력**: 방의 개수/종류, 연결관계, 위치관계를 묘사하는 구조화된 텍스트
- **출력**: 256x256 RGB 평면도 이미지 (각 영역이 고유 색상으로 구분됨)
- **특징**: 실제 사용성을 고려한 부분 조건 입력 지원

### 주요 특징
- 🎯 **구조화된 텍스트 조건**: 방의 개수, 연결관계, 위치관계 등 정확한 제약 조건 처리
- 🔄 **데이터 증강**: 부분 조건 입력을 위한 무작위 마스킹 및 텍스트 변형
- ⚡ **효율적인 파인튜닝**: LoRA를 사용한 메모리 효율적인 모델 적응
- 🚀 **2단계 분산 학습**: Windows 최적화된 워커-마스터 자동화 방식
- 🧠 **Attention 기반 텍스트 청킹**: CLIP 77토큰 제한 극복
- 🎨 **후처리**: 생성된 이미지의 노이즈 제거 및 색상 정규화

## 🧠 동작 원리

### 1. 텍스트 인코딩
```
입력 텍스트 → 청킹/스마트자르기 → CLIP Text Encoder → 텍스트 임베딩
```
- **CLIP 모델**: OpenAI의 CLIP을 사용하여 텍스트를 768차원 벡터로 인코딩
- **구조화된 프롬프트**: 정해진 형식으로 방 정보, 연결관계, 위치관계를 표현
- **토큰 제한 극복**: Attention 기반 청킹으로 77토큰 제한 해결
- **스마트 처리**: 중요 정보 우선 보존 또는 전체 정보 청킹 후 병합

### 2. 이미지 생성 과정
```
텍스트 임베딩 → U-Net (조건부 노이즈 예측) → VAE 디코더 → 최종 이미지
```

#### U-Net 디퓨전 과정:
1. **순방향**: 깨끗한 이미지에 노이즈를 점진적으로 추가
2. **역방향**: 노이즈에서 시작하여 텍스트 조건에 맞는 이미지로 복원
3. **조건부 생성**: 각 스텝에서 텍스트 임베딩을 참조하여 방향 조정

#### VAE (Variational Autoencoder):
- **인코더**: 256×256×3 이미지 → 32×32×4 잠재 공간
- **디코더**: 32×32×4 잠재 표현 → 256×256×3 이미지
- **장점**: 메모리 효율성 (16배 압축)

### 3. LoRA 파인튜닝
```
사전훈련된 Stable Diffusion + LoRA 어댑터 = 평면도 특화 모델
```
- **효율성**: 전체 모델의 ~1%만 학습 (메모리 절약)
- **유연성**: 기존 모델 구조 유지하면서 새로운 도메인 적응
- **품질**: 전체 파인튜닝과 유사한 성능

## 🤖 기반 모델 정보

### Stable Diffusion v1.5
- **개발사**: RunwayML
- **아키텍처**: Latent Diffusion Model
- **훈련 데이터**: LAION-5B (50억 개 이미지-텍스트 쌍)
- **해상도**: 512×512 (본 프로젝트에서는 256×256로 조정)

### 모델 구성 요소
```
1. Text Encoder: CLIP ViT-L/14 (Text → 768dim embedding)
2. U-Net: 860M parameters (노이즈 예측)
3. VAE: 84M parameters (이미지 ↔ 잠재공간 변환)
```

### LoRA 설정
- **Rank (r)**: 16 (저차원 분해의 차원)
- **Alpha**: 32 (스케일링 팩터)
- **Target Modules**: 
  - U-Net: `to_q`, `to_k`, `to_v`, `to_out.0`
  - Text Encoder: `q_proj`, `v_proj`
- **Dropout**: 0.1

## 📊 데이터 처리

### 입력 데이터 형식
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

### 데이터 증강 전략

#### 1. 무작위 마스킹
- **목적**: 부분 정보만으로도 생성 가능한 모델 학습
- **방법**: 각 섹션에서 일부 문장을 랜덤하게 제거
- **확률**: 20-50% 항목 제거

#### 2. 순서 변경
- **연결관계**: "A and B are connected" ↔ "B and A are connected"
- **위치관계**: "A is left-below B" ↔ "B is right-above A"
- **목적**: 텍스트 순서에 robust한 모델 학습

#### 3. 위치 관계 대칭성
```python
position_mappings = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "left-below": "right-above",
    "left-above": "right-below"
}
```

### 출력 이미지 형식
- **해상도**: 256×256×3 (RGB)
- **형식**: PNG
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

### 후처리 파이프라인

#### 1. 노이즈 제거
```python
# 중간값 필터를 사용한 노이즈 제거
processed = cv2.medianBlur(image, kernel_size=3)
```

#### 2. 색상 정규화
```python
# 각 픽셀을 가장 가까운 정의된 색상으로 매핑
def normalize_colors(image):
    for pixel in image:
        closest_color = find_closest_room_color(pixel)
        pixel = closest_color
```

#### 3. 영역 정리
- 작은 고립된 영역 제거
- 영역 경계 스무딩
- 일관성 있는 색상 적용

## ✨ 주요 기능

- **디퓨전 모델 훈련**: LoRA를 사용한 효율적인 파인튜닝
- **Attention 기반 텍스트 처리**: CLIP 77토큰 제한 극복으로 긴 텍스트 완전 활용
- **2단계 분산학습**: 워커 설정(1회) + 마스터 자동배포로 멀티노드 쉽게 구축
- **유연한 생성**: 단일/배치/파일/대화형 생성 모드
- **자동 후처리**: 생성된 이미지의 노이즈 제거 및 색상 정규화
- **실험 추적**: Weights & Biases 연동

## 📁 프로젝트 구조

```
text-to-image generation/
├── configs/                    # 설정 파일
│   ├── train_config.yaml      # 메인 훈련 설정
│   └── accelerate/            # Accelerate 설정들
│       └── single_gpu.yaml    # 단일 GPU 설정
├── data/                      # 데이터셋
│   └── train/
│       ├── input_text/        # 조건 텍스트 (n.txt)
│       └── label_floorplan/   # 정답 이미지 (n.png)
├── src/                       # 소스 코드
│   ├── data/                  # 데이터 처리
│   │   ├── dataset.py         # 데이터셋 클래스
│   │   ├── text_chunking.py   # Attention 기반 텍스트 청킹
│   │   └── augmentation.py    # 데이터 증강
│   ├── models/                # 모델 정의
│   │   ├── diffusion_model.py # 메인 디퓨전 모델
│   │   ├── lora_wrapper.py    # LoRA 래퍼
│   │   └── extended_clip.py   # 확장된 CLIP (개념적)
│   ├── training/              # 훈련 관련
│   │   └── trainer.py         # 통합 훈련기 (Accelerate)
│   ├── inference/             # 추론 관련
│   │   ├── generator.py       # 평면도 생성기
│   │   └── post_processor.py  # 후처리기
│   └── utils/                 # 유틸리티
│       ├── config_utils.py    # 설정 로드
│       ├── logging_utils.py   # 로깅
│       └── visualization.py   # 시각화
├── scripts/                   # 실행 스크립트
│   ├── train.py              # 훈련 스크립트
│   ├── generate.py           # 생성 스크립트
│   ├── test_token_limits.py  # 텍스트 청킹 테스트
│   ├── worker_setup.ps1      # 워커 노드 설정 (Windows)
│   └── master_deploy.ps1     # 마스터 자동 배포 (Windows)
├── docs/                     # 상세 문서
│   └── distributed_training_guide.md  # 분산학습 상세 가이드
├── examples/                 # 예제 및 데모
│   ├── simple_attention_demo.py      # 어텐션 메커니즘 데모
│   └── performance_benchmark.py     # 성능 벤치마크
├── checkpoints/              # 모델 체크포인트
├── outputs/                  # 생성 결과
├── logs/                    # 로그 파일
├── README_DISTRIBUTED.md    # 분산학습 빠른 가이드
├── DISTRIBUTED_QUICK_START.md  # 초간단 분산학습 가이드
└── pyproject.toml           # 프로젝트 설정 (uv)
```

## 🛠️ 설치 및 설정

### 1. 환경 요구사항
- **Python**: 3.9+
- **CUDA**: 11.0+ (GPU 사용 시)
- **메모리**: 최소 16GB RAM, 8GB+ VRAM 권장

### 2. 의존성 설치
이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다.

```bash
# uv 설치 (아직 설치하지 않은 경우)
pip install uv

# 프로젝트 클론
git clone <repository-url>
cd text-to-image-generation

# 의존성 설치
uv sync

# 가상환경 활성화 (Windows)
.\.venv\Scripts\activate

# 가상환경 활성화 (Unix/macOS)
source .venv/bin/activate
```

## 📖 사용법

### 1. 모델 훈련

#### 단일 GPU 훈련
```bash
python scripts/train.py --config configs/train_config.yaml
```

#### 단일 노드 멀티 GPU 훈련
```bash
accelerate launch --config_file configs/accelerate/single_gpu.yaml scripts/train.py
```

#### 멀티 노드 분산 훈련 (Windows 2단계 방식)

**1단계: 워커 노드 설정 (각 워커에서 1회)**
```powershell
# 관리자 권한 PowerShell에서 실행
.\scripts\worker_setup.ps1 -MasterIP "마스터_IP"
```

**2단계: 마스터에서 자동 배포 및 훈련**
```powershell
# 대화형 모드 (권장)
.\scripts\master_deploy.ps1

# 또는 워커 IP 직접 지정
.\scripts\master_deploy.ps1 -WorkerIPs @("워커1_IP", "워커2_IP")
```

> 자세한 분산학습 가이드는 [README_DISTRIBUTED.md](README_DISTRIBUTED.md) 또는 [DISTRIBUTED_QUICK_START.md](DISTRIBUTED_QUICK_START.md)를 참조하세요.

### 2. 평면도 생성

#### 단일 텍스트 생성
```bash
python scripts/generate.py \
    --checkpoint checkpoints/checkpoints \
    --text "SJH-Style FloorPlan Generation [Number and Type of Rooms] The floorplan have 1 living room, 1 kitchen, 1 bathroom"
```

#### 파일에서 배치 생성
```bash
python scripts/generate.py \
    --checkpoint checkpoints/checkpoints \
    --text-file prompts.txt \
    --output-dir results/
```

#### 대화형 생성
```bash
python scripts/generate.py \
    --checkpoint checkpoints/checkpoints \
    --interactive
```

### 3. 생성 옵션

```bash
python scripts/generate.py \
    --checkpoint checkpoints/checkpoints \
    --text "your prompt here" \
    --num-inference-steps 50 \     # 추론 스텝 수 (품질↑, 시간↑)
    --guidance-scale 7.5 \         # 가이던스 강도 (높을수록 텍스트 조건 강하게 반영)
    --seed 42 \                    # 재현가능한 결과를 위한 시드
    --output-dir custom_output/
```

## ⚙️ 설정 파일

### `configs/train_config.yaml`
```yaml
# 모델 설정
training:
  model_name: "runwayml/stable-diffusion-v1-5"  # 기반 모델
  
  # LoRA 설정
  lora_rank: 16                    # LoRA rank (낮을수록 파라미터 적음)
  lora_alpha: 32                   # LoRA alpha (학습률 스케일)
  lora_dropout: 0.1                # LoRA dropout
  
  # 훈련 하이퍼파라미터
  learning_rate: 0.0001            # 학습률
  batch_size: 2                    # 배치 크기
  num_epochs: 100                  # 에포크 수
  
  # 최적화
  optimizer: "AdamW"
  weight_decay: 0.01
  gradient_accumulation_steps: 4   # 그래디언트 누적
  
  # 스케줄러
  scheduler_type: "cosine"
  warmup_steps: 1000
  
  # 로깅 및 체크포인트
  log_every: 10                    # 로그 출력 주기
  save_every: 50                   # 체크포인트 저장 주기
  
  # 혼합 정밀도 및 실험 추적
  mixed_precision: "fp16"          # 메모리 절약
  wandb:
    project: "floorplan-diffusion"
    entity: "your-wandb-username"

# 데이터 설정
data:
  train_text_dir: "./data/train/input_text"
  train_image_dir: "./data/train/label_floorplan"
  val_split: 0.2                   # 검증 데이터 비율
  num_workers: 0                   # 데이터 로딩 워커 수 (Windows에서는 0)
  
  # 데이터 증강
  augmentation:
    room_mask_prob: 0.3            # 방 정보 마스킹 확률
    connection_mask_prob: 0.3      # 연결 정보 마스킹 확률
    position_mask_prob: 0.3        # 위치 정보 마스킹 확률
    swap_prob: 0.5                 # 순서 바꾸기 확률
```

## 🧠 텍스트 처리 고급 기능

### CLIP 토큰 제한 해결 방법

프로젝트는 CLIP의 77토큰 제한을 극복하기 위한 다양한 방법을 제공합니다:

#### 방법 1: Attention 기반 스마트 자르기 (기본값)
```yaml
# configs/train_config.yaml에 자동 설정됨
# 중요한 정보 우선 보존, 나머지는 attention으로 지능적 처리
```

#### 방법 2: 텍스트 청킹
```bash
# 모든 정보를 청크로 나누어 attention으로 병합
python scripts/test_token_limits.py
```

#### 방법 3: 확장된 CLIP (개념적)
```python
# src/models/extended_clip.py 참조
# 위치 임베딩 확장으로 더 긴 시퀀스 처리
```

### 성능 비교
| 방법 | 정보 보존율 | 처리 시간 | 추천도 |
|------|-------------|-----------|--------|
| 기본 자르기 | 30% | 1배 | ❌ |
| 스마트 자르기 | 80% | 1배 | ⭐⭐⭐ |
| 텍스트 청킹 | 100% | 5배 | ⭐⭐⭐⭐ |

## 📊 결과 예시

### 입력 텍스트
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

### 출력 파일
- `generation_raw_000.png`: 모델이 직접 생성한 원본 이미지
- `generation_processed_000.png`: 후처리된 최종 이미지

### 성능 지표
- **생성 시간**: ~5초 (RTX 3080, 50 스텝)
- **메모리 사용량**: ~6GB VRAM (배치 크기 2)
- **훈련 시간**: ~30분 (100 에포크, 5개 샘플)

## 🔧 문제 해결

### 일반적인 문제들

#### 1. CUDA 메모리 부족
```bash
# 배치 크기 줄이기
batch_size: 1

# 그래디언트 누적으로 효과적인 배치 크기 유지
gradient_accumulation_steps: 8
```

#### 2. Windows 멀티프로세싱 에러
```yaml
# train_config.yaml에서
data:
  num_workers: 0  # Windows에서는 0으로 설정
```

#### 3. 토큰 길이 초과
기본적으로 attention 기반 스마트 처리가 적용되어 있어 대부분의 경우 문제없습니다.

**텍스트 청킹 테스트**
```bash
python scripts/test_token_limits.py
```

**수동 설정 (필요한 경우)**
```yaml
# configs/train_config.yaml
# 기본값이 이미 최적화되어 있어 수정 불필요
```

#### 4. 생성 품질 향상
```bash
# 더 많은 추론 스텝
--num-inference-steps 100

# 가이던스 스케일 조정 (7.5~15.0)
--guidance-scale 10.0

# 더 긴 훈련
num_epochs: 200
```

### 로그 확인
```bash
# 훈련 로그
tail -f logs/training.log

# 애플리케이션 로그
tail -f logs/app.log
```

### 체크포인트 관리
```python
# 최신 체크포인트 확인
ls -la checkpoints/checkpoints/

# 메타데이터 확인
python -c "import torch; print(torch.load('checkpoints/checkpoints/metadata.pt'))"
```

## 🤝 기여하기

1. **이슈 리포트**: 버그나 개선사항을 GitHub Issues에 제출
2. **코드 기여**: Pull Request를 통한 코드 개선
3. **문서 개선**: README나 코드 주석 개선
4. **데이터셋**: 더 다양한 평면도 데이터 기여

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

## 📚 참고 자료

### 논문 및 기술 문서
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)
- [Diffusers Library](https://huggingface.co/docs/diffusers/)

### 프로젝트 문서
- **[README_DISTRIBUTED.md](README_DISTRIBUTED.md)**: 분산학습 빠른 가이드
- **[DISTRIBUTED_QUICK_START.md](DISTRIBUTED_QUICK_START.md)**: 초간단 분산학습 시작
- **[docs/distributed_training_guide.md](docs/distributed_training_guide.md)**: 분산학습 상세 기술 가이드

### 예제 및 테스트
- **`scripts/test_token_limits.py`**: 텍스트 청킹 방법 비교 및 테스트
- **`examples/simple_attention_demo.py`**: Attention 메커니즘 시각적 데모
- **`examples/performance_benchmark.py`**: 성능 벤치마크 도구

---

더 자세한 정보나 도움이 필요하시면 Issues를 통해 문의해주세요! 🚀