"""
Diffusion model for floorplan generation
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from diffusers import StableDiffusionPipeline, DDIMScheduler
from .lora_wrapper import LoRAWrapper


class FloorPlanDiffusionModel:
    """평면도 생성을 위한 디퓨전 모델 클래스"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: 기본 모델 이름
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            device: 디바이스
        """
        self.model_name = model_name
        self.device = device
        
        # LoRA 래퍼 초기화
        self.lora_wrapper = LoRAWrapper(
            model_name=model_name,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # 파이프라인 로드
        self.pipeline = self.lora_wrapper.load_model(device=device)
        
        # 스케줄러 설정
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        
        # 토크나이저
        self.tokenizer = self.pipeline.tokenizer
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        noise_scheduler,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, float]:
        """
        훈련 스텝
        
        Args:
            batch: 배치 데이터
            noise_scheduler: 노이즈 스케줄러
            optimizer: 옵티마이저
            scaler: 그래디언트 스케일러 (AMP용)
            
        Returns:
            손실 정보
        """
        # 모델을 훈련 모드로 설정
        self.lora_wrapper.prepare_for_training()
        
        # 배치 데이터 추출
        text_ids = batch['text'].to(self.device)
        images = batch['image'].to(self.device)
        
        # 노이즈 추가
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (images.shape[0],), device=images.device)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
        
        # 텍스트 인코딩
        with torch.no_grad():
            text_embeddings = self.pipeline.text_encoder(text_ids)[0]
        
        # UNet 예측
        noise_pred = self.pipeline.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=text_embeddings
        ).sample
        
        # 손실 계산
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        # 역전파
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        return {
            'loss': loss.item(),
            'noise_pred_norm': noise_pred.norm().item(),
            'noise_norm': noise.norm().item()
        }
    
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 256,
        width: int = 256
    ) -> torch.Tensor:
        """
        평면도 생성
        
        Args:
            prompt: 조건 텍스트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
            seed: 랜덤 시드
            height: 이미지 높이
            width: 이미지 너비
            
        Returns:
            생성된 평면도 이미지
        """
        # 모델을 추론 모드로 설정
        self.lora_wrapper.prepare_for_inference()
        
        # 시드 설정
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # 이미지 생성
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                return_dict=False
            )[0]
        
        return image
    
    def save_model(self, save_path: str):
        """
        모델 저장
        
        Args:
            save_path: 저장 경로
        """
        # LoRA 가중치 저장
        self.lora_wrapper.save_lora_weights(save_path)
        
        # 설정 정보 저장
        config = {
            'model_name': self.model_name,
            'lora_rank': self.lora_wrapper.lora_rank,
            'lora_alpha': self.lora_wrapper.lora_alpha,
            'lora_dropout': self.lora_wrapper.lora_dropout
        }
        
        torch.save(config, f"{save_path}/config.pt")
    
    def load_model(self, load_path: str):
        """
        모델 로드
        
        Args:
            load_path: 로드 경로
        """
        # LoRA 가중치 로드
        self.lora_wrapper.load_lora_weights(load_path)
        
        # 설정 정보 로드
        config = torch.load(f"{load_path}/config.pt")
        print(f"Loaded model config: {config}")
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """훈련 가능한 파라미터 수 반환"""
        return self.lora_wrapper.get_trainable_parameters()
    
    def print_trainable_parameters(self):
        """훈련 가능한 파라미터 정보 출력"""
        self.lora_wrapper.print_trainable_parameters()
    
    def set_gradient_checkpointing(self, enabled: bool = True):
        """그래디언트 체크포인팅 설정"""
        self.lora_wrapper.set_gradient_checkpointing(enabled)
    
    def to_device(self, device: str):
        """모델을 특정 디바이스로 이동"""
        self.device = device
        self.pipeline = self.pipeline.to(device)
    
    def get_tokenizer(self):
        """토크나이저 반환"""
        return self.tokenizer 