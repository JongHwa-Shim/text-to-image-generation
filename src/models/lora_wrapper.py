"""
LoRA wrapper for Stable Diffusion fine-tuning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model


class LoRAWrapper:
    """LoRA 파인튜닝을 위한 래퍼 클래스"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ):
        """
        Args:
            model_name: 기본 모델 이름
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: LoRA를 적용할 모듈들
        """
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 기본 target modules 설정
        if target_modules is None:
            self.target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc1",
                "fc2",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "conv1",
                "conv2",
                "conv_shortcut",
                "time_emb_proj",
                "conv_out"
            ]
        else:
            self.target_modules = target_modules
        
        self.pipeline = None
        self.unet = None
        self.text_encoder = None
    
    def load_model(self, device: str = "cuda") -> StableDiffusionPipeline:
        """
        모델을 로드하고 LoRA 설정을 적용
        
        Args:
            device: 디바이스
            
        Returns:
            LoRA가 적용된 파이프라인
        """
        # 기본 파이프라인 로드
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # LoRA 설정 (Diffusion 모델용)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none"
            # task_type은 Diffusion 모델에서 사용하지 않음
        )
        
        # UNet에 LoRA 적용
        self.unet = get_peft_model(self.pipeline.unet, lora_config)
        self.pipeline.unet = self.unet
        
        # Text Encoder에 LoRA 적용
        text_encoder_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.lora_dropout,
            bias="none"
            # task_type은 Text Encoder에서도 제거
        )
        
        self.text_encoder = get_peft_model(self.pipeline.text_encoder, text_encoder_lora_config)
        self.pipeline.text_encoder = self.text_encoder
        
        # 디바이스로 이동
        self.pipeline = self.pipeline.to(device)
        
        return self.pipeline
    
    def save_lora_weights(self, save_path: str):
        """
        LoRA 가중치 저장
        
        Args:
            save_path: 저장 경로
        """
        if self.unet is not None:
            self.unet.save_pretrained(f"{save_path}/unet")
        
        if self.text_encoder is not None:
            self.text_encoder.save_pretrained(f"{save_path}/text_encoder")
    
    def load_lora_weights(self, load_path: str):
        """
        LoRA 가중치 로드
        
        Args:
            load_path: 로드 경로
        """
        if self.unet is not None:
            self.unet.load_state_dict(
                torch.load(f"{load_path}/unet/adapter_model.bin"),
                strict=False
            )
        
        if self.text_encoder is not None:
            self.text_encoder.load_state_dict(
                torch.load(f"{load_path}/text_encoder/adapter_model.bin"),
                strict=False
            )
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        훈련 가능한 파라미터 수 반환
        
        Returns:
            모델별 훈련 가능한 파라미터 수
        """
        trainable_params = {}
        
        if self.unet is not None:
            trainable_params["unet"] = sum(
                p.numel() for p in self.unet.parameters() if p.requires_grad
            )
        
        if self.text_encoder is not None:
            trainable_params["text_encoder"] = sum(
                p.numel() for p in self.text_encoder.parameters() if p.requires_grad
            )
        
        return trainable_params
    
    def print_trainable_parameters(self):
        """훈련 가능한 파라미터 정보 출력"""
        trainable_params = self.get_trainable_parameters()
        
        print("Trainable parameters:")
        for model_name, param_count in trainable_params.items():
            print(f"  {model_name}: {param_count:,} parameters")
        
        total_params = sum(trainable_params.values())
        print(f"Total trainable parameters: {total_params:,}")
    
    def set_gradient_checkpointing(self, enabled: bool = True):
        """
        그래디언트 체크포인팅 설정
        
        Args:
            enabled: 활성화 여부
        """
        if self.unet is not None:
            self.unet.enable_gradient_checkpointing()
        
        if self.text_encoder is not None:
            self.text_encoder.gradient_checkpointing_enable()
    
    def prepare_for_training(self):
        """훈련을 위한 모델 준비"""
        # 훈련 모드로 설정
        if self.unet is not None:
            self.unet.train()
        
        if self.text_encoder is not None:
            self.text_encoder.train()
        
        # 그래디언트 체크포인팅 활성화
        self.set_gradient_checkpointing(True)
    
    def prepare_for_inference(self):
        """추론을 위한 모델 준비"""
        # 평가 모드로 설정
        if self.unet is not None:
            self.unet.eval()
        
        if self.text_encoder is not None:
            self.text_encoder.eval()
        
        # 그래디언트 체크포인팅 비활성화
        self.set_gradient_checkpointing(False) 