"""
Training class for floorplan generation with Accelerate support
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from diffusers import DDIMScheduler
from transformers import CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs
import yaml
import wandb
from tqdm import tqdm
from typing import Dict, Optional
import logging

from ..models import FloorPlanDiffusionModel
from ..data import FloorPlanDataLoader


class FloorPlanTrainer:
    """평면도 생성 모델 훈련 클래스"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        
        # Accelerate 초기화
        self._setup_accelerator()
        
        # 로깅 설정
        self._setup_logging()
        
        # 모델 초기화
        self.model = FloorPlanDiffusionModel(
            model_name=self.config['training']['model_name'],
            lora_rank=self.config['training']['lora_rank'],
            lora_alpha=self.config['training']['lora_alpha'],
            lora_dropout=self.config['training']['lora_dropout'],
            device=self.accelerator.device
        )
        
        # 토크나이저
        self.tokenizer = self.model.get_tokenizer()
        
        # 데이터 로더 초기화
        self._setup_data_loaders()
        
        # 옵티마이저 및 스케줄러
        self._setup_optimizer()
        
        # 노이즈 스케줄러
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            self.config['training']['model_name'],
            subfolder="scheduler"
        )
        
        # Accelerate로 모델, 옵티마이저, 데이터로더 준비
        self._prepare_accelerate()
        
        # 훈련 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 로깅 설정
        self._setup_wandb()
        
        # 체크포인트 디렉토리 생성
        os.makedirs(self.config['training']['save_dir'], exist_ok=True)
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_accelerator(self):
        """Accelerator 설정"""
        # 프로젝트 설정
        project_config = ProjectConfiguration(
            project_dir=self.config['training']['save_dir'],
            logging_dir=os.path.join(self.config['training']['save_dir'], "logs")
        )
        
        # DDP kwargs 설정
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        # Accelerator 초기화
        self.accelerator = Accelerator(
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
            log_with="wandb" if self.config.get('wandb', {}).get('enabled', False) else None,
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps']
        )
    
    def _setup_logging(self):
        """로깅 설정 (메인 프로세스에서만)"""
        if self.accelerator.is_main_process:
            # 로그 디렉토리 생성
            os.makedirs('logs', exist_ok=True)
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/training.log'),
                    logging.StreamHandler()
                ]
            )
        else:
            # 비메인 프로세스는 WARNING 레벨만
            logging.basicConfig(level=logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_data_loaders(self):
        """데이터 로더 설정"""
        self.data_loader = FloorPlanDataLoader(
            text_dir="data/train/input_text",
            image_dir="data/train/label_floorplan",
            tokenizer=self.tokenizer,
            batch_size=self.config['training']['batch_size'],
            image_size=self.config['training']['image_size'],
            augment_config=self.config['training']['augmentation'],
            train_split=self.config['training']['train_split'],
            num_workers=self.config['training']['num_workers']
        )
        
        self.train_loader = self.data_loader.get_train_loader()
        self.val_loader = self.data_loader.get_val_loader()
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 설정"""
        # 훈련 가능한 파라미터만 선택 (UNet과 Text Encoder에서)
        trainable_params = []
        
        # UNet 파라미터 (LoRA 적용된 것만)
        for name, param in self.model.pipeline.unet.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        # Text Encoder 파라미터 (LoRA 적용된 것만)
        for name, param in self.model.pipeline.text_encoder.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
    
    def _prepare_accelerate(self):
        """Accelerate로 모델, 옵티마이저, 데이터로더 준비"""
        self.model.pipeline, self.optimizer, self.scheduler, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model.pipeline, self.optimizer, self.scheduler, self.train_loader, self.val_loader
        )
    
    def _setup_wandb(self):
        """Weights & Biases 설정"""
        if self.config.get('wandb', {}).get('enabled', False) and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.config['wandb'].get('project', 'floorplan-diffusion'),
                config=self.config
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        # 개별 구성 요소를 train 모드로 설정
        self.model.pipeline.unet.train()
        self.model.pipeline.text_encoder.train()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 메인 프로세스에서만 progress bar 표시
        if self.accelerator.is_main_process:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            progress_bar = self.train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model.pipeline):
                # 훈련 스텝 (accelerate 기반)
                loss_info = self._train_step_accelerate(batch)
                
                # 손실 누적
                epoch_loss += loss_info['loss']
                
                # 메인 프로세스에서만 진행률 업데이트
                if self.accelerator.is_main_process and hasattr(progress_bar, 'set_postfix'):
                    progress_bar.set_postfix({
                        'loss': f"{loss_info['loss']:.4f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                    })
                
                # 로깅
                if self.global_step % self.config['training']['log_every'] == 0:
                    self._log_training_step(loss_info)
                
                # 체크포인트 저장
                if self.global_step % self.config['training']['save_every'] == 0:
                    self._save_checkpoint()
                
                # 검증
                if self.global_step % self.config['training']['eval_every'] == 0:
                    val_loss = self._validate()
                    self._log_validation(val_loss)
                
                self.global_step += 1
        
        # 에포크 평균 손실 (모든 프로세스에서 평균)
        avg_loss = epoch_loss / num_batches
        avg_loss = self.accelerator.gather_for_metrics(torch.tensor(avg_loss)).mean().item()
        
        return {'train_loss': avg_loss}
    
    def _train_step_accelerate(self, batch):
        """Accelerate 기반 훈련 스텝"""
        # 배치 데이터 가져오기
        pixel_values = batch['image']
        input_ids = batch['text']
        
        # 이미지를 latent space로 변환 (VAE 인코딩)
        with torch.no_grad():
            # 픽셀 값을 [-1, 1] 범위로 정규화 (현재 [0, 1] -> [-1, 1])
            pixel_values = pixel_values * 2.0 - 1.0
            latents = self.model.pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.model.pipeline.vae.config.scaling_factor
        
        # 타임스텝 샘플링
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        
        # 노이즈 추가 (latent space에서)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # UNet 예측
        encoder_hidden_states = self.model.pipeline.text_encoder(input_ids)[0]
        noise_pred = self.model.pipeline.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 손실 계산
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # 역전파
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 통계 정보
        return {
            'loss': loss.detach().item(),
            'noise_pred_norm': noise_pred.norm().detach().item(),
            'noise_norm': noise.norm().detach().item()
        }
    
    def _validate(self) -> float:
        """검증"""
        # 개별 구성 요소를 eval 모드로 설정
        self.model.pipeline.unet.eval()
        self.model.pipeline.text_encoder.eval()
        
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 검증 스텝 (훈련과 동일하지만 역전파 없음)
                loss_info = self._validate_step_accelerate(batch)
                val_loss += loss_info['loss']
        
        # 모든 프로세스에서 평균
        avg_val_loss = val_loss / num_batches
        avg_val_loss = self.accelerator.gather_for_metrics(torch.tensor(avg_val_loss)).mean().item()
        
        return avg_val_loss
    
    def _validate_step_accelerate(self, batch):
        """Accelerate 기반 검증 스텝"""
        # 배치 데이터 가져오기
        pixel_values = batch['image']
        input_ids = batch['text']
        
        # 이미지를 latent space로 변환 (VAE 인코딩)
        with torch.no_grad():
            # 픽셀 값을 [-1, 1] 범위로 정규화 (현재 [0, 1] -> [-1, 1])
            pixel_values = pixel_values * 2.0 - 1.0
            latents = self.model.pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.model.pipeline.vae.config.scaling_factor
        
        # 타임스텝 샘플링
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        
        # 노이즈 추가 (latent space에서)
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # UNet 예측
        encoder_hidden_states = self.model.pipeline.text_encoder(input_ids)[0]
        noise_pred = self.model.pipeline.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # 손실 계산
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # 통계 정보
        return {
            'loss': loss.detach().item(),
            'noise_pred_norm': noise_pred.norm().detach().item(),
            'noise_norm': noise.norm().detach().item()
        }
    
    def _log_training_step(self, loss_info: Dict[str, float]):
        """훈련 스텝 로깅 (메인 프로세스에서만)"""
        if self.accelerator.is_main_process:
            log_data = {
                'train/loss': loss_info['loss'],
                'train/noise_pred_norm': loss_info['noise_pred_norm'],
                'train/noise_norm': loss_info['noise_norm'],
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/global_step': self.global_step,
                'train/epoch': self.current_epoch
            }
            
            self.accelerator.log(log_data)
            self.logger.info(f"Step {self.global_step}: loss={loss_info['loss']:.4f}")
    
    def _log_validation(self, val_loss: float):
        """검증 로깅 (메인 프로세스에서만)"""
        if self.accelerator.is_main_process:
            log_data = {
                'val/loss': val_loss,
                'val/global_step': self.global_step,
                'val/epoch': self.current_epoch
            }
            
            self.accelerator.log(log_data)
            self.logger.info(f"Validation loss: {val_loss:.4f}")
            
            # 최고 성능 모델 저장
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(is_best=True)
    
    def _save_checkpoint(self, is_best: bool = False):
        """체크포인트 저장 (메인 프로세스에서만)"""
        if self.accelerator.is_main_process:
            # accelerate를 사용한 체크포인트 저장
            checkpoint_dir = os.path.join(self.config['training']['save_dir'], 'checkpoints')
            self.accelerator.save_state(checkpoint_dir)
            
            # 추가 메타데이터 저장
            metadata = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'config': self.config
            }
            
            metadata_path = os.path.join(checkpoint_dir, 'metadata.pt')
            torch.save(metadata, metadata_path)
            
            # 최고 성능 체크포인트 복사
            if is_best:
                best_checkpoint_dir = os.path.join(self.config['training']['save_dir'], 'best_checkpoint')
                os.makedirs(best_checkpoint_dir, exist_ok=True)
                
                # accelerate 상태를 best로 복사
                import shutil
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                shutil.copytree(checkpoint_dir, best_checkpoint_dir)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        # accelerate를 사용한 체크포인트 로드
        self.accelerator.load_state(checkpoint_path)
        
        # 메타데이터 로드
        metadata_path = os.path.join(checkpoint_path, 'metadata.pt')
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, map_location=self.accelerator.device)
            self.current_epoch = metadata['epoch']
            self.global_step = metadata['global_step']
            self.best_loss = metadata['best_loss']
        
        if self.accelerator.is_main_process:
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self):
        """전체 훈련 과정"""
        if self.accelerator.is_main_process:
            self.logger.info("Starting training...")
            self.model.print_trainable_parameters()
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # 훈련
            train_info = self.train_epoch()
            
            # 검증
            val_loss = self._validate()
            
            # 로깅
            self._log_epoch(train_info, val_loss)
            
            # 스케줄러 업데이트
            self.scheduler.step()
            
            # 에포크 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint()
            
            # 프로세스 동기화
            self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_main_process:
            self.logger.info("Training completed!")
        
        # 최종 체크포인트 저장
        self._save_checkpoint()
    
    def _log_epoch(self, train_info: Dict[str, float], val_loss: float):
        """에포크 로깅 (메인 프로세스에서만)"""
        if self.accelerator.is_main_process:
            log_data = {
                'epoch/train_loss': train_info['train_loss'],
                'epoch/val_loss': val_loss,
                'epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.current_epoch
            }
            
            self.accelerator.log(log_data)
            
            self.logger.info(
                f"Epoch {self.current_epoch}: "
                f"train_loss={train_info['train_loss']:.4f}, "
                f"val_loss={val_loss:.4f}"
            ) 