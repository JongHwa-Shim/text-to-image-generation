#!/usr/bin/env python3
"""
평면도 생성기 클래스
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

from src.utils.config_utils import load_config
from src.training import FloorPlanTrainer


class AccelerateFloorPlanGenerator:
    """Accelerate 체크포인트를 사용하는 평면도 생성기"""
    
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "auto"):
        """
        Args:
            checkpoint_path: Accelerate 체크포인트 경로
            config_path: 설정 파일 경로
            device: 디바이스
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        
        # 디바이스 설정
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 설정 로드
        self.config = load_config(config_path)
        
        # 훈련기 초기화 (체크포인트 로드)
        self.trainer = FloorPlanTrainer(config_path)
        self.trainer.load_checkpoint(checkpoint_path)
        
        # 후처리기 초기화
        from src.inference.post_processor import FloorPlanPostProcessor
        self.post_processor = FloorPlanPostProcessor(self.config)
        
        print(f"Generator initialized successfully!")
        print(f"Device: {self.trainer.accelerator.device}")
        print(f"Model components loaded:")
        print(f"  - UNet: {type(self.trainer.model.pipeline.unet)}")
        print(f"  - VAE: {type(self.trainer.model.pipeline.vae)}")
        print(f"  - Text Encoder: {type(self.trainer.model.pipeline.text_encoder)}")
    
    def generate_single(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        단일 평면도 생성
        
        Args:
            prompt: 조건 텍스트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
            seed: 랜덤 시드
            
        Returns:
            (raw_image, processed_image)
        """
        # 시드 설정
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # 모델을 추론 모드로 설정
        self.trainer.model.pipeline.unet.eval()
        self.trainer.model.pipeline.text_encoder.eval()
        
        # 이미지 생성
        with torch.no_grad():
            result = self.trainer.model.pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=256,
                width=256,
                return_dict=True
            )
            
            # 결과에서 이미지 추출
            if isinstance(result, dict):
                image = result['images'][0]
            elif isinstance(result, list):
                image = result[0]
            else:
                image = result
        
        # 이미지를 numpy 배열로 변환
        if hasattr(image, 'squeeze'):  # 텐서인 경우
            raw_image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            raw_image = ((raw_image + 1) * 127.5).astype(np.uint8)
        else:  # PIL.Image인 경우
            raw_image = np.array(image)
        
        # 후처리
        processed_image = self.post_processor.process_image(raw_image)
        
        return raw_image, processed_image
    
    def generate_batch(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        배치 평면도 생성
        
        Args:
            prompts: 조건 텍스트 리스트
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
            seed: 랜덤 시드
            
        Returns:
            [(raw_image, processed_image), ...]
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            # 각 프롬프트마다 시드 설정
            current_seed = seed + i if seed is not None else None
            
            raw_image, processed_image = self.generate_single(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed
            )
            
            results.append((raw_image, processed_image))
        
        return results
    
    def generate_from_file(
        self,
        prompt_file: str,
        output_dir: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ):
        """
        파일에서 텍스트를 읽어서 평면도 생성
        
        Args:
            prompt_file: 텍스트 파일 경로
            output_dir: 출력 디렉토리
            num_inference_steps: 추론 스텝 수
            guidance_scale: 가이던스 스케일
            seed: 랜덤 시드
        """
        # 텍스트 파일 읽기
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"Generating {len(prompts)} floorplans from file: {prompt_file}")
        
        # 배치 생성
        results = self.generate_batch(
            prompts=prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # 결과 저장
        self.save_generated_images(results, output_dir, "batch_generation")
        
        print(f"Generated floorplans saved to: {output_dir}")
    
    def save_generated_images(
        self,
        images: List[Tuple[np.ndarray, np.ndarray]],
        output_dir: str,
        prefix: str = "generation"
    ):
        """
        생성된 이미지 저장
        
        Args:
            images: [(raw_image, processed_image), ...]
            output_dir: 출력 디렉토리
            prefix: 파일명 접두사
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (raw_image, processed_image) in enumerate(images):
            # 원본 이미지 저장
            raw_path = os.path.join(output_dir, f"{prefix}_raw_{i:03d}.png")
            Image.fromarray(raw_image).save(raw_path)
            
            # 후처리된 이미지 저장
            processed_path = os.path.join(output_dir, f"{prefix}_processed_{i:03d}.png")
            Image.fromarray(processed_image).save(processed_path)
        
        print(f"Saved {len(images)} image pairs to {output_dir}")
    
    def interactive_generation(self):
        """대화형 평면도 생성"""
        print("=== Interactive Floorplan Generation ===")
        print("Enter text prompts (type 'quit' to exit):")
        
        while True:
            try:
                prompt = input("\nEnter text prompt: ").strip()
                
                if prompt.lower() == 'quit':
                    break
                
                if not prompt:
                    continue
                
                print("Generating floorplan...")
                raw_image, processed_image = self.generate_single(prompt)
                
                # 이미지 저장
                import time
                timestamp = int(time.time())
                output_dir = "interactive_output"
                os.makedirs(output_dir, exist_ok=True)
                
                raw_path = os.path.join(output_dir, f"interactive_raw_{timestamp}.png")
                processed_path = os.path.join(output_dir, f"interactive_processed_{timestamp}.png")
                
                Image.fromarray(raw_image).save(raw_path)
                Image.fromarray(processed_image).save(processed_path)
                
                print(f"Floorplan saved: {raw_path}, {processed_path}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")