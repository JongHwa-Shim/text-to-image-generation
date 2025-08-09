"""
Dataset classes for floorplan generation
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import CLIPTokenizer
from .augmentation import TextAugmenter


class FloorPlanDataset(Dataset):
    """평면도 생성용 데이터셋 클래스"""
    
    def __init__(
        self,
        text_dir: str,
        image_dir: str,
        tokenizer: CLIPTokenizer,
        image_size: int = 256,
        augment_config: Optional[Dict] = None,
        split: str = "train"
    ):
        """
        Args:
            text_dir: 텍스트 파일 디렉토리
            image_dir: 이미지 파일 디렉토리
            tokenizer: CLIP 토크나이저
            image_size: 이미지 크기
            augment_config: 증강 설정
            split: 데이터 분할 ("train", "val")
        """
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.split = split
        
        # 증강기 초기화
        if augment_config and split == "train":
            self.augmenter = TextAugmenter(augment_config)
        else:
            self.augmenter = None
        
        # 데이터 파일 목록 생성
        self.data_files = self._get_data_files()
        
        # 텍스트 최대 길이 설정
        self.max_text_length = 77  # CLIP 토크나이저 기본값
    
    def _get_data_files(self) -> List[str]:
        """데이터 파일 목록을 가져옴"""
        text_files = []
        image_files = []
        
        # 텍스트 파일 목록
        for filename in os.listdir(self.text_dir):
            if filename.endswith('.txt'):
                text_files.append(filename)
        
        # 이미지 파일 목록
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                image_files.append(filename)
        
        # 텍스트와 이미지 파일이 매칭되는 것만 선택
        data_files = []
        for text_file in text_files:
            base_name = text_file.replace('.txt', '')
            image_file = f"{base_name}.png"
            if image_file in image_files:
                data_files.append(base_name)
        
        return sorted(data_files)
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        데이터셋에서 하나의 샘플을 가져옴
        
        Args:
            idx: 샘플 인덱스
            
        Returns:
            토크나이즈된 텍스트와 이미지 텐서
        """
        base_name = self.data_files[idx]
        
        # 텍스트 파일 읽기
        text_path = os.path.join(self.text_dir, f"{base_name}.txt")
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # 이미지 파일 읽기
        image_path = os.path.join(self.image_dir, f"{base_name}.png")
        image = Image.open(image_path).convert('RGB')
        
        # 데이터 증강 (훈련 시에만)
        if self.augmenter and self.split == "train":
            text = self.augmenter.augment_text(text)
        
        # 텍스트 토크나이징
        tokenized_text = self._tokenize_text(text)
        
        # 이미지 전처리
        processed_image = self._preprocess_image(image)
        
        return {
            'text': tokenized_text,
            'image': processed_image,
            'text_raw': text,
            'image_path': image_path
        }
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """텍스트를 토크나이징"""
        # CLIP 토크나이저 사용
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return tokens.input_ids.squeeze(0)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """이미지 전처리"""
        # 리사이즈
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # PIL 이미지를 텐서로 변환
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # 정규화 (0-255 -> 0-1)
        image_tensor = image_tensor / 255.0
        
        # 채널 순서 변경 (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.permute(2, 0, 1)
        
        return image_tensor
    
    def get_sample_text(self, idx: int) -> str:
        """특정 인덱스의 원본 텍스트를 가져옴"""
        base_name = self.data_files[idx]
        text_path = os.path.join(self.text_dir, f"{base_name}.txt")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def get_sample_image_path(self, idx: int) -> str:
        """특정 인덱스의 이미지 경로를 가져옴"""
        base_name = self.data_files[idx]
        return os.path.join(self.image_dir, f"{base_name}.png")


class FloorPlanDataLoader:
    """데이터 로더 래퍼 클래스"""
    
    def __init__(
        self,
        text_dir: str,
        image_dir: str,
        tokenizer: CLIPTokenizer,
        batch_size: int = 4,
        image_size: int = 256,
        augment_config: Optional[Dict] = None,
        train_split: float = 0.8,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        """
        Args:
            text_dir: 텍스트 파일 디렉토리
            image_dir: 이미지 파일 디렉토리
            tokenizer: CLIP 토크나이저
            batch_size: 배치 크기
            image_size: 이미지 크기
            augment_config: 증강 설정
            train_split: 훈련 데이터 비율
            num_workers: 워커 수
            shuffle: 셔플 여부
        """
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment_config = augment_config
        self.train_split = train_split
        self.num_workers = num_workers
        self.shuffle = shuffle
        
        # 전체 데이터셋 생성
        self.full_dataset = FloorPlanDataset(
            text_dir=text_dir,
            image_dir=image_dir,
            tokenizer=tokenizer,
            image_size=image_size,
            augment_config=augment_config,
            split="train"
        )
        
        # 훈련/검증 분할
        total_size = len(self.full_dataset)
        train_size = int(total_size * train_split)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_dataset, [train_size, val_size]
        )
        
        # 훈련 데이터셋에 증강 설정 적용
        self.train_dataset.dataset.augmenter = TextAugmenter(augment_config) if augment_config else None
        self.val_dataset.dataset.augmenter = None  # 검증 시에는 증강하지 않음
    
    def get_train_loader(self):
        """훈련 데이터 로더 반환"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self):
        """검증 데이터 로더 반환"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 