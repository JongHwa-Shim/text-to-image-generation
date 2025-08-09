"""
Post-processing utilities for generated floorplans
"""

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
import yaml


class FloorPlanPostProcessor:
    """생성된 평면도 후처리 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정 딕셔너리 또는 설정 파일 경로
        """
        # 기본 설정
        self.config = {
            'noise_threshold': 30,
            'min_region_size': 10,
            'smoothing_kernel_size': 3
        }
        
        # 방 색상 매핑
        self.room_colors = {
            'living_room': [255, 255, 220],
            'master_room': [0, 255, 0],
            'bed_room': [30, 140, 50],
            'kitchen': [190, 90, 90],
            'bathroom': [66, 78, 255],
            'balcony': [50, 180, 255],
            'storage': [210, 210, 90],
            'external_area': [255, 255, 255],
            'wall': [95, 95, 95],
            'front_door': [255, 255, 0],
            'interior_door': [160, 100, 0],
            'entrance': [0, 200, 100],
            'window': [125, 190, 190]
        }
        
        # 설정 로드
        if config:
            if isinstance(config, str):
                # 설정 파일 경로인 경우
                with open(config, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                    if 'post_processing' in config_dict:
                        self.config.update(config_dict['post_processing'])
                    if 'room_colors' in config_dict:
                        self.room_colors.update(config_dict['room_colors'])
            elif isinstance(config, dict):
                # 설정 딕셔너리인 경우
                if 'post_processing' in config:
                    self.config.update(config['post_processing'])
                if 'room_colors' in config:
                    self.room_colors.update(config['room_colors'])
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """
        생성된 이미지를 후처리
        
        Args:
            image: 생성된 이미지 numpy 배열 (H, W, C) 또는 (C, H, W)
            
        Returns:
            후처리된 이미지 (H, W, 3)
        """
        # numpy 배열로 변환 (이미 numpy 배열인 경우 그대로 사용)
        if isinstance(image, np.ndarray):
            image_np = image
        else:
            # 텐서인 경우 numpy로 변환
            if hasattr(image, 'cpu'):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)
        
        # 차원 확인 및 조정
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            # (C, H, W) -> (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # 0-1 범위를 0-255로 변환
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        
        # 노이즈 제거
        processed_image = self._remove_noise(image_np)
        
        # 색상 정규화
        processed_image = self._normalize_colors(processed_image)
        
        # 영역 정리
        processed_image = self._clean_regions(processed_image)
        
        # 스무딩
        processed_image = self._smooth_image(processed_image)
        
        return processed_image
    
    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        # 중간값 필터로 노이즈 제거
        kernel_size = self.config['smoothing_kernel_size']
        if kernel_size > 1:
            image = cv2.medianBlur(image, kernel_size)
        
        return image
    
    def _normalize_colors(self, image: np.ndarray) -> np.ndarray:
        """색상을 가장 가까운 방 색상으로 정규화"""
        height, width, channels = image.shape
        normalized_image = np.zeros_like(image)
        
        # 각 픽셀을 가장 가까운 방 색상으로 매핑
        for y in range(height):
            for x in range(width):
                pixel = image[y, x]
                closest_color = self._find_closest_color(pixel)
                normalized_image[y, x] = closest_color
        
        return normalized_image
    
    def _find_closest_color(self, pixel: np.ndarray) -> np.ndarray:
        """가장 가까운 방 색상 찾기"""
        min_distance = float('inf')
        closest_color = None
        
        for room_type, color in self.room_colors.items():
            distance = np.linalg.norm(pixel - np.array(color))
            if distance < min_distance:
                min_distance = distance
                closest_color = np.array(color)
        
        return closest_color
    
    def _clean_regions(self, image: np.ndarray) -> np.ndarray:
        """작은 영역 제거 및 정리"""
        # 각 색상별로 마스크 생성
        cleaned_image = image.copy()
        
        for room_type, color in self.room_colors.items():
            # 해당 색상의 마스크 생성
            mask = np.all(image == color, axis=2)
            
            # 연결 요소 분석
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            
            # 작은 영역 제거
            min_size = self.config['min_region_size']
            for i in range(1, num_labels):  # 0은 배경
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    # 작은 영역을 배경색으로 변경
                    cleaned_image[labels == i] = self.room_colors['external_area']
        
        return cleaned_image
    
    def _smooth_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 스무딩"""
        # 바이래터럴 필터로 엣지 보존하면서 스무딩
        smoothed_image = cv2.bilateralFilter(
            image, 
            d=9, 
            sigmaColor=75, 
            sigmaSpace=75
        )
        
        return smoothed_image
    
    def save_image(self, image: np.ndarray, save_path: str):
        """
        후처리된 이미지 저장
        
        Args:
            image: 후처리된 이미지
            save_path: 저장 경로
        """
        # PIL 이미지로 변환하여 저장
        pil_image = Image.fromarray(image)
        pil_image.save(save_path)
    
    def visualize_regions(self, image: np.ndarray, save_path: Optional[str] = None) -> np.ndarray:
        """
        영역별 시각화
        
        Args:
            image: 평면도 이미지
            save_path: 저장 경로 (선택사항)
            
        Returns:
            시각화된 이미지
        """
        height, width, _ = image.shape
        visualization = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 각 영역별로 다른 색상으로 시각화
        for i, (room_type, color) in enumerate(self.room_colors.items()):
            mask = np.all(image == color, axis=2)
            if mask.any():
                # 각 영역에 고유한 색상 할당
                hue = (i * 30) % 180  # HSV 색상 공간에서 색상 분포
                visualization[mask] = [hue, 255, 255]
        
        # HSV를 BGR로 변환
        visualization = cv2.cvtColor(visualization, cv2.COLOR_HSV2BGR)
        
        if save_path:
            cv2.imwrite(save_path, visualization)
        
        return visualization
    
    def get_room_statistics(self, image: np.ndarray) -> Dict[str, int]:
        """
        방별 픽셀 수 통계
        
        Args:
            image: 평면도 이미지
            
        Returns:
            방별 픽셀 수 딕셔너리
        """
        statistics = {}
        
        for room_type, color in self.room_colors.items():
            mask = np.all(image == color, axis=2)
            pixel_count = np.sum(mask)
            statistics[room_type] = int(pixel_count)
        
        return statistics
    
    def validate_floorplan(self, image: np.ndarray) -> Dict[str, bool]:
        """
        평면도 유효성 검증
        
        Args:
            image: 평면도 이미지
            
        Returns:
            검증 결과 딕셔너리
        """
        validation_results = {}
        
        # 기본 검증 항목들
        height, width, _ = image.shape
        
        # 1. 이미지 크기 검증
        validation_results['valid_size'] = height == 256 and width == 256
        
        # 2. 필수 방 존재 여부 검증
        statistics = self.get_room_statistics(image)
        validation_results['has_living_room'] = statistics.get('living_room', 0) > 0
        
        # 3. 배경 영역 비율 검증
        total_pixels = height * width
        background_ratio = statistics.get('external_area', 0) / total_pixels
        validation_results['reasonable_background'] = background_ratio < 0.8
        
        # 4. 벽 존재 여부 검증
        validation_results['has_walls'] = statistics.get('wall', 0) > 0
        
        return validation_results 