"""
Text augmentation utilities for floorplan generation
"""

import random
from typing import Dict, List, Tuple, Optional
from .preprocessing import TextPreprocessor


class TextAugmenter:
    """텍스트 조건 데이터 증강 클래스"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 증강 설정 딕셔너리
        """
        self.config = config
        self.preprocessor = TextPreprocessor()
        
        # 증강 설정
        self.text_masking_prob = config.get('text_masking_prob', 0.8)  # 기본값 80%로 상향
        self.connection_swap_prob = config.get('connection_swap_prob', 0.5)
        self.position_swap_prob = config.get('position_swap_prob', 0.5)
        self.min_rooms_to_keep = config.get('min_rooms_to_keep', 0)  # 0개부터 가능
        self.max_rooms_to_keep = config.get('max_rooms_to_keep', None)  # 상한 없음
    
    def augment_text(self, text: str) -> str:
        """
        텍스트 조건을 증강하여 새로운 텍스트 생성
        
        Args:
            text: 원본 조건 텍스트
            
        Returns:
            증강된 조건 텍스트
        """
        # 텍스트 파싱
        parsed_data = self.preprocessor.parse_text(text)
        
        # 항상 증강 적용 (이중 확률 구조 제거)
        parsed_data['rooms'] = self._augment_rooms(parsed_data['rooms'])
        parsed_data['connections'] = self._augment_connections(parsed_data['connections'])
        parsed_data['positions'] = self._augment_positions(parsed_data['positions'])
        
        # 텍스트 재구성
        return self.preprocessor.reconstruct_text(parsed_data)
    
    def _augment_rooms(self, rooms: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """방 정보 증강 - 2~3개 정도만 남기도록 강력한 마스킹"""
        if not rooms:
            return rooms
        
        # 강력한 마스킹: 2~3개가 주로 남도록 가중치 적용
        total_rooms = len(rooms)
        
        if total_rooms == 0:
            return rooms
        
        # 더 자연스러운 확률 분포: 0개(5%), 1개(15%), 2개(20%), 3개(20%), 4개(15%), 5개(10%), 원본(15%)
        rand = random.random()
        if rand < 0.05:
            num_rooms_to_keep = 0
        elif rand < 0.20:
            num_rooms_to_keep = min(1, total_rooms)
        elif rand < 0.40:
            num_rooms_to_keep = min(2, total_rooms)
        elif rand < 0.60:
            num_rooms_to_keep = min(3, total_rooms)
        elif rand < 0.75:
            num_rooms_to_keep = min(4, total_rooms)
        elif rand < 0.85:
            num_rooms_to_keep = min(5, total_rooms)
        else:
            # 나머지 15%는 원본 유지 또는 많은 개수
            num_rooms_to_keep = random.randint(min(6, total_rooms), total_rooms)
        
        if num_rooms_to_keep == 0:
            return []
        
        # 0개가 아닌 방들을 우선적으로 선택
        non_zero_rooms = [(room_type, count) for room_type, count in rooms if count > 0]
        zero_rooms = [(room_type, count) for room_type, count in rooms if count == 0]
        
        selected_rooms = []
        
        # 0개가 아닌 방들을 먼저 선택
        if non_zero_rooms and num_rooms_to_keep > 0:
            num_non_zero = min(num_rooms_to_keep, len(non_zero_rooms))
            selected_rooms.extend(random.sample(non_zero_rooms, num_non_zero))
        
        # 나머지 자리를 0개 방으로 채움
        remaining_slots = num_rooms_to_keep - len(selected_rooms)
        if remaining_slots > 0 and zero_rooms:
            num_zero = min(remaining_slots, len(zero_rooms))
            selected_rooms.extend(random.sample(zero_rooms, num_zero))
        
        return selected_rooms
    
    def _augment_connections(self, connections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """연결관계 증강 - 1~3개 정도만 남기도록 강력한 마스킹"""
        if not connections:
            return connections
        
        total_connections = len(connections)
        
        if total_connections == 0:
            return connections
        
        # 더 자연스러운 확률 분포: 0개(10%), 1개(20%), 2개(25%), 3개(20%), 4개(10%), 원본(15%)
        rand = random.random()
        if rand < 0.10:
            num_connections_to_keep = 0
        elif rand < 0.30:
            num_connections_to_keep = min(1, total_connections)
        elif rand < 0.55:
            num_connections_to_keep = min(2, total_connections)
        elif rand < 0.75:
            num_connections_to_keep = min(3, total_connections)
        elif rand < 0.85:
            num_connections_to_keep = min(4, total_connections)
        else:
            # 나머지 15%는 원본 유지 또는 많은 개수
            num_connections_to_keep = random.randint(min(5, total_connections), total_connections)
        
        if num_connections_to_keep == 0:
            return []
        
        # 무작위로 연결관계 선택
        selected_connections = random.sample(connections, num_connections_to_keep)
        
        # 선택된 연결관계의 순서를 50% 확률로 바꿈
        augmented_connections = []
        for connection in selected_connections:
            if random.random() < self.connection_swap_prob:
                connection = self.preprocessor.swap_connection_order(connection)
            augmented_connections.append(connection)
        
        return augmented_connections
    
    def _augment_positions(self, positions: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """위치관계 증강 - 1~3개 정도만 남기도록 강력한 마스킹"""
        if not positions:
            return positions
        
        total_positions = len(positions)
        
        if total_positions == 0:
            return positions
        
        # 더 자연스러운 확률 분포: 0개(10%), 1개(20%), 2개(25%), 3개(20%), 4개(10%), 원본(15%)
        rand = random.random()
        if rand < 0.10:
            num_positions_to_keep = 0
        elif rand < 0.30:
            num_positions_to_keep = min(1, total_positions)
        elif rand < 0.55:
            num_positions_to_keep = min(2, total_positions)
        elif rand < 0.75:
            num_positions_to_keep = min(3, total_positions)
        elif rand < 0.85:
            num_positions_to_keep = min(4, total_positions)
        else:
            # 나머지 15%는 원본 유지 또는 많은 개수
            num_positions_to_keep = random.randint(min(5, total_positions), total_positions)
        
        if num_positions_to_keep == 0:
            return []
        
        # 무작위로 위치관계 선택
        selected_positions = random.sample(positions, num_positions_to_keep)
        
        # 선택된 위치관계의 순서를 50% 확률로 바꿈
        augmented_positions = []
        for position in selected_positions:
            if random.random() < self.position_swap_prob:
                position = self.preprocessor.swap_position_order(position)
            augmented_positions.append(position)
        
        return augmented_positions
    
    def generate_multiple_augmentations(self, text: str, num_augmentations: int = 5) -> List[str]:
        """
        하나의 텍스트로부터 여러 개의 증강된 텍스트 생성
        
        Args:
            text: 원본 조건 텍스트
            num_augmentations: 생성할 증강 텍스트 개수
            
        Returns:
            증강된 텍스트 리스트
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented_text = self.augment_text(text)
            augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def validate_augmented_text(self, original_text: str, augmented_text: str) -> bool:
        """
        증강된 텍스트가 유효한지 검증
        
        Args:
            original_text: 원본 텍스트
            augmented_text: 증강된 텍스트
            
        Returns:
            유효성 여부
        """
        try:
            # 파싱 가능한지 확인
            parsed_original = self.preprocessor.parse_text(original_text)
            parsed_augmented = self.preprocessor.parse_text(augmented_text)
            
            # 최소한의 구조는 유지되어야 함
            if not parsed_augmented['rooms']:
                return False
            
            # 재구성 가능한지 확인
            reconstructed = self.preprocessor.reconstruct_text(parsed_augmented)
            
            return len(reconstructed.strip()) > 0
            
        except Exception:
            return False 