"""
Text preprocessing utilities for floorplan generation
"""

import re
from typing import Dict, List, Tuple, Optional
import yaml


class TextPreprocessor:
    """텍스트 조건 데이터 전처리 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.positional_relationships = {
            "left": "right",
            "right": "left", 
            "above": "below",
            "below": "above",
            "left-below": "right-above",
            "right-above": "left-below",
            "left-above": "right-below",
            "right-below": "left-above"
        }
        
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if 'positional_relationships' in config:
                    self.positional_relationships.update(config['positional_relationships'])
    
    def parse_text(self, text: str) -> Dict[str, List]:
        """
        텍스트 조건을 파싱하여 구조화된 데이터로 변환
        
        Args:
            text: 조건 텍스트
            
        Returns:
            파싱된 조건 데이터
        """
        sections = self._split_sections(text)
        
        parsed_data = {
            'rooms': self._parse_rooms(sections.get('Number and Type of Rooms', '')),
            'connections': self._parse_connections(sections.get('Connection Between Rooms', '')),
            'positions': self._parse_positions(sections.get('Positional Relationship Between Rooms', ''))
        }
        
        return parsed_data
    
    def _split_sections(self, text: str) -> Dict[str, str]:
        """텍스트를 섹션별로 분리"""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # 섹션 헤더 확인
            if line.startswith('[') and line.endswith(']'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line[1:-1]  # [Section] -> Section
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # 마지막 섹션 처리
        if current_section:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def _parse_rooms(self, room_text: str) -> List[Tuple[str, int]]:
        """방의 개수와 종류 파싱"""
        rooms = []
        if not room_text:
            return rooms
            
        # "The floorplan have 1 living room, 1 master room, ..." 형식 파싱
        match = re.search(r'The floorplan have (.+)', room_text)
        if match:
            room_list = match.group(1).split(',')
            for room_item in room_list:
                room_item = room_item.strip()
                if room_item:
                    # "1 living room" 또는 "0 storage" 형식 파싱
                    parts = room_item.split()
                    if len(parts) >= 2:
                        count = int(parts[0])
                        room_type = ' '.join(parts[1:])
                        rooms.append((room_type, count))
        
        return rooms
    
    def _parse_connections(self, connection_text: str) -> List[Tuple[str, str]]:
        """방 간의 연결관계 파싱"""
        connections = []
        if not connection_text:
            return connections
            
        for line in connection_text.split('\n'):
            line = line.strip()
            if not line or not line.endswith('are connected.'):
                continue
                
            # "room1 #1 and room2 #1 are connected." 형식 파싱
            match = re.search(r'(.+?) #\d+ and (.+?) #\d+ are connected\.', line)
            if match:
                room1 = match.group(1).strip()
                room2 = match.group(2).strip()
                connections.append((room1, room2))
        
        return connections
    
    def _parse_positions(self, position_text: str) -> List[Tuple[str, str, str]]:
        """방 간의 위치관계 파싱"""
        positions = []
        if not position_text:
            return positions
            
        for line in position_text.split('\n'):
            line = line.strip()
            if not line or not line.endswith('.'):
                continue
                
            # "room1 #1 is position relative to room2 #1." 형식 파싱
            match = re.search(r'(.+?) #\d+ is (.+?) (.+?) #\d+\.', line)
            if match:
                room1 = match.group(1).strip()
                position = match.group(2).strip()
                room2 = match.group(3).strip()
                positions.append((room1, position, room2))
        
        return positions
    
    def reconstruct_text(self, parsed_data: Dict[str, List]) -> str:
        """
        파싱된 데이터를 다시 텍스트로 재구성
        
        Args:
            parsed_data: 파싱된 조건 데이터
            
        Returns:
            재구성된 텍스트
        """
        text_parts = ["SJH-Style FloorPlan Generation"]
        
        # 방의 개수와 종류
        if parsed_data['rooms']:
            text_parts.append("[Number and Type of Rooms]")
            room_text = "The floorplan have " + ", ".join([
                f"{count} {room_type}" for room_type, count in parsed_data['rooms']
            ])
            text_parts.append(room_text)
        
        # 연결관계
        if parsed_data['connections']:
            text_parts.append("[Connection Between Rooms]")
            for room1, room2 in parsed_data['connections']:
                text_parts.append(f"{room1} #1 and {room2} #1 are connected.")
        
        # 위치관계
        if parsed_data['positions']:
            text_parts.append("[Positional Relationship Between Rooms]")
            for room1, position, room2 in parsed_data['positions']:
                text_parts.append(f"{room1} #1 is {position} {room2} #1.")
        
        return '\n'.join(text_parts)
    
    def swap_connection_order(self, connection: Tuple[str, str]) -> Tuple[str, str]:
        """연결관계의 방 순서를 바꿈"""
        return (connection[1], connection[0])
    
    def swap_position_order(self, position: Tuple[str, str, str]) -> Tuple[str, str, str]:
        """위치관계의 방 순서를 바꾸고 위치 표현도 반대로 변경"""
        room1, position_rel, room2 = position
        
        # 위치 표현을 반대로 변경
        opposite_position = self.positional_relationships.get(position_rel, position_rel)
        
        return (room2, opposite_position, room1) 