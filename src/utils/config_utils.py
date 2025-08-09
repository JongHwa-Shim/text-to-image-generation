"""
Configuration utilities
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    설정 파일 저장
    
    Args:
        config: 설정 딕셔너리
        config_path: 저장 경로
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 딕셔너리 병합
    
    Args:
        base_config: 기본 설정
        override_config: 오버라이드 설정
        
    Returns:
        병합된 설정
    """
    merged_config = base_config.copy()
    
    def _merge_dicts(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_dicts(base[key], value)
            else:
                base[key] = value
    
    _merge_dicts(merged_config, override_config)
    return merged_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    설정 유효성 검증
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        유효성 여부
    """
    required_keys = [
        'training.model_name',
        'training.batch_size',
        'training.learning_rate',
        'training.num_epochs'
    ]
    
    for key in required_keys:
        keys = key.split('.')
        current = config
        for k in keys:
            if k not in current:
                print(f"Missing required config key: {key}")
                return False
            current = current[k]
    
    return True 