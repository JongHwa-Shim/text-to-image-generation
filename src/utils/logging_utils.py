"""
Logging utilities
"""

import logging
import os
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        log_dir: 로그 디렉토리
        log_level: 로그 레벨
        log_format: 로그 포맷
        
    Returns:
        로거 객체
    """
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"), encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    특정 이름의 로거 가져오기
    
    Args:
        name: 로거 이름
        
    Returns:
        로거 객체
    """
    return logging.getLogger(name)


def log_training_info(logger: logging.Logger, config: dict):
    """
    훈련 정보 로깅
    
    Args:
        logger: 로거 객체
        config: 설정 딕셔너리
    """
    logger.info("=== Training Configuration ===")
    logger.info(f"Model: {config['training']['model_name']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"LoRA rank: {config['training']['lora_rank']}")
    logger.info(f"LoRA alpha: {config['training']['lora_alpha']}")
    logger.info("=============================") 