#!/usr/bin/env python3
"""
Training script for floorplan generation model with Accelerate support
Supports single GPU, multi-GPU, and multi-node training automatically
"""

import argparse
import os
import sys
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import FloorPlanTrainer
from src.utils.config_utils import load_config, validate_config


def main():
    parser = argparse.ArgumentParser(description="Train floorplan generation model with Accelerate")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    try:
        config = load_config(args.config)
        if not validate_config(config):
            print("ERROR: Invalid configuration")
            sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # 훈련 시작
    try:
        trainer = FloorPlanTrainer(args.config)
        
        # 체크포인트 로드 (있는 경우)
        if args.checkpoint and os.path.exists(args.checkpoint):
            if trainer.accelerator.is_main_process:
                print(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # 훈련 실행
        trainer.train()
        
        if trainer.accelerator.is_main_process:
            print("Training completed successfully!")
        
    except Exception as e:
        if 'trainer' in locals() and trainer.accelerator.is_main_process:
            print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 