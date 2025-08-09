#!/usr/bin/env python3
"""
평면도 생성 스크립트
"""

import argparse
import logging
import os
import sys
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.generator import AccelerateFloorPlanGenerator
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="훈련된 모델을 사용하여 평면도 생성")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="모델 체크포인트 경로"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="평면도 생성을 위한 텍스트 프롬프트"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="텍스트 프롬프트가 포함된 파일 (한 줄에 하나씩)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="생성된 평면도의 출력 디렉토리"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="추론 스텝 수"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="생성 가이던스 스케일"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="생성을 위한 랜덤 시드"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="사용할 디바이스 (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="대화형 모드로 실행"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로깅 레벨"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(log_level=log_level)
    
    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # 생성기 초기화
    try:
        generator = AccelerateFloorPlanGenerator(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 생성 모드 결정
    if args.interactive:
        # 대화형 모드
        logger.info("대화형 생성 모드 시작")
        generator.interactive_generation()
        
    elif args.text:
        # 단일 텍스트 생성
        logger.info(f"텍스트에 대한 평면도 생성: {args.text[:100]}...")
        
        raw_image, processed_image = generator.generate_single(
            prompt=args.text,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        # 결과 저장
        generator.save_generated_images(
            [(raw_image, processed_image)],
            args.output_dir,
            "single_generation"
        )
        
        logger.info(f"평면도가 {args.output_dir}에 저장되었습니다")
        
    elif args.text_file:
        # 파일에서 텍스트 읽어서 생성
        if not os.path.exists(args.text_file):
            logger.error(f"텍스트 파일을 찾을 수 없습니다: {args.text_file}")
            sys.exit(1)
        
        logger.info(f"텍스트 파일에서 평면도 생성: {args.text_file}")
        
        generator.generate_from_file(
            prompt_file=args.text_file,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        logger.info(f"평면도들이 {args.output_dir}에 저장되었습니다")
        
    else:
        logger.error("--text, --text-file, 또는 --interactive 중 하나를 제공해주세요")
        sys.exit(1)


if __name__ == "__main__":
    main()