"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple


def visualize_floorplan(
    image: np.ndarray,
    title: str = "Floorplan",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    평면도 시각화
    
    Args:
        image: 평면도 이미지
        title: 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def create_room_legend(room_colors: Dict[str, List[int]], figsize: Tuple[int, int] = (8, 6)):
    """
    방 색상 범례 생성
    
    Args:
        room_colors: 방 색상 딕셔너리
        figsize: 그림 크기
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = 0
    for room_type, color in room_colors.items():
        # 색상 패치 생성
        rect = plt.Rectangle((0, y_pos), 1, 0.8, facecolor=np.array(color) / 255)
        ax.add_patch(rect)
        
        # 텍스트 추가
        ax.text(1.2, y_pos + 0.4, room_type.replace('_', ' ').title(), 
                fontsize=12, verticalalignment='center')
        
        y_pos += 1
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(room_colors))
    ax.set_title("Room Color Legend")
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_floorplans(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    여러 평면도 비교 시각화
    
    Args:
        images: 평면도 이미지 리스트
        titles: 제목 리스트
        save_path: 저장 경로
        figsize: 그림 크기
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """
    훈련 히스토리 플롯
    
    Args:
        train_losses: 훈련 손실 리스트
        val_losses: 검증 손실 리스트
        save_path: 저장 경로
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def create_floorplan_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    n_cols: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    평면도 그리드 시각화
    
    Args:
        images: 평면도 이미지 리스트
        titles: 제목 리스트
        n_cols: 열 수
        save_path: 저장 경로
        figsize: 그림 크기
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, image in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        
        axes[row, col].imshow(image)
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
        axes[row, col].axis('off')
    
    # 빈 서브플롯 숨기기
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def highlight_room(
    image: np.ndarray,
    room_type: str,
    room_colors: Dict[str, List[int]],
    highlight_color: List[int] = [255, 0, 0],
    alpha: float = 0.7
) -> np.ndarray:
    """
    특정 방 하이라이트
    
    Args:
        image: 평면도 이미지
        room_type: 방 타입
        room_colors: 방 색상 딕셔너리
        highlight_color: 하이라이트 색상
        alpha: 투명도
        
    Returns:
        하이라이트된 이미지
    """
    if room_type not in room_colors:
        raise ValueError(f"Unknown room type: {room_type}")
    
    target_color = room_colors[room_type]
    mask = np.all(image == target_color, axis=2)
    
    highlighted_image = image.copy()
    highlighted_image[mask] = np.array(highlight_color)
    
    # 알파 블렌딩
    result = cv2.addWeighted(image, 1 - alpha, highlighted_image, alpha, 0)
    
    return result 