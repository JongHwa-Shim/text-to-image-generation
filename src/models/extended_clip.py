#!/usr/bin/env python3
"""
확장된 CLIP Text Encoder - 긴 시퀀스 지원
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextConfig
from typing import Optional, Tuple


class ExtendedCLIPTextEncoder(nn.Module):
    """긴 시퀀스를 지원하는 확장된 CLIP Text Encoder"""
    
    def __init__(
        self,
        base_model_name: str = "runwayml/stable-diffusion-v1-5",
        max_length: int = 150,  # 기본 77에서 150으로 확장
        pooling_method: str = "sliding_window"
    ):
        """
        Args:
            base_model_name: 기반 모델 이름
            max_length: 최대 시퀀스 길이
            pooling_method: 긴 시퀀스 처리 방법 ("sliding_window", "hierarchical", "attention_pooling")
        """
        super().__init__()
        
        self.max_length = max_length
        self.base_max_length = 77  # CLIP 기본 길이
        self.pooling_method = pooling_method
        
        # 기본 CLIP Text Encoder 로드
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model_name, 
            subfolder="text_encoder"
        )
        
        # 위치 임베딩 확장
        self._extend_position_embeddings()
        
        # 추가 레이어들
        if pooling_method == "hierarchical":
            self._setup_hierarchical_layers()
        elif pooling_method == "attention_pooling":
            self._setup_attention_pooling()
    
    def _extend_position_embeddings(self):
        """위치 임베딩을 확장된 길이로 확장"""
        if self.max_length <= self.base_max_length:
            return
        
        old_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        old_num_embeddings = old_embeddings.num_embeddings
        old_embedding_dim = old_embeddings.embedding_dim
        
        # 새로운 위치 임베딩 생성
        new_embeddings = nn.Embedding(self.max_length, old_embedding_dim)
        
        # 기존 가중치 복사
        with torch.no_grad():
            new_embeddings.weight[:old_num_embeddings] = old_embeddings.weight
            
            # 추가 위치에 대해서는 기존 패턴을 반복/보간
            if self.max_length > old_num_embeddings:
                remaining = self.max_length - old_num_embeddings
                # 기존 위치 임베딩의 마지막 부분을 반복
                repeat_start = max(1, old_num_embeddings - remaining)
                repeat_embeddings = old_embeddings.weight[repeat_start:old_num_embeddings]
                
                for i in range(remaining):
                    idx = old_num_embeddings + i
                    source_idx = repeat_start + (i % len(repeat_embeddings))
                    new_embeddings.weight[idx] = old_embeddings.weight[source_idx]
        
        # 새 임베딩으로 교체
        self.text_encoder.text_model.embeddings.position_embedding = new_embeddings
        self.text_encoder.text_model.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))
    
    def _setup_hierarchical_layers(self):
        """계층적 처리를 위한 레이어 설정"""
        hidden_size = self.text_encoder.config.hidden_size
        
        self.chunk_processor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        self.chunk_pooling = nn.AdaptiveAvgPool1d(self.base_max_length)
    
    def _setup_attention_pooling(self):
        """어텐션 풀링을 위한 레이어 설정"""
        hidden_size = self.text_encoder.config.hidden_size
        
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.pooling_query = nn.Parameter(torch.randn(1, self.base_max_length, hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            input_ids: 토큰 ID (batch_size, seq_len)
            attention_mask: 어텐션 마스크
            
        Returns:
            (last_hidden_state, pooler_output)
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len <= self.base_max_length:
            # 기본 길이 이하면 원래 CLIP 사용
            return self.text_encoder(input_ids, attention_mask)
        
        if self.pooling_method == "sliding_window":
            return self._sliding_window_forward(input_ids, attention_mask)
        elif self.pooling_method == "hierarchical":
            return self._hierarchical_forward(input_ids, attention_mask)
        elif self.pooling_method == "attention_pooling":
            return self._attention_pooling_forward(input_ids, attention_mask)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
    
    def _sliding_window_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """슬라이딩 윈도우 방식으로 긴 시퀀스 처리"""
        batch_size, seq_len = input_ids.shape
        window_size = self.base_max_length
        stride = window_size // 2  # 50% 오버랩
        
        all_hidden_states = []
        
        for start in range(0, seq_len, stride):
            end = min(start + window_size, seq_len)
            window_input_ids = input_ids[:, start:end]
            
            # 필요시 패딩
            if window_input_ids.size(1) < window_size:
                pad_length = window_size - window_input_ids.size(1)
                padding = torch.zeros(batch_size, pad_length, dtype=torch.long, device=input_ids.device)
                window_input_ids = torch.cat([window_input_ids, padding], dim=1)
            
            window_mask = None
            if attention_mask is not None:
                window_mask = attention_mask[:, start:end]
                if window_mask.size(1) < window_size:
                    pad_length = window_size - window_mask.size(1)
                    mask_padding = torch.zeros(batch_size, pad_length, dtype=torch.long, device=attention_mask.device)
                    window_mask = torch.cat([window_mask, mask_padding], dim=1)
            
            # CLIP 인코딩
            outputs = self.text_encoder(window_input_ids, window_mask)
            all_hidden_states.append(outputs.last_hidden_state)
        
        # 윈도우 결과들을 결합
        if len(all_hidden_states) == 1:
            combined_hidden_states = all_hidden_states[0]
        else:
            # 평균 풀링
            combined_hidden_states = torch.stack(all_hidden_states, dim=0).mean(dim=0)
        
        # Pooler output 계산
        pooler_output = combined_hidden_states[:, 0, :]  # [CLS] 토큰
        
        return combined_hidden_states, pooler_output
    
    def _hierarchical_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """계층적 방식으로 긴 시퀀스 처리"""
        batch_size, seq_len = input_ids.shape
        
        # 청크로 분할
        chunk_size = self.base_max_length
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        chunk_embeddings = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_input_ids = input_ids[:, start:end]
            
            # 패딩
            if chunk_input_ids.size(1) < chunk_size:
                pad_length = chunk_size - chunk_input_ids.size(1)
                padding = torch.zeros(batch_size, pad_length, dtype=torch.long, device=input_ids.device)
                chunk_input_ids = torch.cat([chunk_input_ids, padding], dim=1)
            
            # CLIP 인코딩
            chunk_outputs = self.text_encoder(chunk_input_ids)
            chunk_embeddings.append(chunk_outputs.pooler_output)  # (batch_size, hidden_size)
        
        # 청크 임베딩들을 결합
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)  # (batch_size, num_chunks, hidden_size)
        
        # 트랜스포머로 청크들 간의 관계 학습
        processed_chunks = self.chunk_processor(chunk_embeddings)
        
        # 원래 길이로 풀링
        if processed_chunks.size(1) > self.base_max_length:
            # 적응적 풀링으로 길이 조정
            processed_chunks = processed_chunks.transpose(1, 2)  # (batch_size, hidden_size, num_chunks)
            pooled = self.chunk_pooling(processed_chunks)  # (batch_size, hidden_size, base_max_length)
            last_hidden_state = pooled.transpose(1, 2)  # (batch_size, base_max_length, hidden_size)
        else:
            last_hidden_state = processed_chunks
            if last_hidden_state.size(1) < self.base_max_length:
                # 패딩
                pad_length = self.base_max_length - last_hidden_state.size(1)
                padding = torch.zeros(
                    batch_size, pad_length, last_hidden_state.size(2),
                    device=last_hidden_state.device
                )
                last_hidden_state = torch.cat([last_hidden_state, padding], dim=1)
        
        pooler_output = last_hidden_state[:, 0, :]
        
        return last_hidden_state, pooler_output
    
    def _attention_pooling_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """어텐션 풀링 방식으로 긴 시퀀스 처리"""
        # 먼저 전체 시퀀스를 인코딩 (확장된 위치 임베딩 사용)
        outputs = self.text_encoder(input_ids, attention_mask)
        long_hidden_states = outputs.last_hidden_state  # (batch_size, long_seq_len, hidden_size)
        
        batch_size = long_hidden_states.size(0)
        
        # 학습 가능한 쿼리로 어텐션 풀링
        query = self.pooling_query.expand(batch_size, -1, -1)  # (batch_size, base_max_length, hidden_size)
        
        pooled_output, attention_weights = self.attention_pooling(
            query, long_hidden_states, long_hidden_states
        )
        
        pooled_output = self.layer_norm(pooled_output)
        pooler_output = pooled_output[:, 0, :]
        
        return pooled_output, pooler_output
