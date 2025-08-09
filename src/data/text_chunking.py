#!/usr/bin/env python3
"""
CLIP 토큰 제한을 해결하기 위한 텍스트 청킹 유틸리티
"""

import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer
from typing import List, Tuple
import re


class CLIPTextChunker:
    """CLIP 토큰 제한을 해결하기 위한 텍스트 청킹 클래스"""
    
    def __init__(self, tokenizer: CLIPTokenizer, max_chunk_length: int = 75):
        """
        Args:
            tokenizer: CLIP 토크나이저
            max_chunk_length: 각 청크의 최대 토큰 길이 (77에서 특수토큰 제외)
        """
        self.tokenizer = tokenizer
        self.max_chunk_length = max_chunk_length
    
    def split_text_semantically(self, text: str) -> List[str]:
        """
        텍스트를 의미론적으로 분할
        
        Args:
            text: 입력 텍스트
            
        Returns:
            분할된 텍스트 청크들
        """
        # 섹션별로 분할
        sections = []
        
        # [Number and Type of Rooms] 섹션 추출
        rooms_match = re.search(r'\[Number and Type of Rooms\](.*?)(?=\[|$)', text, re.DOTALL)
        if rooms_match:
            rooms_section = rooms_match.group(1).strip()
            if rooms_section:
                sections.append(f"[Number and Type of Rooms] {rooms_section}")
        
        # [Connection Between Rooms] 섹션 추출
        connections_match = re.search(r'\[Connection Between Rooms\](.*?)(?=\[|$)', text, re.DOTALL)
        if connections_match:
            connections_section = connections_match.group(1).strip()
            if connections_section:
                # 연결 관계를 문장별로 분할
                connection_sentences = [s.strip() for s in connections_section.split('.') if s.strip()]
                if connection_sentences:
                    sections.append(f"[Connection Between Rooms] {'. '.join(connection_sentences[:3])}.")
                    if len(connection_sentences) > 3:
                        sections.append(f"[Connection Between Rooms] {'. '.join(connection_sentences[3:])}.")
        
        # [Positional Relationship Between Rooms] 섹션 추출
        positions_match = re.search(r'\[Positional Relationship Between Rooms\](.*?)(?=\[|$)', text, re.DOTALL)
        if positions_match:
            positions_section = positions_match.group(1).strip()
            if positions_section:
                # 위치 관계를 문장별로 분할
                position_sentences = [s.strip() for s in positions_section.split('.') if s.strip()]
                if position_sentences:
                    sections.append(f"[Positional Relationship Between Rooms] {'. '.join(position_sentences[:3])}.")
                    if len(position_sentences) > 3:
                        sections.append(f"[Positional Relationship Between Rooms] {'. '.join(position_sentences[3:])}.")
        
        # 각 섹션이 토큰 제한을 넘지 않는지 확인하고 필요시 추가 분할
        final_chunks = []
        for section in sections:
            chunks = self._split_section_by_tokens(section)
            final_chunks.extend(chunks)
        
        return final_chunks if final_chunks else [text]  # 빈 결과 방지
    
    def _split_section_by_tokens(self, section: str) -> List[str]:
        """
        섹션을 토큰 길이 기준으로 분할
        
        Args:
            section: 분할할 섹션
            
        Returns:
            분할된 청크들
        """
        # 토큰 수 확인
        tokens = self.tokenizer.encode(section, add_special_tokens=False)
        
        if len(tokens) <= self.max_chunk_length:
            return [section]
        
        # 문장별로 분할하여 재조합
        sentences = section.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            test_chunk = current_chunk + sentence + "."
            test_tokens = self.tokenizer.encode(test_chunk, add_special_tokens=False)
            
            if len(test_tokens) <= self.max_chunk_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def tokenize_chunked_text(self, text: str) -> Tuple[torch.Tensor, List[int]]:
        """
        청킹된 텍스트를 토크나이징
        
        Args:
            text: 입력 텍스트
            
        Returns:
            (토큰 텐서, 각 청크의 길이)
        """
        chunks = self.split_text_semantically(text)
        
        all_tokens = []
        chunk_lengths = []
        
        for chunk in chunks:
            tokens = self.tokenizer(
                chunk,
                padding="max_length",
                max_length=77,  # CLIP 기본 길이
                truncation=True,
                return_tensors="pt"
            )
            all_tokens.append(tokens.input_ids.squeeze(0))
            chunk_lengths.append(tokens.input_ids.shape[1])
        
        # 모든 청크를 하나의 텐서로 결합
        if all_tokens:
            combined_tokens = torch.stack(all_tokens, dim=0)
        else:
            # 빈 입력 처리
            empty_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            combined_tokens = empty_tokens.input_ids
            chunk_lengths = [77]
        
        return combined_tokens, chunk_lengths


class CLIPTextEmbeddingAggregator:
    """청킹된 텍스트 임베딩을 결합하는 클래스"""
    
    def __init__(self, aggregation_method: str = "attention"):
        """
        Args:
            aggregation_method: 집계 방법 ("mean", "max", "weighted", "attention")
        """
        self.aggregation_method = aggregation_method
    
    def aggregate_embeddings(self, embeddings: torch.Tensor, chunk_lengths: List[int]) -> torch.Tensor:
        """
        여러 청크의 임베딩을 하나로 집계
        
        Args:
            embeddings: (num_chunks, seq_len, hidden_dim) 형태의 임베딩
            chunk_lengths: 각 청크의 실제 길이
            
        Returns:
            집계된 임베딩 (seq_len, hidden_dim)
        """
        if embeddings.size(0) == 1:
            return embeddings.squeeze(0)
        
        if self.aggregation_method == "mean":
            return self._mean_aggregation(embeddings, chunk_lengths)
        elif self.aggregation_method == "max":
            return self._max_aggregation(embeddings)
        elif self.aggregation_method == "weighted":
            return self._weighted_aggregation(embeddings, chunk_lengths)
        elif self.aggregation_method == "attention":
            return self._attention_aggregation(embeddings)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _mean_aggregation(self, embeddings: torch.Tensor, chunk_lengths: List[int]) -> torch.Tensor:
        """평균 집계"""
        return embeddings.mean(dim=0)
    
    def _max_aggregation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """최대값 집계"""
        return embeddings.max(dim=0)[0]
    
    def _weighted_aggregation(self, embeddings: torch.Tensor, chunk_lengths: List[int]) -> torch.Tensor:
        """가중 평균 집계 (길이에 비례)"""
        weights = torch.tensor(chunk_lengths, dtype=torch.float32, device=embeddings.device)
        weights = weights / weights.sum()
        weights = weights.view(-1, 1, 1)  # (num_chunks, 1, 1)
        
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=0)
    
    def _attention_aggregation(self, embeddings: torch.Tensor) -> torch.Tensor:
        """어텐션 기반 집계"""
        # 간단한 self-attention 메커니즘
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Query, Key, Value 계산 (단순화된 버전)
        q = embeddings.mean(dim=1, keepdim=True)  # (num_chunks, 1, hidden_dim)
        k = embeddings.mean(dim=1)  # (num_chunks, hidden_dim)
        
        # 어텐션 가중치 계산
        attention_scores = torch.matmul(q.squeeze(1), k.transpose(0, 1))  # (num_chunks, num_chunks)
        attention_weights = F.softmax(attention_scores.mean(dim=1), dim=0)  # (num_chunks,)
        
        # 가중 평균
        attention_weights = attention_weights.view(-1, 1, 1)
        weighted_embeddings = embeddings * attention_weights
        
        return weighted_embeddings.sum(dim=0)
