#!/usr/bin/env python3
"""
텍스트 청킹 원리를 시각화하는 예제
"""

def visualize_chunking_process():
    """텍스트 청킹 과정을 단계별로 시각화"""
    
    print("🎯 텍스트 청킹 과정 시각화")
    print("=" * 60)
    
    # 원본 텍스트
    original_text = """SJH-Style FloorPlan Generation

[Number and Type of Rooms]
The floorplan have 1 living room, 1 master room, 1 kitchen, 1 bathroom

[Connection Between Rooms]
living room #1 and master room #1 are connected.
living room #1 and kitchen #1 are connected.
living room #1 and bathroom #1 are connected.

[Positional Relationship Between Rooms]
master room #1 is left-below living room #1.
kitchen #1 is above living room #1.
bathroom #1 is left-of living room #1."""
    
    print("📝 원본 텍스트:")
    print(f"길이: {len(original_text.split())} 단어")
    print(original_text)
    print()
    
    # 청킹 결과 시뮬레이션
    chunks = [
        "[Number and Type of Rooms] The floorplan have 1 living room, 1 master room, 1 kitchen, 1 bathroom",
        "[Connection Between Rooms] living room #1 and master room #1 are connected. living room #1 and kitchen #1 are connected.",
        "[Connection Between Rooms] living room #1 and bathroom #1 are connected.",
        "[Positional Relationship Between Rooms] master room #1 is left-below living room #1. kitchen #1 is above living room #1.",
        "[Positional Relationship Between Rooms] bathroom #1 is left-of living room #1."
    ]
    
    print("🧩 청킹 결과:")
    for i, chunk in enumerate(chunks, 1):
        print(f"청크 {i}: {chunk}")
        print(f"    └─ 길이: {len(chunk.split())} 단어")
    print()
    
    return chunks

def simulate_embedding_process(chunks):
    """임베딩 과정 시뮬레이션"""
    print("🔢 각 청크를 벡터로 변환:")
    print("=" * 60)
    
    import numpy as np
    
    # 각 청크를 임베딩 벡터로 변환했다고 가정 (실제로는 768차원)
    # 여기서는 시각화를 위해 3차원으로 단순화
    embeddings = []
    
    for i, chunk in enumerate(chunks, 1):
        # 의미를 반영한 가상의 임베딩 벡터
        if "Number and Type" in chunk:
            # 방 정보 - 구조적 정보 강함
            embedding = np.array([0.8, 0.2, 0.1])
        elif "Connection Between" in chunk:
            # 연결 정보 - 관계 정보 강함
            embedding = np.array([0.3, 0.9, 0.2])
        elif "Positional Relationship" in chunk:
            # 위치 정보 - 공간 정보 강함
            embedding = np.array([0.2, 0.3, 0.8])
        else:
            embedding = np.array([0.5, 0.5, 0.5])
        
        embeddings.append(embedding)
        print(f"청크 {i} → 벡터: [{embedding[0]:.1f}, {embedding[1]:.1f}, {embedding[2]:.1f}]")
        print(f"    └─ 특성: {'구조' if embedding[0] > 0.5 else '관계' if embedding[1] > 0.5 else '공간'}")
    
    print()
    return np.array(embeddings)

def simple_attention_explanation():
    """어텐션 메커니즘을 간단하게 설명"""
    print("🧠 어텐션이란? (쉬운 예시)")
    print("=" * 60)
    
    print("📚 상황: 여러 책에서 정보를 종합해야 함")
    print()
    print("책 1: '방의 종류와 개수' (매우 중요)")
    print("책 2: '방 간 연결관계' (중요)")  
    print("책 3: '방의 위치관계' (중요)")
    print("책 4: '기타 정보' (덜 중요)")
    print()
    
    print("🤔 질문: '이 평면도의 핵심 특징은?'")
    print()
    print("🎯 어텐션의 역할:")
    print("1. 각 책의 중요도를 자동으로 판단")
    print("2. 중요한 책에 더 많은 '주의'(attention)를 기울임")
    print("3. 가중평균으로 최종 답변 생성")
    print()
    
    # 어텐션 가중치 시뮬레이션
    attention_weights = [0.4, 0.3, 0.25, 0.05]  # 합계 = 1.0
    books = ["방 정보", "연결관계", "위치관계", "기타"]
    
    print("📊 어텐션 가중치:")
    for book, weight in zip(books, attention_weights):
        bar = "█" * int(weight * 20)  # 시각화
        print(f"{book:8s}: {weight:4.1%} {bar}")
    print()

def detailed_attention_mechanism():
    """어텐션 메커니즘의 상세한 작동 원리"""
    print("🔬 어텐션 메커니즘 상세 원리")
    print("=" * 60)
    
    import numpy as np
    
    # 가상의 청크 임베딩 (3개 청크, 각각 4차원)
    embeddings = np.array([
        [0.8, 0.2, 0.1, 0.3],  # 청크 1: 방 정보
        [0.3, 0.9, 0.2, 0.4],  # 청크 2: 연결관계
        [0.2, 0.3, 0.8, 0.5],  # 청크 3: 위치관계
    ])
    
    print("📊 입력 청크 임베딩:")
    print("청크 1 (방 정보):", embeddings[0])
    print("청크 2 (연결관계):", embeddings[1]) 
    print("청크 3 (위치관계):", embeddings[2])
    print()
    
    print("🔍 1단계: Query 생성 (질문 만들기)")
    # 각 청크의 평균을 Query로 사용
    query = embeddings.mean(axis=1, keepdims=True)  # 각 청크별 평균
    print("각 청크의 질문(Query):")
    for i, q in enumerate(query):
        print(f"청크 {i+1} Query: {q[0]}")
    print()
    
    print("🔍 2단계: Key 생성 (답변 준비)")
    # 각 청크의 평균을 Key로 사용
    key = embeddings.mean(axis=1)  # 각 청크의 특성
    print("각 청크의 특성(Key):")
    for i, k in enumerate(key):
        print(f"청크 {i+1} Key: {k:.3f}")
    print()
    
    print("🔍 3단계: 어텐션 스코어 계산")
    print("(각 Query와 모든 Key 간의 유사도)")
    
    # 어텐션 스코어 = Query @ Key^T
    attention_scores = np.dot(query.squeeze(), key.T)
    print("어텐션 스코어 행렬:")
    print(attention_scores)
    print()
    
    print("🔍 4단계: Softmax로 확률 변환")
    
    # 수동으로 softmax 구현 (더 안전함)
    def manual_softmax(x):
        exp_x = np.exp(x - np.max(x))  # 수치 안정성을 위해 최대값 빼기
        return exp_x / np.sum(exp_x)
    
    # 각 행에 대해 softmax 적용
    attention_weights = np.zeros_like(attention_scores)
    for i in range(attention_scores.shape[0]):
        attention_weights[i] = manual_softmax(attention_scores[i])
    
    print("어텐션 가중치 (각 행의 합 = 1.0):")
    for i, weights in enumerate(attention_weights):
        print(f"청크 {i+1}이 다른 청크들에 주는 관심:")
        for j, weight in enumerate(weights):
            print(f"  → 청크 {j+1}: {weight:.3f}")
    print()
    
    print("🔍 5단계: 가중평균으로 최종 결과")
    # 각 청크에 대해 가중평균 계산
    final_embeddings = np.zeros_like(embeddings)
    
    for i in range(len(embeddings)):
        weighted_sum = np.zeros(embeddings.shape[1])
        for j in range(len(embeddings)):
            weighted_sum += attention_weights[i, j] * embeddings[j]
        final_embeddings[i] = weighted_sum
        
        print(f"청크 {i+1} 최종 임베딩: {final_embeddings[i]}")
    
    return final_embeddings

def practical_example():
    """실제 사용 예시"""
    print("\n🏠 실제 평면도 생성 예시")
    print("=" * 60)
    
    print("📝 입력: '거실, 침실, 주방이 있고 거실과 침실이 연결된 평면도'")
    print()
    
    print("🧩 청킹 과정:")
    print("청크 1: '거실, 침실, 주방이 있고'")
    print("청크 2: '거실과 침실이 연결된'")
    print("청크 3: '평면도'")
    print()
    
    print("🧠 어텐션 처리:")
    print("1. 청크 1 (방 정보): 가중치 0.5 (매우 중요)")
    print("2. 청크 2 (연결 정보): 가중치 0.4 (중요)")  
    print("3. 청크 3 (일반 정보): 가중치 0.1 (덜 중요)")
    print()
    
    print("🎨 결과:")
    print("→ 거실, 침실, 주방을 중심으로 한 평면도")
    print("→ 거실-침실 연결을 강조")
    print("→ 전체적인 평면도 형태 유지")

if __name__ == "__main__":
    chunks = visualize_chunking_process()
    embeddings = simulate_embedding_process(chunks)
    simple_attention_explanation()
    detailed_attention_mechanism()
    practical_example()
