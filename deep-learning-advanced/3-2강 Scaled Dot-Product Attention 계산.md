---
title: 3-2강 Scaled Dot-Product Attention 계산
date: 2026-09-03
updated: 2026-09-03
description: KANT 강의 '3-2강 Scaled Dot-Product Attention 계산' 정리
---

## 1. 네 단계로 보는 Attention 계산

Scaled Dot-Product Attention의 공식

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

Scaled Dot-Product Attention 네 단계

1. **비교**: Query와 Key를 비교해 score를 만든다.
2. **조절**: score가 너무 커지지 않도록 `√dₖ`로 나눈다.
3. **비율**: Softmax로 합이 1인 weight를 만든다.
4. **섞기**: Weight만큼 Value를 섞어 새로운 표현을 만든다.

## 2. Score matrix는 어떤 표인가

Query 하나와 Key 하나를 비교하면 관련도 점수 하나가 나온다. <br>
문장 안에 Query와 Key가 여러 개 있으므로, 모든 조합을 한 번에 비교하면 표가 만들어진다

Score matrix

- **행(row)**: 현재 의미를 업데이트하려는 Query
- **열(column)**: 참고 후보인 Key
- **칸(cell)**: 해당 Query와 Key의 관련도 점수

### 2-1. Score는 확률이 아니다

Score는 음수일 수도 있고, 합이 1일 필요가 없다. 단순한 관련도 점수입니다.

```
예시 score: [-0.5, 1.7, 0.2]
```

이 score를 Softmax에 넣어야 참고 비율로 해석하기 쉬운 weight가 된다

## 3. Scaling은 왜 하나요?

Query와 Key의 차원 `dₖ`가 커지면 dot-product score도 큰 값이 되기 쉽다.<br>
너무 큰 값을 Softmax에 넣으면 한 위치에 거의 모든 weight가 몰릴 수 있다.

Scaling은 score를 `√dₖ`로 나눠 값의 크기를 완화합니다.

```
scaled_scores = scores / √dₖ
```

Scaling의 목적은 순위를 억지로 바꾸는 것이 아니라, Softmax가 지나치게 뾰족해지는 것을 완화해 학습이 안정적으로 이루어지도록 돕는 것이다


## 4. Softmax는 무엇을 바꾸나

Softmax는 한 Query가 모든 Key에 준 score를 다음 조건의 weight로 바꾼다

- 각 값은 0과 1 사이다.
- 한 Query 행의 weight 합은 1이다.
- 상대적으로 큰 score는 더 큰 weight가 된다

```
score:  [0.4, 1.8, 0.7]
weight: [0.16, 0.65, 0.19]  # 예시
```

### 4-1. 왜 `dim=-1`인가요?

Score shape이 `[B, L_query, L_key]`라면 마지막 축은 Key 후보 축이다<br> 따라서 각 Query가 모든 Key에 주는 비율을 만들기 위해 마지막 축에 Softmax를 적용한다

```python
weights = torch.softmax(scores, dim=-1)
```

`weights.sum(dim=-1)`은 각 Query마다 약 1이 된다



## 5. Shape 이야기로 전체 흐름 읽기

Attention Shape 흐름

Self-Attention에서 Q, K, V shape이 모두 `[B, L, D]`라고 가정.

| 단계 | Tensor | Shape | 의미 |
| --- | --- | --- | --- |
| 입력 | Q | `[B, L, D]` | 각 토큰의 질문 표현 |
| 입력 | K | `[B, L, D]` | 각 토큰의 비교 기준 표현 |
| 입력 | V | `[B, L, D]` | 각 토큰에서 가져올 정보 |
| 비교 | `Q @ Kᵀ` | `[B, L, L]` | 토큰×토큰 관련도 표 |
| 비율 | Softmax | `[B, L, L]` | 토큰×토큰 참고 비율 |
| 혼합 | `weights @ V` | `[B, L, D]` | 문맥이 반영된 토큰 표현 |

가장 중요한 변화

```
토큰 표현 [B,L,D]
-> 토큰 관계표 [B,L,L]
-> 새 토큰 표현 [B,L,D]
```

## 6. 연습 문제: 작은 Tensor로 계산 흐름 확인

```python
import math
import torch

# 결과를 다시 실행해도 동일하게 만들기 위해 seed를 고정합니다.
torch.manual_seed(42)

# 토큰 세 개를 각각 2차원 벡터로 표현한 아주 작은 예제입니다.
# B=1: 문장 한 개
# L=3: 토큰 세 개
# D=2: 각 토큰 표현 차원 두 개
q = torch.tensor(
    [[[1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0]]],
    dtype=torch.float32,
)

# 이해를 단순하게 하기 위해 K도 같은 값을 사용합니다.
# 실제 Transformer에서는 Q와 K가 서로 다른 projection을 통과합니다.
k = q.clone()

# Value는 가져올 실제 정보입니다.
# 여기서는 계산 결과를 읽기 쉽도록 간단한 값을 사용합니다.
v = torch.tensor(
    [[[1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0]]],
    dtype=torch.float32,
)

# K의 마지막 두 축을 바꿉니다.
# K:   [B, L, D] = [1, 3, 2]
# K^T: [B, D, L] = [1, 2, 3]
k_t = k.transpose(-2, -1)

# 모든 Query와 모든 Key의 dot-product score를 한 번에 계산합니다.
# [1, 3, 2] @ [1, 2, 3] -> [1, 3, 3]
scores = torch.matmul(q, k_t)

# d_k는 Query/Key의 마지막 차원입니다.
d_k = q.size(-1)

# score의 크기를 조절합니다. Shape은 바뀌지 않습니다.
scaled_scores = scores / math.sqrt(d_k)

# 각 Query 행에서 모든 Key에 대한 참고 비율을 만듭니다.
# 마지막 축이 Key 축이므로 dim=-1을 사용합니다.
weights = torch.softmax(scaled_scores, dim=-1)

# Attention weight만큼 Value를 섞습니다.
# [1, 3, 3] @ [1, 3, 2] -> [1, 3, 2]
context = torch.matmul(weights, v)

print("scores shape:", scores.shape)
print("weights shape:", weights.shape)
print("context shape:", context.shape)

# 소수 셋째 자리까지 반올림해 출력합니다.
print("scores:\n", scores.round(decimals=3))
print("weights:\n", weights.round(decimals=3))
print("context:\n", context.round(decimals=3))

# 각 Query의 weight 합이 1인지 확인합니다.
print("weight row sums:", weights.sum(dim=-1))

assert scores.shape == (1, 3, 3)
assert weights.shape == (1, 3, 3)
assert context.shape == (1, 3, 2)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(1, 3),
    atol=1e-6,
)
```

### 관찰할 내용

1. `scores`와 `weights`는 토큰×토큰 표이므로 `[1,3,3]`다.
2. Softmax 이후 각 행의 합은 1이다
3. `context`는 토큰별 새 표현이므로 입력과 같은 `[1,3,2]`다.
4. 가장 큰 weight 하나만 남는 것이 아니라 여러 Value가 함께 섞인다.

## 참고. 이해도 점검

1. Attention 계산의 네 단계를 순서대로 말해 보세요.
2. Score matrix의 행과 열은 각각 무엇인가요?
3. Score와 Attention weight의 차이는 무엇인가요?
4. Softmax를 마지막 축에 적용하는 이유는 무엇인가요?
5. Q, K, V가 `[B,L,D]`일 때 score와 output shape은 무엇인가요?

### 정답 확인

1. Q/K 비교 -> scaling -> softmax -> Value 가중합
2. 행은 Query, 열은 Key다.
3. Score는 정규화 전 관련도이고, weight는 Softmax를 적용해 합이 1이 된 참고 비율
4. 마지막 축이 한 Query가 비교하는 모든 Key 후보 축이기 때문입니다.
5. Score는 `[B,L,L]`, output은 `[B,L,D]`다.

## 9. Scaling과 Mask를 수식·Shape로 설명(Multi-Head Attention 기준)

- Scaling은 shape을 바꾸지 않는다.
- `d_k`는 전체 `D_model`이 아니라 **head 하나의 Query/Key 차원**이다.
- Mask는 Softmax 전에 score에 적용한다


Batch와 head가 있는 score `[B,H,L_q,L_k]`에 padding mask `[B,L_k]`를 적용할 때는 `[B,1,1,L_k]`로 확장해 broadcast해야한다


```python
key_mask = attention_mask[:, None, None, :].bool()
scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
weights = torch.softmax(scores, dim=-1)
```

## 요약

- Scaled Dot-Product Attention은 비교, 조절, 비율, 정보 혼합의 네 단계다
- `QKᵀ`는 모든 Query와 Key의 관계를 담은 score matrix를 만든다
- Scaling은 큰 score가 Softmax를 지나치게 뾰족하게 만드는 것을 완화한다
- Softmax는 score를 합이 1인 Attention weight로 바꾼다
- Weight와 V를 곱하면 문맥이 반영된 새로운 토큰 표현이 만들어진다


