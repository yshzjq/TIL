---
title: 3-5강 Multi-Head Attention Shape과 디버깅
date: 2026-09-03
updated: 2026-09-03
description: KANT 강의 '3-5강 Multi-Head Attention Shape과 디버깅' 정리
---

## 1. 왜 여러 Head를 사용할까?

단일 Attention도 여러 토큰을 참고할 수 있다

Multi-Head Attention은 같은 입력을 여러 head가 서로 다른 projection으로 바라볼 기회를 준다



<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_heads_perspectives.png
' | relative_url }}" alt="01_heads_perspectives.png
" loading="lazy">


여러 Head의 관점

한 head는 주어와 동사의 관계를, 다른 head는 수식어와 명사의 관계를,<br> 또 다른 head는 멀리 떨어진 토큰 관계를 학습할 수 있다.

## 2. D_model, num_heads, D_head

Multi-Head Attention에서는 전체 표현 차원 `D_model`을 head 수 `H`로 나눈다

```
D_head = D_model / num_heads
```

각 head가 같은 크기를 가져야 하므로 다음 조건이 필요하다

```
D_model % num_heads == 0
```

| D_model | num_heads | D_head | 가능 여부 |
| --- | --- | --- | --- |
| 8 | 2 | 4 | 가능 |
| 12 | 3 | 4 | 가능 |
| 16 | 4 | 4 | 가능 |
| 10 | 3 | 정수가 아님 | 불가능 |


### 2-1. Q, K, V는 입력을 그대로 세 번 복사한 값이 아니다

Self-Attention에서는 Q, K, V의 출발점이 같은 입력 `X`이지만, 실제 값은 서로 다른 학습 가능한 선형변환을 거쳐 만들어진다

```
X: [B,L,D_model]

Q = XW_Q
K = XW_K
V = XW_V

Q, K, V: 각각 [B,L,D_model]
```

`W_Q`, `W_K`, `W_V`는 사람이 미리 정한 규칙이 아니라 학습 중 역전파로 업데이트되는 파라미터다. 

즉 W_Q, W_K, W_V가 학습하면서 값이 계속 조금씩 수정된다는 뜻

W_Q, W_K, W_V는 고정된 공식이 아니라 모델이 학습하면서 loss를 줄이는 방향으로 값이 계속 조정되는 가중치 행렬



같은 토큰 표현 `X`를 사용하더라도 각 투영 행렬이 서로 다르기 때문에 다음 역할에 맞는 표현을 따로 학습할 수 있다.

### 2-2. Head 분리는 선형 투영 다음에 이루어진다

일반적인 구현 흐름

```
입력 X [B,L,D_model]
-> Q, K, V 선형 투영 [B,L,D_model]
-> Head 분리 [B,H,L,D_head]
-> Head별 Attention 계산
-> Head 결과 [B,H,L,D_head]
-> Head 결합 [B,L,D_model]
-> Output Projection W_O
-> 최종 출력 [B,L,D_model]
```

각 head는 전체 차원 중 `D_head`만 사용하지만, 단순히 원본 `X`를 앞부분·뒷부분으로 잘라 보는 것이 아니다<br> 
먼저 학습된 Projection을 적용한 뒤 head 축으로 재배열한다


### 2-3. Concatenate 뒤의 W_O는 왜 필요한가

Head별 Context Vector를 이어 붙이면 다시 `[B,L,D_model]`이 됩니다. 하지만 이어 붙이기만 하면 각 head의 결과가 단순히 옆에 배치된 상태입니다.

마지막의 학습 가능한 `W_O`는 여러 head의 결과를 다시 섞어 다음 Sublayer가 사용할 표현으로 바꾼다

W_O는 여러 Head가 따로 만든 Context를 하나의 통합된 토큰 표현으로 다시 섞어주는 Linear layer다.

```
Concat(head_1, ..., head_H): [B,L,D_model]
Output = Concat(...) W_O:    [B,L,D_model]
```

`W_O`까지 포함해야 Multi-Head Attention의 출력 투영이 완성된다.

여기서 투영(projection)**은 Linear layer를 이용해서 벡터를 다른 표현으로 바꾸는 것

PyTorch 같은 라이브러리는 내부 최적화를 위해 `W_Q`, `W_K`, `W_V`를 하나의 큰 행렬로 묶어 저장할 수도 있지만, 

개념적으로는 세 투영을 구분해서 이해하면 된다
<br>실제 코드에서는 W_Q, W_K, W_V가 하나의 큰 Tensor로 묶여 있어도, 공부할 때는 역할을 세 개로 따로 생각하라는 뜻

### 2-4. Tensor Shape으로 전체 흐름 확인하기

```python
import math

import torch
from torch import nn

BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
D_MODEL = 8
NUM_HEADS = 2

assert D_MODEL % NUM_HEADS == 0
D_HEAD = D_MODEL // NUM_HEADS

# Transformer Block에 들어오는 토큰 표현입니다.
# Shape: [B, L, D_model]
x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)

# 서로 다른 학습 가능한 선형변환으로 Q, K, V를 만듭니다.
q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

q = q_proj(x)
k = k_proj(x)
v = v_proj(x)

# [B, L, D_model] -> [B, L, H, D_head] -> [B, H, L, D_head]
q_heads = q.reshape(BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, D_HEAD).transpose(1, 2)
k_heads = k.reshape(BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, D_HEAD).transpose(1, 2)
v_heads = v.reshape(BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, D_HEAD).transpose(1, 2)

# 모든 Query-Key 조합의 점수를 계산합니다.
# [B,H,L,D_head] @ [B,H,D_head,L] -> [B,H,L,L]
scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(D_HEAD)
weights = torch.softmax(scores, dim=-1)

# Head별로 Value를 섞습니다.
# [B,H,L,L] @ [B,H,L,D_head] -> [B,H,L,D_head]
head_context = torch.matmul(weights, v_heads)

# [B,H,L,D_head] -> [B,L,H,D_head] -> [B,L,D_model]
merged_context = (
    head_context.transpose(1, 2)
    .contiguous()
    .reshape(BATCH_SIZE, SEQUENCE_LENGTH, D_MODEL)
)

# 여러 head의 정보를 다시 섞는 Output Projection입니다.
output = out_proj(merged_context)

assert q_heads.shape == (BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, D_HEAD)
assert weights.shape == (BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
assert output.shape == x.shape

print("Q head shape:", q_heads.shape)
print("Attention weight shape:", weights.shape)
print("MHA output shape:", output.shape)
```
이 코드는 교육용으로 각 Projection을 따로 만들었다.

 실제 `nn.MultiheadAttention`은 성능을 위해 Projection 파라미터를 묶어 계산할 수 있지만, 수학적 흐름과 출력 shape은 같은 관점으로 읽을 수 있다.

8차원 전체 내적과 4차원 두 개의 내적을 단순히 더하는 것만 보면 같을 수 있다. 하지만 Multi-Head Attention은 각 Head가 별도의 score → Softmax → Value 가중합을 수행하므로 결과적으로 같은 계산이 아니다


## 3. Head로 나누고 다시 합치기

Head split과 concat

입력이 `[B,L,D_model]`이라고 가정

```
[B, L, D_model]
-> D_model을 H × D_head로 나누기
[B, L, H, D_head]
-> Head 축을 앞으로 이동
[B, H, L, D_head]
```

각 head에서 Attention이 끝난 뒤에는 반대 순서로 다시 합친다

```
[B, H, L, D_head]
-> [B, L, H, D_head]
-> [B, L, D_model]
```

전체 표현 차원 `D_model`은 나누기 전과 합친 후에 동일하다

### 3-1. 왜 Head 축을 앞으로 옮기나요?

`[B,H,L,D_head]` 구조로 만들면 B개의 샘플과 H개의 head를 한 번에 병렬 계산하기 편합니다. 


[B, H, L, D_head]
@
[B, H, D_head, L]

→ [B, H, L, L]
즉 앞의 B, H는 유지하고 마지막 두 차원끼리 행렬곱



## 4. PyTorch MultiheadAttention 입출력

PyTorch MHA 입출력

`batch_first=True`일 때 기본 shape

| 항목 | Shape |
| --- | --- |
| Query 입력 | `[B,L_query,D_model]` |
| Key 입력 | `[B,L_key,D_model]` |
| Value 입력 | `[B,L_key,D_model]` |
| Output | `[B,L_query,D_model]` |
| Head별 weight | `[B,H,L_query,L_key]` |

Self-Attention에서는 Query, Key, Value에 같은 `x`를 전달


## 5. 연습 문제: Head split/merge Shape 확인

```python
import torch

# 예시 shape을 정합니다.
B = 2          # 배치 크기
L = 5          # 토큰 길이
D_MODEL = 8    # 전체 토큰 표현 차원
NUM_HEADS = 2  # Head 수

# 각 head의 차원이 정수인지 먼저 확인합니다.
assert D_MODEL % NUM_HEADS == 0
D_HEAD = D_MODEL // NUM_HEADS

# 입력은 [B, L, D_model]입니다.
x = torch.randn(B, L, D_MODEL)

# 마지막 차원 D_model을 H × D_head로 나눕니다.
# [B, L, D_model] -> [B, L, H, D_head]
split = x.reshape(B, L, NUM_HEADS, D_HEAD)

# Head 축과 토큰 길이 축의 순서를 바꿉니다.
# [B, L, H, D_head] -> [B, H, L, D_head]
split = split.transpose(1, 2)

print("x shape:", x.shape)
print("split shape:", split.shape)

# Attention 계산이 끝났다고 가정하고 다시 합칩니다.
# [B, H, L, D_head] -> [B, L, H, D_head]
merged = split.transpose(1, 2)

# H와 D_head를 다시 D_model 하나의 축으로 합칩니다.
# reshape는 같은 원소 수를 유지하면서 [B, L, D_model]로 되돌립니다.
merged = merged.reshape(B, L, D_MODEL)

print("merged shape:", merged.shape)

assert split.shape == (B, NUM_HEADS, L, D_HEAD)
assert merged.shape == x.shape

# 이 예제에서는 값 변경 없이 나누고 다시 합쳤으므로 원래 x와 같습니다.
assert torch.allclose(merged, x)
```
출력
```
x shape: torch.Size([2, 5, 8])
split shape: torch.Size([2, 2, 5, 4])
merged shape: torch.Size([2, 5, 8])

```
merged = split.transpose(1, 2) 하는 이유

- Q @ Kᵀ 행렬곱을 하기 위해서 shape를 맞추어야 함
- 2,5,2,4[B, L, H, D_head] 로는 행렬곱을 할 수 없음 
- 토큰과 토큰의 관계 계산이 목적
- [B, H, L, D_head] 축 변경 축 변경해도 이 행렬의 의미는 변하지 않음
- 2,2,5,4(Q) [B, L, H, D_head] @ 2,2,4,5(Kᵀ)  [B, L, D_head, H] 행렬곱을 진행
- 2,2,5,5

transpose는 데이터의 값을 바꾸는 게 아니라 축의 배치 순서를 바꾼다. 다만 축 순서가 바뀌면 이후 연산에서 그 Tensor를 해석하고 계산하는 방식은 달라진다.

정보는 그대로 유지되고 구조만 바뀐다. 그리고 구조를 [B,H,L,D_head]로 바꿈으로써 각 Head마다 토큰끼리 비교하는 Attention 행렬곱을 할 수 있게 된다.




### 관찰할 내용

- `D_model=8`, `num_heads=2`이면 `D_head=4`다.
- Head로 나눈 shape은 `[2,2,5,4]`
- 다시 합치면 입력과 같은 `[2,5,8]`이 된다

## 6. 연습 문제: nn.MultiheadAttention 실행

```python
import torch
from torch import nn

# 결과를 재현하기 위해 seed를 고정합니다.
torch.manual_seed(42)

B = 2
L = 5
D_MODEL = 8
NUM_HEADS = 2

# PyTorch MultiheadAttention 모듈을 만듭니다.
# batch_first=True이면 입력과 출력이 [B, L, D] 순서입니다.
mha = nn.MultiheadAttention(
    embed_dim=D_MODEL,
    num_heads=NUM_HEADS,
    batch_first=True,
)

# 문장 두 개, 토큰 다섯 개, 표현 차원 여덟 개의 입력입니다.
x = torch.randn(B, L, D_MODEL)

# Self-Attention이므로 query, key, value에 같은 x를 전달합니다.
# need_weights=True는 Attention weight도 함께 반환합니다.
# average_attn_weights=False는 head 평균을 내지 않고 H 축을 유지합니다.
output, weights = mha(
    query=x,
    key=x,
    value=x,
    need_weights=True,
    average_attn_weights=False,
)

print("input shape:", x.shape)
print("output shape:", output.shape)
print("weights shape:", weights.shape)

# Output은 Query 위치마다 D_model 차원의 새 표현을 반환합니다.
assert output.shape == (B, L, D_MODEL)

# Head별 weight는 [B, H, L_query, L_key]입니다.
assert weights.shape == (B, NUM_HEADS, L, L)

# 각 head와 Query 행에서 Key weight 합이 1인지 확인합니다.
row_sums = weights.sum(dim=-1)
print("weight row sums shape:", row_sums.shape)
print("first sample, first head row sums:", row_sums[0, 0])

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6,
)
```

출력

```
input shape: torch.Size([2, 5, 8])
output shape: torch.Size([2, 5, 8])
weights shape: torch.Size([2, 2, 5, 5])
weight row sums shape: torch.Size([2, 2, 5])
first sample, first head row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], grad_fn=<SelectBackward0>)
```

### 6-1. `average_attn_weights=True`라면?

기본값에서는 여러 head의 weight를 평균해 Head 축이 사라질 수 있다.

```
Head별 weight: [B,H,L,L]
Head 평균 weight: [B,L,L]
```

Head별 패턴을 보고 싶다면 `average_attn_weights=False`를 사용한다

## 참고. 이해도 점검

1. Multi-Head Attention을 쉬운 문장으로 설명해 보세요.
2. `D_model=12`, `num_heads=3`일 때 `D_head`는 얼마인가
3. `[B,L,D_model]`을 head로 나눈 뒤 대표 shape은 무엇인가
4. Head별 Attention weight를 보려면 어떤 옵션을 설정해야 하나
5. `D_model % num_heads == 0` 조건이 필요한 이유는 무엇인가

### 정답 확인

1. 같은 입력을 여러 head가 서로 다른 관점으로 Attention한 뒤 결과를 합치는 구조다.
2. 4
3. `[B,H,L,D_head]`입니다.
4. `average_attn_weights=False`로 설정합니다.
5. 각 head가 동일한 정수 크기의 `D_head`를 가져야 하기 때문이다.

## 9. MHA·MQA·GQA의 Q/K/V Head 수 (선택 학습)

Decoder-only LLM의 config에서는 Query head 수와 Key/Value head 수가 다를 수 있습니다.

| 구조 | Query head | Key/Value head | 특징 |
| --- | --- | --- | --- |
| MHA | `H_q` | `H_kv = H_q` | 각 Query head가 별도 K/V head를 사용합니다. |
| MQA | `H_q` | `H_kv = 1` | 모든 Query head가 하나의 K/V를 공유합니다. |
| GQA | `H_q` | `1 < H_kv < H_q` | Query head 그룹이 K/V head를 공유합니다. |

MHA
Q1 → K1,V1
Q2 → K2,V2
Q3 → K3,V3
...

MQA
Q1 ─┐
Q2 ─┤
Q3 ─┤→ K1,V1 하나를 모두 공유
Q4 ─┘

GQA
Q1,Q2,Q3,Q4 → K1,V1 공유
Q5,Q6,Q7,Q8 → K2,V2 공유

Attention 계산은 다 똑같다



예를 들어 `num_attention_heads=32`, `num_key_value_heads=8`이면 Query head 네 개가 K/V head 하나를 공유합니다.

MQA = Q는 여러 Head 그대로 두고, K/V만 1개씩 공유한다




- Q shape: `[B,32,L,D_head]`
- K/V shape: `[B,8,L,D_head]`
- Attention 계산에서는 K/V를 Query 그룹과 대응시킨다

KV Cache 크기는 `H_q`가 아니라 `H_kv`에 비례하므로 GQA는 긴 문맥 추론의 메모리를 줄이는 데 유리합니다.


## 요약

- 선택 학습에서는 MHA·MQA·GQA의 Query와 K/V head 수 차이와 KV Cache 변화를 비교한다
- Multi-Head Attention은 같은 입력을 여러 head가 서로 다른 관점으로 처리할 기회를 준다
- `D_head = D_model / num_heads`이며 나머지가 0이어야 합니다.
- 입력 `[B,L,D_model]`은 `[B,H,L,D_head]`로 나뉘고 다시 `[B,L,D_model]`로 합쳐진다
- PyTorch `nn.MultiheadAttention`으로 output과 head별 weight shape을 관찰할 수 있다.
- 디버깅에서는 `D_model`, `num_heads`, `batch_first`, `average_attn_weights`를 먼저 확인한다