---
title: 4-3강 Encoder Block과 Encoder-only 구조(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-3강 Encoder Block과 Encoder-only 구조' 정리
---

## 1. Encoder Block의 핵심 역할



<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_encoder_block_roles.png
' | relative_url }}" alt="01_encoder_block_roles.png
" loading="lazy">

### 1-1. 토큰 간 소통: Self-Attention

각 토큰이 같은 입력 시퀀스의 다른 실제 토큰을 참고해 문맥이 반영된 표현을 만든다

```
"은행" + 주변 문맥
-> 금융기관인지 강가인지 구분하는 표현
```

### 1-2. 토큰별 가공: Position-wise FFN

Attention으로 문맥 정보를 모은 뒤, 각 토큰 위치에 같은 MLP를 적용해 표현을 비선형적으로 변환

```
각 토큰의 D차원 벡터
-> D_ff로 확장
-> Activation
-> D로 축소
```

## 2. Encoder Layer의 흐름

흐름을 단순화

```
X [B,L,D]
-> Self-Attention
-> Residual + LayerNorm
-> FFN : FFN으로 각 토큰 표현 가공
-> Residual + LayerNorm
-> Y [B,L,D]
```

Residual Connection은 변환한 결과에 원래 입력을 다시 더해주는 것

Residual = 원래 입력 + 새로 계산한 결과

"Attention이 새로 알아낸 정보만 쓰지 말고, 원래 가지고 있던 정보도 같이 가져가자." 라는 의미

LayerNorm 한 토큰의 D개 feature 값을 적당한 범위로 정리해주는 정규화 방법

LayerNorm
→ 각 토큰 내부의 D차원을 기준으로 정규화, 평균을 0 근처, 분산을 1 근처가 되도록 크기를 조정한다



Residual
= 원래 입력을 변환 결과에 더해서 기존 정보를 보존하고 학습을 쉽게 함

LayerNorm
= 각 토큰의 D차원 값들을 정규화해서 학습을 안정적으로 만듦


### 왜 shape을 유지할까요?

Residual Connection에서 원래 입력과 Sublayer 출력을 더하려면 shape이 같아야 한다

```
X:           [B,L,D]
Sublayer(X): [B,L,D]
```

Shape이 유지된다는 것은 값이 그대로라는 뜻이 아니다

각 Layer를 지나면서 값은 달라지고 더 깊은 문맥 정보가 반영된다

## 3. 여러 Layer를 Stack으로 쌓기

Encoder는 동일한 구조의 Layer를 여러 개 쌓는다.

```
입력 [B,L,D]
-> Encoder Layer 1 [B,L,D]
-> Encoder Layer 2 [B,L,D]
-> Encoder Layer 3 [B,L,D]
```

각 Layer마다 파라미터는 일반적으로 서로 다르다.
"동일한 Layer를 쌓는다"는 말은 **구조가 같다**는 뜻이지, 파라미터를 반드시 공유한다는 뜻은 아니다

## 4. Encoder-only 모델과 Task Head

Encoder-only 모델은 입력 전체를 양방향으로 문맥화하는 데 강하다

### 문장 분류

```
hidden [B,L,D]
-> 대표 문장 표현 또는 pooling
-> Classification Head
-> logits [B,C]
```

### 토큰 분류

```
hidden [B,L,D]
-> 각 토큰 위치에 Linear Head
-> logits [B,L,C]
```

대표 활용:

- 고객 문의 intent 분류
- 감성 분류
- 개체명 인식
- 개인정보 포함 토큰 탐지


## 5. Padding Mask 연결

Padding Mask

Batch 안 문장 길이가 다르면 짧은 문장에 `[PAD]`를 추가합니다.

Encoder Self-Attention이 이 위치를 실제 정보처럼 참고하지 않도록 Padding Mask를 전달한다

### Hugging Face에서 흔한 의미

```
attention_mask = 1: 실제 토큰
attention_mask = 0: padding
```

### PyTorch key padding mask에서 흔한 의미

```
True: 무시할 위치
False: 사용할 위치
```

> 같은 mask라도 API마다 값의 의미가 반대일 수 있다. 이름만 보고 추측하지 말고 공식 문서를 확인해야한다.


## 선택 · 6. 연습 문제: PyTorch Encoder Stack 실행

```python
import torch
from torch import nn

# 같은 가상 입력과 초기 가중치를 만들기 위해 seed를 고정합니다.
torch.manual_seed(42)

BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
D_MODEL = 16
NUM_HEADS = 4
NUM_LAYERS = 2

# Encoder Layer 한 층의 구조를 정의합니다.
# batch_first=True이므로 입력/출력 Shape은 [B,L,D]입니다.
encoder_layer = nn.TransformerEncoderLayer(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    dim_feedforward=32,
    dropout=0.0,          # 결과 비교를 단순하게 하기 위해 Dropout을 끕니다.
    batch_first=True,
    norm_first=True,      # Pre-Norm 흐름을 사용합니다.
)

# 위 Layer 구조를 NUM_LAYERS개 쌓아 Encoder Stack을 만듭니다.
encoder = nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=NUM_LAYERS,
)

# Token Embedding과 Position 정보까지 합쳐진 입력이라고 가정합니다.
# Shape: [B,L,D]
x = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
)

# 두 번째 샘플의 마지막 두 위치가 padding이라고 가정합니다.
# PyTorch src_key_padding_mask에서 True는 Attention이 무시할 Key 위치입니다.
# Shape: [B,L]
padding_mask = torch.tensor(
    [
        [False, False, False, False, False],
        [False, False, False, True, True],
    ],
    dtype=torch.bool,
)

# Encoder Stack을 통과합니다.
# Padding Mask가 실제 토큰의 문맥 계산에 PAD 위치가 사용되지 않도록 합니다.
encoded = encoder(
    src=x,
    src_key_padding_mask=padding_mask,
)

print("input shape:", x.shape)
print("padding_mask shape:", padding_mask.shape)
print("encoded shape:", encoded.shape)

# Encoder Layer는 대표적으로 입력과 같은 [B,L,D] shape을 반환합니다.
assert encoded.shape == x.shape
```

Padding Mask는 PAD 위치를 Key/Value로 참고하지 않도록 제한한다.

그러나 출력 Tensor의 PAD Query 위치가 자동으로 모두 0이 된다는 뜻은 아니다

이후 pooling이나 loss 계산에서도 실제 토큰 위치를 구분해야 한다




## 선택 · 7. 연습 문제: Masked Mean Pooling으로 분류 입력 만들기

문장 분류에서는 실제 토큰의 hidden state만 평균내 하나의 문장 벡터를 만들 수 있다.

```python
# padding_mask는 True가 PAD 위치입니다.
# 실제 토큰 위치를 1로 만들기 위해 논리 반전을 적용합니다.
# Shape: [B,L]
valid_token_mask = ~padding_mask

# Hidden dimension과 곱할 수 있도록 마지막 축을 추가합니다.
# Shape: [B,L,1]
valid_token_mask_3d = valid_token_mask.unsqueeze(-1)

# bool mask를 hidden state와 곱할 수 있도록 같은 dtype으로 바꿉니다.
valid_token_mask_3d = valid_token_mask_3d.to(encoded.dtype)

# PAD 위치의 hidden state를 0으로 만듭니다.
# Shape: [B,L,D]
masked_hidden = encoded * valid_token_mask_3d

# 문장별 실제 토큰 수를 계산합니다.
# Shape: [B,1]
valid_counts = valid_token_mask_3d.sum(dim=1)

# 실제 토큰 hidden state만 합한 뒤 토큰 수로 나눕니다.
# Shape: [B,D]
pooled = masked_hidden.sum(dim=1) / valid_counts.clamp_min(1.0)

# 두 클래스의 점수를 만드는 Classification Head입니다.
classifier = nn.Linear(D_MODEL, 2)

# Shape: [B,D] -> [B,C]
logits = classifier(pooled)

print("pooled shape:", pooled.shape)
print("logits shape:", logits.shape)

assert pooled.shape == (BATCH_SIZE, D_MODEL)
assert logits.shape == (BATCH_SIZE, 2)
```

## 참고 · 9. 이해도 점검

1. Encoder Block의 두 핵심 작업은 무엇인가
2. FFN은 토큰 사이 정보를 섞나요?
3. Encoder Layer 전후에 `[B,L,D]`가 유지되는 이유를 설명
4. PyTorch key padding mask에서 `True`는 일반적으로 무엇을 뜻하나
5. 문장 분류 logits를 만들기 전에 `[B,L,D]`를 어떤 shape으로 줄일 수 있나

### 정답 확인

1. Self-Attention을 통한 토큰 간 정보 교환과 FFN을 통한 토큰별 표현 가공입니다.
2. 아니요. 각 위치에 같은 MLP를 독립적으로 적용
3. Residual 덧셈과 Stack 연결을 위해 Sublayer가 hidden dimension D로 다시 출력하기 때문
4. Attention에서 무시할 padding 위치를 뜻한다
5. Pooling 등을 통해 `[B,D]` 문장 표현으로 줄일 수 있다.

## 요약

- Encoder Block은 Self-Attention과 FFN을 중심으로 구성됩니다.
- Attention은 토큰끼리 정보를 교환하고, FFN은 각 토큰을 개별적으로 가공합니다.
- Residual과 LayerNorm은 Block을 여러 층 쌓는 학습 경로를 돕습니다.
- Encoder Stack의 대표 입출력 shape은 `[B,L,D]`입니다.
- Encoder-only 모델은 분류, 검색, token classification 같은 이해 중심 문제에 활용됩니다.
- 다음 강의에서는 미래 토큰을 가리는 Decoder-only 구조와 Causal Mask를 학습합니다.