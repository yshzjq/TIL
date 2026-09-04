---
title: 4-4강 Decoder-only Block과 Causal Mask(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-4강 Decoder-only Block과 Causal Mask' 정리
---

## 1. Decoder-only란 무엇인가?



<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_decoder_only_flow.png
' | relative_url }}" alt="01_decoder_only_flow.png
" loading="lazy">

Decoder-only 흐름

GPT형 Decoder-only 모델은 별도의 Encoder 없이 하나의 토큰 시퀀스를 처리한다

```
Token + Position
-> Causal Multi-Head Self-Attention
-> FFN
-> Residual + LayerNorm
-> 여러 Block 반복
-> LM Head
-> 다음 토큰 logits
```

### Encoder Block과 비슷한 점

- Multi-Head Self-Attention
- FFN
- Residual Connection
- LayerNorm

### 다른 점

- 미래 토큰을 보지 못하도록 Causal Mask를 사용한다
- 별도 Encoder가 없으므로 Encoder memory를 보는 Cross-Attention도 없다.



## 2. Causal Mask를 행 단위로 읽기

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_causal_mask_matrix.png
' | relative_url }}" alt="02_causal_mask_matrix.png
" loading="lazy">

Causal Mask 행렬

Causal Mask의 각 행은 **한 Query 위치가 어떤 Key 위치까지 볼 수 있는지** 나타낸다

예를 들어 토큰이 5개라면 다음처럼 읽습니다.

| Query 위치 | 볼 수 있는 Key 위치 |
| --- | --- |
| 0 | 0 |
| 1 | 0, 1 |
| 2 | 0, 1, 2 |
| 3 | 0, 1, 2, 3 |
| 4 | 0, 1, 2, 3, 4 |

따라서 Mask 표에서는 대각선과 아래쪽은 허용하고, 위쪽 미래 영역은 차단한다

> 현재 토큰은 볼 수 있습니다. 다음 토큰을 예측할 때 현재 위치의 hidden state가 지금까지 주어진 모든 토큰을 반영해야 하기 때문이다.

## 3. 다음 토큰 예측과 Label Shift

문장 `나는 AI를 배운다`를 예로 들면 Causal LM 학습은 다음 쌍을 만든다

```
입력:  <BOS> 나는 AI를 배운다
정답:  나는  AI를 배운다 <EOS>
```

## 4. 학습과 생성의 차이

### 학습

정답 문장 전체가 이미 있으므로 여러 위치의 다음 토큰 loss를 한 번에 계산할 수 있다. 

다만 Causal Mask로 각 위치가 미래 정답을 직접 보지 못하게 한다

### 생성

다음 토큰이 아직 없으므로 한 번 예측하고 입력 뒤에 추가하는 과정을 반복한다

```
Prompt
-> 다음 토큰 예측
-> 입력에 추가
-> 다시 다음 토큰 예측
-> 반복
```

따라서 Transformer 학습은 위치 병렬화가 가능하지만,<br> Autoregressive 생성은 기본적으로 순차적이다.

## 선택 · 5. 원래 Decoder와 Decoder-only 비교

!04_original_vs_decoder_only.png

원래 Decoder와 Decoder-only

| 구성 | 원래 Encoder-Decoder의 Decoder | GPT형 Decoder-only |
| --- | --- | --- |
| Causal Self-Attention | 있음 | 있음 |
| Cross-Attention | 있음 | 없음 |
| 별도 Encoder | 있음 | 없음 |
| 대표 목적 | 입력 문장을 조건으로 출력 생성 | 하나의 시퀀스를 이어서 생성 |

## 6. 연습 문제: Causal Mask 만들기

```python
import torch

def make_causal_mask(sequence_length: int) -> torch.Tensor:
    """미래 Key 위치를 True로 표시하는 bool Causal Mask를 만듭니다."""

    # 모든 위치 쌍을 False로 시작합니다.
    # Shape: [L,L]
    mask = torch.zeros(
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    )

    # diagonal=1은 주대각선 바로 위부터 선택합니다.
    # 따라서 현재 위치보다 오른쪽에 있는 미래 위치만 True가 됩니다.
    mask = torch.triu(
        torch.ones_like(mask),
        diagonal=1,
    )

    # PyTorch MultiheadAttention의 bool attn_mask에서는
    # True가 Attention에서 차단할 위치를 뜻합니다.
    return mask

causal_mask = make_causal_mask(sequence_length=5)

print(causal_mask)
print("shape:", causal_mask.shape)

assert causal_mask.shape == (5, 5)
```

### 예상 결과

```
[[False,  True,  True,  True,  True],
 [False, False,  True,  True,  True],
 [False, False, False,  True,  True],
 [False, False, False, False,  True],
 [False, False, False, False, False]]
```

## 7. 연습 문제: 미래 토큰 변경 실험

Causal Mask가 제대로 동작한다면 미래 위치의 입력을 바꾸어도 첫 번째 Query의 출력은 바뀌지 않아야 한다

```python
import torch
from torch import nn

# 같은 가중치와 입력을 재현하기 위해 seed를 고정합니다.
torch.manual_seed(42)

BATCH_SIZE = 1
SEQUENCE_LENGTH = 4
D_MODEL = 8
NUM_HEADS = 2

# 교육용 Multi-Head Self-Attention입니다.
# dropout=0.0과 eval()을 사용해 두 실행을 정확히 비교합니다.
attention = nn.MultiheadAttention(
    embed_dim=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=0.0,
    batch_first=True,
)
attention.eval()

# 원본 입력입니다.
# Shape: [B,L,D]
x_original = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
)

# 미래 위치 하나를 크게 바꾼 복사본입니다.
x_changed = x_original.clone()
x_changed[:, 3, :] = x_changed[:, 3, :] + 100.0

# 미래 Key 위치를 True로 차단합니다.
causal_mask = torch.triu(
    torch.ones(
        SEQUENCE_LENGTH,
        SEQUENCE_LENGTH,
        dtype=torch.bool,
    ),
    diagonal=1,
)

with torch.no_grad():
    # Self-Attention이므로 Query, Key, Value에 같은 입력을 전달합니다.
    output_original, weights_original = attention(
        query=x_original,
        key=x_original,
        value=x_original,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    output_changed, weights_changed = attention(
        query=x_changed,
        key=x_changed,
        value=x_changed,
        attn_mask=causal_mask,
        need_weights=True,
        average_attn_weights=False,
    )

# 첫 번째 Query 위치는 자기 자신 위치 0만 볼 수 있습니다.
# 따라서 미래 위치 3을 바꾸어도 첫 번째 위치 출력은 같아야 합니다.
first_position_equal = torch.allclose(
    output_original[:, 0, :],
    output_changed[:, 0, :],
    atol=1e-6,
)

# 마지막 위치는 0~3 전체를 볼 수 있으므로 바뀐 미래 토큰의 영향을 받습니다.
last_position_equal = torch.allclose(
    output_original[:, 3, :],
    output_changed[:, 3, :],
    atol=1e-6,
)

print("output shape:", output_original.shape)
print("weights shape:", weights_original.shape)
print("첫 번째 위치 출력이 같은가?:", first_position_equal)
print("마지막 위치 출력이 같은가?:", last_position_equal)

assert first_position_equal is True
assert last_position_equal is False
```


## 참고 · 9. 이해도 점검

1. Causal Mask의 한 행은 무엇을 나타내나
2. Query 위치 2는 어떤 Key 위치까지 볼 수 있나
3. 학습 때 Target 전체를 넣어도 미래 정보 누출이 발생하지 않게 하는 장치는 무엇인가
4. 생성이 순차적인 이유는 무엇인가
5. 원래 Decoder와 Decoder-only의 가장 큰 구조적 차이는 무엇인가

### 정답 확인

1. 해당 Query 위치가 참고할 수 있는 Key 위치를 나타낸다
2. 0, 1, 2 위치를 볼 수 있습니다.
3. Causal Mask입니다.
4. 방금 생성한 토큰을 다음 단계의 입력으로 사용해야 하기 때문이다.
5. 원래 Decoder에는 Encoder memory를 보는 Cross-Attention이 있지만 Decoder-only에는 없다.


## 10. Padding Mask와 Causal Mask를 함께 적용할 때


Causal mask는 미래 Key를 막고, padding mask는 존재하지 않는 Key를 막습니다. 
<br>두 조건은 목적이 다르므로 decoder-only batch에서는 함께 필요할 수 있다.

- score shape: `[B,H,L_q,L_k]`
- causal mask: `[1,1,L_q,L_k]`
- key padding mask: `[B,1,1,L_k]`

두 mask를 broadcast해 합친 뒤 Softmax 전에 적용한다

Left padding을 사용하는 생성 batch에서는 position과 cache 위치까지 모델 API가 기대하는 방식으로 맞춰야 합니다.

왼쪽에 PAD를 추가했을 때 실제 토큰의 위치 번호와 KV Cache의 위치가 꼬이지 않도록, 사용하는 모델 API 방식에 맞게 처리해야 한다

## 요약

- Decoder-only는 Causal Self-Attention과 FFN을 반복하는 생성 중심 구조다
- Causal Mask는 각 위치가 미래 토큰을 보지 못하게 한다
- Causal Mask는 행별로 “현재 Query가 어디까지 볼 수 있는가?”를 기준으로 읽는다
- 학습에서는 여러 위치의 loss를 병렬 계산할 수 있지만, 생성은 토큰을 하나씩 이어가므로 순차적이다
- 원래 Decoder와 Decoder-only는 Cross-Attention 유무로 구분할 수 있다.

