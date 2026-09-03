---
title: 3-4강 Tensor로 Self-Attention 미니 구현
date: 2026-09-03
updated: 2026-09-03
description: KANT 강의 '3-4강 Tensor로 Self-Attention 미니 구현' 정리
---

## 1. 교육용 구현에서 단순화하는 것

실제 Transformer는 입력 x에 서로 다른 선형 projection을 적용해 Q, K, V를 만든다



```
# projection을 생략하고 입력 X를 세 역할에 공통으로 사용
Q = x
K = x
V = x
```

이 단순화는 “모든 토큰이 서로 비교되고, weight로 정보를 섞는다”는 핵심을 확인

## 2. 코드의 다섯 단계

교육용 Self-Attention 코드 흐름

1. 입력 `x`를 Q, K, V로 사용
2. `x @ xᵀ`로 토큰×토큰 score를 만든다
3. `√D`로 나눠 scaling
4. Softmax로 Attention weight를 만든다.
5. `weight @ x`로 Context를 만든다

## 3. 입력·출력 Shape 계약

함수 구현 전에 입력과 출력의 shape을 먼저 정한다

Self-Attention 함수 계약

| 항목 | Shape | 의미 |
| --- | --- | --- |
| 입력 `x` | `[B,L,D]` | 토큰의 현재 표현 |
| 선택 입력 `valid_key_mask` | `[B,L]` | 실제 토큰인지 여부 |
| 출력 `context` | `[B,L,D]` | 문맥이 반영된 새 표현 |
| 출력 `weights` | `[B,L,L]` | Query별 Key 참고 비율 |

valid_key_mask [B,L] = 각 문장의 각 토큰 위치가 실제 토큰인지(PAD가 아닌지)를 나타내는 Boolean Tensor

## 4. 연습 문제: Self-Attention 함수 구현

```python
from __future__ import annotations

import math
import torch
from torch import Tensor

def simple_self_attention(
    x: Tensor,
    valid_key_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """학습용 Self-Attention을 계산합니다.

    이번 구현은 흐름을 단순하게 보기 위해 Q=K=V=x로 둡니다.

    Args:
        x:
            토큰 표현 Tensor입니다.
            Shape은 [B, L, D]입니다.
            B는 배치 크기, L은 토큰 길이, D는 표현 차원입니다.

        valid_key_mask:
            실제로 참고할 수 있는 Key 위치를 나타냅니다.
            Shape은 [B, L]이고 dtype은 bool입니다.
            True는 실제 토큰, False는 padding처럼 제외할 위치입니다.

    Returns:
        context:
            문맥이 반영된 새 토큰 표현입니다.
            Shape은 [B, L, D]입니다.

        weights:
            Query별 Key 참고 비율입니다.
            Shape은 [B, L, L]입니다.
    """

    # 입력이 [B, L, D] 구조인지 먼저 확인합니다.
    if x.ndim != 3:
        raise ValueError(
            f"x must have shape [B, L, D], but got{tuple(x.shape)}"
        )

    # 마지막 차원이 토큰 표현 차원 D입니다.
    # Scaling에 사용하기 위해 Python 정수로 꺼냅니다.
    d_model = x.size(-1)

    # 학습용 단순화: Q=K=V=x로 사용합니다.
    q = x
    k = x
    v = x

    # K의 마지막 두 축을 바꿉니다.
    # [B, L, D] -> [B, D, L]
    k_t = k.transpose(-2, -1)

    # 모든 Query와 모든 Key의 관련도 score를 계산합니다.
    # [B, L, D] @ [B, D, L] -> [B, L, L]
    scores = torch.matmul(q, k_t)

    # Dot-product score가 너무 커지는 것을 완화합니다.
    # 나눗셈은 값의 크기만 바꾸며 shape은 유지합니다.
    scaled_scores = scores / math.sqrt(d_model)

    if valid_key_mask is not None:
        # Mask shape과 dtype을 명확하게 확인합니다.
        if valid_key_mask.ndim != 2:
            raise ValueError(
                "valid_key_mask must have shape [B, L]"
            )
        if valid_key_mask.dtype != torch.bool:
            raise TypeError(
                "valid_key_mask must be a boolean Tensor"
            )
        if valid_key_mask.shape != x.shape[:2]:
            raise ValueError(
                "valid_key_mask must match x's [B, L] dimensions"
            )

        # 모든 Key가 False이면 한 행 전체가 -inf가 되어 Softmax가 NaN이 될 수 있습니다.
        # 각 샘플에 최소 하나의 실제 토큰이 있는지 확인합니다.
        if not torch.all(valid_key_mask.any(dim=-1)):
            raise ValueError(
                "Each sample must contain at least one valid key token"
            )

        # [B, L] -> [B, 1, L]로 축 하나를 추가합니다.
        # 이 mask는 Query 축 전체에 broadcasting됩니다.
        key_mask = valid_key_mask.unsqueeze(1)

        # False인 Key 위치의 score를 -inf로 바꿉니다.
        # Softmax 이후 이 위치의 weight는 0이 됩니다.
        scaled_scores = scaled_scores.masked_fill(
            ~key_mask,
            float("-inf"),
        )

    # 마지막 축은 각 Query가 비교하는 모든 Key 위치입니다.
    # 따라서 dim=-1에 Softmax를 적용합니다.
    weights = torch.softmax(scaled_scores, dim=-1)

    # Attention weight만큼 Value를 섞습니다.
    # [B, L, L] @ [B, L, D] -> [B, L, D]
    context = torch.matmul(weights, v)

    # 함수의 출력 계약을 assert로 검증합니다.
    batch_size, seq_len, _ = x.shape
    assert weights.shape == (batch_size, seq_len, seq_len)
    assert context.shape == x.shape

    # 각 Query 행의 weight 합이 약 1인지 확인합니다.
    expected_ones = torch.ones(
        batch_size,
        seq_len,
        device=x.device,
        dtype=x.dtype,
    )
    assert torch.allclose(
        weights.sum(dim=-1),
        expected_ones,
        atol=1e-6,
    )

    return context, weights
```

### 4-1. 함수 실행하기

```python
# 결과를 다시 실행해도 같은 입력이 만들어지도록 seed를 고정합니다.
torch.manual_seed(42)

# 문장 한 개(B=1), 토큰 네 개(L=4), 표현 차원 세 개(D=3)의 입력입니다.
x = torch.randn(1, 4, 3, dtype=torch.float32)

context, weights = simple_self_attention(x)

print("x shape:", x.shape)
print("weights shape:", weights.shape)
print("context shape:", context.shape)
print("weight row sums:\n", weights.sum(dim=-1))

assert x.shape == (1, 4, 3)
assert weights.shape == (1, 4, 4)
assert context.shape == (1, 4, 3)
```

### 관찰할 내용

- Weight는 토큰×토큰 관계표이므로 `[1,4,4]`다.
- Context는 토큰마다 새 표현 하나가 나오므로 `[1,4,3]`다.
- 각 Query 행의 weight 합은 1다.

## 5. Mask 맛보기

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_mask.png
' | relative_url }}" alt="02_mask.png
" loading="lazy">

Mask 적용 전후

### 5-1. Mask와 함께 실행하기

```python
# 마지막 위치가 padding이라고 가정합니다.
# True는 실제 토큰, False는 제외할 위치입니다.
valid_key_mask = torch.tensor(
    [[True, True, True, False]],
    dtype=torch.bool,
)

masked_context, masked_weights = simple_self_attention(
    x=x,
    valid_key_mask=valid_key_mask,
)

print("masked weights:\n", masked_weights)

# 마지막 Key 열은 padding 위치이므로 모든 Query에서 weight가 0이어야 합니다.
print("padding column:\n", masked_weights[..., -1])

assert torch.allclose(
    masked_weights[..., -1],
    torch.zeros_like(masked_weights[..., -1]),
    atol=1e-6,
)
```

Mask의 `True/False` 의미는 API마다 다를 수 있다.

이 교육용 함수에서는 `True=사용 가능`, `False=제외`로 직접 정의했습니다. 

실제 API를 사용할 때는 공식 문서를 확인해야한다

## 6. 디버깅 점검 순서

Self-Attention 디버깅 체크리스트

1. 입력이 `[B,L,D]`인가
2. K의 마지막 두 축을 바꾸었나
3. Softmax가 Key 축인 `dim=-1`에 적용이 되었나
4. Mask가 `[B,1,L]`로 확장되어 Key 축에 적용됐나
5. Weight 합이 1이고 Context shape이 입력과 같은가

## 7. 공식 PyTorch API와의 연결

PyTorch는 최적화된 `torch.nn.functional.scaled_dot_product_attention()` API를 제공

실제 모델 구현에서는 직접 계산한 함수보다 이 API를 사용하는 경우가 많다.


## 8. 이해도 점검

1. `x [B,L,D] @ xᵀ [B,D,L]`의 결과 shape은 무엇인가요?
2. Softmax는 어느 축에 적용하나
3. Padding Key의 weight를 0으로 만들기 위해 mask는 언제 적용하나
4. Context output shape이 입력과 같은 이유는 무엇인가

### 정답 확인

1. `[B,L,L]`입니다.
2. Key 후보가 있는 마지막 축 `dim=-1`
3. Softmax 전에 해당 score를 매우 작은 값 또는 `inf`로 바꾼다
4. 각 Query 위치마다 Value 차원 D의 새 표현 하나를 만들기 때문이다

## 9. 실제 구현으로 넘어갈 때 확인할 것

실제 Self-Attention에서는 서로 다른 projection을 사용

```python
q = q_proj(x)
k = k_proj(x)
v = v_proj(x)
```

projection 파라미터까지 gradient가 흐르는지 짧은 backward 점검을 해두면 구현 누락을 빨리 찾을 수 있다.

```python
loss = context.pow(2).mean()
loss.backward()

layers = {"q": q_proj, "k": k_proj, "v": v_proj}
for name, layer in layers.items():
    assert layer.weight.grad is not None, name
    assert torch.isfinite(layer.weight.grad).all()
```

PyTorch의 `scaled_dot_product_attention()`을 사용할 때도 의미는 같다. 

다만 bool mask의 의미, additive mask 값, `is_causal`, dropout 동작을 API 기준으로 다시 확인해야 한다. 

수동 구현과 최적화 API의 출력을 비교할 때는 평가 모드와 dropout 0을 먼저 맞춘다

## 요약
.
- Score, scaling, softmax, Value 가중합의 순서로 계산
- Weight shape은 `[B,L,L]`, Context shape은 `[B,L,D]`입니다.
- Padding 위치는 Softmax 전에 mask해 weight가 0이 되게 합니다.
- 오류가 발생하면 shape, transpose, Softmax 축, mask 순서로 점검한다