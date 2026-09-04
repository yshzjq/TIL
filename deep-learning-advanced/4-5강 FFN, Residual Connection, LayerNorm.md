---
title: 4-5강 FFN, Residual Connection, LayerNorm(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-5강 FFN, Residual Connection, LayerNorm' 정리
---

## 1. Position-wise FFN



<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_positionwise_ffn.png
' | relative_url }}" alt="01_positionwise_ffn.png
" loading="lazy">

Position-wise FFN

FFN은 각 토큰 위치에 같은 MLP를 독립적으로 적용

```
입력 [B,L,D]
-> Linear(D,D_ff)
-> Activation
-> Linear(D_ff,D)
-> 출력 [B,L,D]
```

### Attention과 FFN 비교

| 구성요소 | 주된 역할 |
| --- | --- |
| Self-Attention | 토큰 사이의 정보를 섞습니다. |
| Position-wise FFN | 각 토큰 벡터 내부의 feature를 변환한다. |

### 왜 중간 차원을 키울까요?

`D_ff`를 `D`보다 크게 두면 토큰 표현을 더 넓은 공간으로 확장한 뒤 비선형 변환을 적용할 수 있다.

```
예: D=768 -> D_ff=3072 -> D=768
```

정확한 배수와 activation은 모델마다 다르다 

최근 모델은 GELU, SwiGLU 등 다양한 FFN 구조를 사용할 수 있다.

## 2. Residual Connection

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_residual_route.png
' | relative_url }}" alt="02_residual_route.png
" loading="lazy">

Residual Connection

Residual Connection은 Sublayer 출력에 원래 입력을 더한다

```
Y = X + Sublayer(X)
```

### 왜 사용할까

- 원래 입력 정보를 우회 경로로 전달한다
- Sublayer는 전체 표현을 처음부터 다시 만들기보다 입력에 필요한 변화량을 더할 수 있다
- 깊은 모델에서 gradient가 전달되는 직접 경로를 제공한다

### Shape 조건

```
X:           [B,L,D]
Sublayer(X): [B,L,D]
```

두 Tensor의 shape이 같아야 위치별 원소를 더할 수 있다.

## 3. LayerNorm

Transformer에서 `LayerNorm(D)`는 일반적으로 각 토큰 벡터의 마지막 hidden feature 축을 기준으로 평균과 분산을 계산한다

### BatchNorm과의 직관적 차이

- BatchNorm: 여러 샘플의 batch 통계를 활용하는 경우가 많다.
- LayerNorm: 각 샘플·각 토큰의 feature 축 안에서 정규화한다


LayerNorm은 정규화 뒤에 학습 가능한 scale과 bias를 적용할 수 있다. 

따라서 단순히 “출력을 항상 평균 0, 분산 1로 고정한다”고만 이해하면 부족하다

## 4. Pre-Norm과 Post-Norm

### Post-Norm

```
Y = LayerNorm(X + Sublayer(X))
```

### Pre-Norm

```
Y = X + Sublayer(LayerNorm(X))
```


> 둘 중 하나가 모든 상황에서 항상 정답인 것은 아니다. 실제 모델의 config와 구현에서 `norm_first`, `pre_norm`, `post_norm` 설정을 확인한다

## 선택 · 5. 연습 문제: Position-wise FFN 구현

```python
from __future__ import annotations

import torch
from torch import Tensor, nn

class PositionWiseFFN(nn.Module):
    """각 토큰 위치에 같은 두 층 MLP를 적용합니다."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # 첫 Linear는 마지막 hidden dimension을 D에서 D_ff로 확장합니다.
        self.linear_in = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
        )

        # GELU는 Transformer 계열에서 널리 사용하는 비선형 activation 중 하나입니다.
        self.activation = nn.GELU()

        # 학습 시 일부 activation을 확률적으로 0으로 만드는 regularization입니다.
        self.dropout = nn.Dropout(dropout)

        # Residual 덧셈을 위해 마지막 차원을 다시 D로 줄입니다.
        self.linear_out = nn.Linear(
            in_features=d_ff,
            out_features=d_model,
        )

    def forward(self, x: Tensor) -> Tensor:
        """입력 [B,L,D]를 같은 shape의 새 토큰 표현으로 변환합니다."""

        # [B,L,D] -> [B,L,D_ff]
        hidden = self.linear_in(x)

        # Shape은 유지되고 값에 비선형 변환이 적용됩니다.
        hidden = self.activation(hidden)

        # Dropout은 학습 모드에서만 확률적으로 값을 0으로 만듭니다.
        hidden = self.dropout(hidden)

        # [B,L,D_ff] -> [B,L,D]
        output = self.linear_out(hidden)

        # Residual과 더할 수 있는지 shape을 검증합니다.
        assert output.shape == x.shape

        return output

# 실행 예시
BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
D_MODEL = 16
D_FF = 64

x = torch.randn(
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
)

ffn = PositionWiseFFN(
    d_model=D_MODEL,
    d_ff=D_FF,
    dropout=0.0,
)

ffn_output = ffn(x)

print("input:", x.shape)
print("ffn_output:", ffn_output.shape)

assert ffn_output.shape == x.shape
```

## 선택 · 6. 연습 문제: Residual과 LayerNorm 연결

아래는 Pre-Norm 방식으로 FFN Sublayer를 연결한 간단한 모듈입니다.

```python
class PreNormFFNBlock(nn.Module):
    """LayerNorm -> FFN -> Residual 덧셈 흐름을 관찰합니다."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ) -> None:
        super().__init__()

        # 각 토큰의 마지막 hidden dimension D를 정규화합니다.
        self.norm = nn.LayerNorm(d_model)

        # 앞에서 구현한 Position-wise FFN을 사용합니다.
        self.ffn = PositionWiseFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=0.0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """입력 [B,L,D]를 같은 shape의 새 표현으로 변환합니다."""

        # Pre-Norm: Sublayer 전에 입력을 정규화합니다.
        normalized = self.norm(x)

        # 각 토큰 위치에 FFN을 적용합니다.
        transformed = self.ffn(normalized)

        # 원래 입력 X에 FFN의 변화량을 더합니다.
        # 두 Tensor가 [B,L,D]로 같아야 합니다.
        output = x + transformed

        assert output.shape == x.shape
        return output

block = PreNormFFNBlock(
    d_model=D_MODEL,
    d_ff=D_FF,
)

block_output = block(x)

print("block_output:", block_output.shape)
assert block_output.shape == x.shape
```

## 참고 · 7. LayerNorm 통계 관찰

```python
# 첫 번째 샘플의 첫 번째 토큰 벡터를 확인합니다.
token_vector = x[0, 0, :]

# LayerNorm을 적용합니다.
layer_norm = nn.LayerNorm(D_MODEL)
normalized_vector = layer_norm(token_vector)

print("적용 전 평균:", token_vector.mean().item())
print("적용 전 분산:", token_vector.var(unbiased=False).item())
print("적용 후 평균:", normalized_vector.mean().item())
print("적용 후 분산:", normalized_vector.var(unbiased=False).item())
```

초기 상태의 LayerNorm scale=1, bias=0에서는 평균이 0, 분산이 1에 가까워진다. 

학습이 진행되면 affine 파라미터에 따라 최종 출력 통계는 달라질 수 있다.


## 참고 · 9. 이해도 점검

1. Attention과 FFN의 역할 차이를 설명해보시오.
2. FFN에서 `D -> D_ff -> D`로 돌아오는 이유는 무엇인가
3. Residual 덧셈에 필요한 shape 조건은 무엇인가
4. `LayerNorm(D)`는 일반적으로 어느 축을 정규화하나
5. Pre-Norm과 Post-Norm의 차이는 무엇인가

### 정답 확인

1. Attention은 토큰 사이 정보를 섞고 FFN은 각 토큰 벡터를 독립적으로 가공한다
2. 표현력을 확장하면서도 Residual 덧셈과 다음 Block 연결을 위해 마지막 차원을 D로 복원한다
3. `X`와 `Sublayer(X)`의 shape이 서로 같거나 덧셈 가능한 형태여야 한다.
4. 각 토큰의 마지막 hidden feature 축 D입니다.
5. LayerNorm이 Sublayer 전에 적용되는지, Residual 덧셈 뒤에 적용되는지의 차이다.

## [이번 강의 요약]

- 선택 학습에서는 Decoder-only LLM에서 사용하는 Pre-Norm·RMSNorm과 SwiGLU 계열 FFN을 살펴보고 실제 구성은 Config에서 확인한다
- FFN은 각 토큰 위치에 같은 MLP를 적용하며 `D -> D_ff -> D`로 변환한다
- Residual Connection은 원래 입력 경로를 남기고 Sublayer의 변화량을 더한다
- LayerNorm은 각 토큰의 hidden feature 축을 정규화한다
- Pre-Norm과 Post-Norm은 LayerNorm의 적용 순서가 다르다
- Attention, FFN, Residual, LayerNorm이 함께 있어야 Transformer Block의 전체 역할을 이해할 수 있다.