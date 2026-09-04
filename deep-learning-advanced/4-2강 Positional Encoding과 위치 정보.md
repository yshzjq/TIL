---
title: 4-2강 Positional Encoding과 위치 정보(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-2강 Positional Encoding과 위치 정보' 정리
---

## 1. 순서가 왜 중요한가

같은 단어, 다른 순서

다음 두 문장은 비슷한 토큰을 사용하지만 역할이 반대다.

```
철수가 영희를 도왔다.
영희가 철수를 도왔다.
```

모델이 토큰 내용만 알고 위치를 전혀 모른다면, 누가 행동의 주체인지 구분하기 어려워질 수 있다.

## 2. Self-Attention만으로 부족한 정보

Self-Attention은 각 토큰이 다른 토큰을 얼마나 참고할지 계산한다.

하지만 위치 정보가 입력에 포함되지 않으면, 연산 자체는 **“첫 번째”, “두 번째”, “바로 앞”, “두 칸 뒤”** 같은 순서 좌표를 별도로 알지 못한다

조금 더 직관적으로 말하면 다음 두 정보가 모두 필요하다

- Token Embedding: **무슨 토큰인가?**
- Position 정보: **몇 번째 위치인가?**

위치 정보는 단어 의미를 대신하지 않습니다. 토큰 의미에 순서 좌표를 추가한다.

## 3. Token Embedding과 Position 정보 결합


<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_embedding_plus_position.png
' | relative_url }}" alt="02_embedding_plus_position.png
" loading="lazy">


Embedding과 Position 결합

두 벡터를 더하려면 같은 shape이어야 합니다.

```
Token Embedding: [B,L,D]
Position:        [1,L,D] 또는 [B,L,D]
결합 결과:       [B,L,D]
```

Position Tensor의 batch 크기가 1이면 broadcasting을 이용해 batch 전체에 같은 위치 패턴을 적용할 수 있다.

```python
# token_hidden: [B, L, D]
# position_hidden: [1, L, D]
transformer_input = token_hidden + position_hidden

# 위치 정보를 더해도 축 구조는 바뀌지 않습니다.
assert transformer_input.shape == token_hidden.shape
```

## 4. Learned Position Embedding

Learned Position Embedding은 "이 토큰이 문장에서 몇 번째 위치에 있는지"를 벡터로 바꿔주는 방식



Learned 방식은 위치 0, 1, 2, … 각각에 학습 가능한 D차원 벡터를 배정한다

각 위치 번호마다 Token Embedding과 같은 크기의 벡터 하나를 준비하고, 그 벡터를 학습시킨다.

```
위치 0 -> 학습되는 벡터
위치 1 -> 학습되는 벡터
위치 2 -> 학습되는 벡터
```

Token Embedding이 Token ID를 벡터로 바꾸는 것과 매우 비슷하다

| Embedding 종류 | 입력 ID | 출력 |
| --- | --- | --- |
| Token Embedding | Vocabulary 안의 Token ID | Token 의미 벡터 |
| Position Embedding | 문장 안의 Position ID | 위치 벡터 |

장점은 데이터와 학습 목적에 맞는 위치 표현을 직접 학습할 수 있다는 있다. 

반면 모델 config의 최대 위치 수에 영향을 받습니다.

Learned Position Embedding은 문장 속 위치 번호(0, 1, 2, …)를 학습 가능한 벡터로 바꿔서 Token Embedding에 더해주는 방식

이를 통해 Transformer가 토큰의 내용뿐 아니라 순서 정보도 알 수 있게 한다


Token Embedding은 토큰 자체를 벡터로 바꾸고, Learned Position Embedding은 그 토큰이 있는 위치를 벡터로 바꾼다. 그리고 두 벡터를 더해서 Transformer에 넣는다.


## 5. Sinusoidal Positional Encoding의 직관


Learned Position Embedding과 Sinusoidal Positional Encoding은 
<br>
둘 다 "이 토큰이 몇 번째 위치에 있는지"를 D차원 벡터로 만들어 Token Embedding에 더한다


Learned와 Sinusoidal 비교
```
나는   학교에   간다
 ↓       ↓       ↓
위치 0   위치 1   위치 2
```
```
"학교에"

Token Embedding
[0.4, 0.7, -0.2, ...]

+

Position 정보
[0.1, -0.3, 0.5, ...]

=

Transformer에 들어갈 벡터
[0.5, 0.4, 0.3, ...]
```
두 종류의 벡터가 생긴다

여기까지는 똑같다

Learned Position Embedding은 모델이 위치 벡터 자체를 학습한다.

**Sinusoidal Positional Encoding**

```
위치 0 → sin/cos 공식으로 계산
위치 1 → sin/cos 공식으로 계산
위치 2 → sin/cos 공식으로 계산
```


"위치가 3번이면 이 공식으로 계산해서 이 벡터를 사용해."

라고 미리 규칙을 정해놓는다



Learned
→ 위치 벡터가 학습 파라미터

Sinusoidal
→ 위치 벡터를 수학 공식으로 계산

Sinusoidal 방식은 학습 파라미터를 저장하지 않고 위치와 차원에 따라 sin/cos 값을 계산한다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\04_sinusoidal_intuition.png
' | relative_url }}" alt="04_sinusoidal_intuition.png
" loading="lazy">

왜 파도가 여러 개인가?

파란색, 초록색, 보라색 파동은 각각을 Position 벡터의 한 차원이라고 생각


D=4라면 Position Encoding
```
[차원0, 차원1, 차원2, 차원3]
```

숫자 4개가 필요하다

Sinusoidal에서는 각각의 차원이 서로 다른 주기의 sin/cos 함수를 사용

단순화 하면

```
차원 0 → 빠르게 움직이는 파도
차원 1 → 빠르게 움직이는 다른 파도
차원 2 → 천천히 움직이는 파도
차원 3 → 더 천천히 움직이는 파도

이미지에서는 
빠른 주기
중간 주기
느린 주기
```

위 처럼 표현

위치가 바뀌면 파도의 갑도 바뀐다

위치 0의 위치 벡터는 대충
```
[0.2, 0.4, 0.1, ...]
```
위치 1로 이동
```
위치 1

빠른 파도 → 0.8
중간 파도 → 0.6
느린 파도 → 0.2
```

위치 2

```

[0.4, 0.9, 0.3, ...]
```

결과적으로
```
위치 0 → [서로 다른 숫자들]
위치 1 → [서로 다른 숫자들]
위치 2 → [서로 다른 숫자들]
```
만들어진다

Sinusoidal Encoding의 목적은 각 위치마다 서로 다른 D차원 숫자 패턴을 만들어주는 것



<img src="{{ '/assets/images/uploads\deep-learning-advanced\d2l_positional_heatmap.png
' | relative_url }}" alt="d2l_positional_heatmap.png
" loading="lazy">
```
세로(Row) = position
가로(Column) = encoding dimension
```

예를 들면
```
세로 0번째 줄
→ 위치 0의 Position Encoding 벡터

세로 1번째 줄
→ 위치 1의 Position Encoding 벡터

세로 2번째 줄
→ 위치 2의 Position Encoding 벡터
```


D가 32라면 한 줄에 숫자가 32개 있다는 뜻.

```
위치 0
[값0, 값1, 값2, ... 값31]

위치 1
[값0, 값1, 값2, ... 값31]

위치 2
[값0, 값1, 값2, ... 값31]
```



색깔은 무엇인가

오른쪽에 보면:
```
1.0
0.5
0
-0.5
-1.0

```

sin/cos 값이 기본적으로 -1 ~ 1 사이

그래서 각각의 칸은 실제로 숫자 하나

```
위치 10, 차원 3 → 0.72
위치 10, 차원 4 → -0.31
위치 10, 차원 5 → 0.95
```

같은 숫자를 색으로 표현

큰 숫자 행렬을 색칠



각 차원이 위치에 따라 어떻게 변하는가

```
차원 0 → 빠른 파도
차원 1 → 또 다른 파도
차원 2 → 느린 파도
```
그 파도들의 값을 각 위치에서 하나씩 뽑는다

예를 들렴
```
차원 0 파도의 값 → 0.8
차원 1 파도의 값 → -0.3
차원 2 파도의 값 → 0.4
차원 3 파도의 값 → 0.9
```
전부 모으면
```
위치 5의 Position Encoding

[0.8, -0.3, 0.4, 0.9, ...]
```

이 벡터 하나가 가로 한 줄

즉

```

두 번째 그림
여러 sin/cos 파도
        ↓
특정 위치에서 각 파도의 값 가져오기
        ↓
[0.8, -0.3, 0.4, 0.9, ...]
        ↓
세 번째 그림의 한 Row

```

그리고 Token Embedding과 더한다

```
나는   학교에   간다
 ↓       ↓       ↓
위치0   위치1   위치2
```

학교에의 Token Embedding

```
[0.5, 0.2, -0.1, 0.8]
```

위치 1의 Sinusoidal Encoding이
```
[0.84, 0.54, 0.01, 0.99]
```



```
Token Embedding
[0.50, 0.20, -0.10, 0.80]

+

Position Encoding
[0.84, 0.54, 0.01, 0.99]

=

Transformer 입력
[1.34, 0.74, -0.09, 1.79]

```

그래서 Transformer 입장에서는 이 벡터에

```
"이게 어떤 토큰인가"
+
"몇 번째 위치인가"

```

두 정보가 함께 들어가게 된다

## 선택 · 6. 연습 문제: Learned Position Embedding 구현

```python
import torch
from torch import nn

# 실행할 때마다 같은 초기값을 만들기 위해 seed를 고정합니다.
torch.manual_seed(42)

# 교육용 설정값입니다.
BATCH_SIZE = 2
SEQUENCE_LENGTH = 5
VOCAB_SIZE = 100
MAX_POSITION = 32
D_MODEL = 16

# 두 문장의 Token ID를 가정합니다.
# Shape: [B, L]
input_ids = torch.tensor(
    [
        [10, 20, 30, 40, 50],
        [11, 21, 31, 41, 51],
    ],
    dtype=torch.long,
)

# Token ID를 D차원 의미 벡터로 바꾸는 학습 가능한 Embedding입니다.
token_embedding = nn.Embedding(
    num_embeddings=VOCAB_SIZE,
    embedding_dim=D_MODEL,
)

# Position ID를 D차원 위치 벡터로 바꾸는 학습 가능한 Embedding입니다.
position_embedding = nn.Embedding(
    num_embeddings=MAX_POSITION,
    embedding_dim=D_MODEL,
)

# 0, 1, 2, 3, 4의 위치 ID를 만듭니다.
# Shape: [L]
position_ids = torch.arange(
    SEQUENCE_LENGTH,
    dtype=torch.long,
)

# Batch의 각 문장에 같은 위치 ID를 사용하도록 batch 축을 추가합니다.
# unsqueeze(0): [L] -> [1,L]
# expand(B,-1): [1,L] -> [B,L]
position_ids = position_ids.unsqueeze(0).expand(
    BATCH_SIZE,
    -1,
)

# Token ID를 의미 벡터로 변환합니다.
# Shape: [B,L] -> [B,L,D]
token_hidden = token_embedding(input_ids)

# Position ID를 위치 벡터로 변환합니다.
# Shape: [B,L] -> [B,L,D]
position_hidden = position_embedding(position_ids)

# 의미와 위치 정보를 원소별로 더합니다.
# 두 Tensor의 shape이 같아야 합니다.
transformer_input = token_hidden + position_hidden

print("input_ids:", input_ids.shape)
print("position_ids:", position_ids.shape)
print("token_hidden:", token_hidden.shape)
print("position_hidden:", position_hidden.shape)
print("transformer_input:", transformer_input.shape)

assert token_hidden.shape == (
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    D_MODEL,
)
assert position_hidden.shape == token_hidden.shape
assert transformer_input.shape == token_hidden.shape
```

### 예상 결과

```
input_ids: torch.Size([2, 5])
position_ids: torch.Size([2, 5])
token_hidden: torch.Size([2, 5, 16])
position_hidden: torch.Size([2, 5, 16])
transformer_input: torch.Size([2, 5, 16])
```

## 7. 선택 학습: Sinusoidal Encoding 코드 읽기

```python
import math
import torch

def make_sinusoidal_encoding(
    max_length: int,
    d_model: int,
) -> torch.Tensor:
    """[1, max_length, d_model] 형태의 fixed 위치 정보를 만듭니다."""

    # 이 교육용 구현은 sin/cos를 두 차원씩 짝지으므로 짝수 D만 사용합니다.
    if d_model % 2 != 0:
        raise ValueError("d_model must be even")

    # 각 토큰 위치 0, 1, ..., max_length-1을 세로 벡터로 만듭니다.
    # Shape: [L,1]
    position = torch.arange(
        max_length,
        dtype=torch.float32,
    ).unsqueeze(1)

    # 짝수 hidden dimension에 사용할 주파수 계수를 만듭니다.
    # Shape: [D/2]
    frequency = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )

    # 모든 위치와 모든 주파수 조합을 만듭니다.
    # [L,1] * [D/2]가 broadcasting되어 [L,D/2]가 됩니다.
    angles = position * frequency

    # 최종 위치 Encoding을 0으로 초기화합니다.
    # Shape: [L,D]
    encoding = torch.zeros(
        max_length,
        d_model,
        dtype=torch.float32,
    )

    # 짝수 hidden dimension에는 sin을 넣습니다.
    encoding[:, 0::2] = torch.sin(angles)

    # 홀수 hidden dimension에는 cos를 넣습니다.
    encoding[:, 1::2] = torch.cos(angles)

    # Batch 축으로 broadcasting하기 위해 맨 앞에 크기 1의 축을 추가합니다.
    # Shape: [1,L,D]
    return encoding.unsqueeze(0)

position_values = make_sinusoidal_encoding(
    max_length=5,
    d_model=16,
)

print(position_values.shape)
assert position_values.shape == (1, 5, 16)
```

## 참고 · 9. 이해도 점검

1. Token Embedding과 Position 정보는 각각 어떤 질문에 답하나
2. 두 정보를 더하려면 어떤 shape 조건이 필요한가
3. Learned Position Embedding은 무엇이 학습되나
4. Sinusoidal 방식에서 여러 주기를 사용하는 이유를 직관적으로 설명
5. 위치 정보를 더한 뒤 sequence length 축은 어떻게 되나

### 정답 확인

1. Token Embedding은 “무슨 토큰인가?”, Position은 “몇 번째 위치인가?”를 나타낸다
2. 마지막 hidden dimension을 포함해 서로 더할 수 있는 동일하거나 broadcast 가능한 shape이어야 한다
3. 각 Position ID에 대응하는 D차원 벡터가 학습된다
4. 하나의 주기보다 여러 주기를 조합하면 위치별로 더 다양한 패턴을 만들 수 있다.
5. 그대로 `L`을 유지합니다.

## 10. RoPE는 위치를 Q와 K에 반영합니다 (선택 학습)