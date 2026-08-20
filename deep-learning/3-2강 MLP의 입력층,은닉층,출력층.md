---
title: 3-2강 MLP의 입력층,은닉층,출력층
date: 2026-08-20
updated: 2026-08-20
description: KANT 강의 '3-2강 MLP의 입력층,은닉층,출력층' 정리
---

## 1. 이번에 우리가 배울 것

퍼셉트론을 여러 개 묶어 **층(layer)** 을 만들고, 그 층을 여러 개 쌓아 MLP를 만듭니다.

<img src="{{ '/assets/images/uploads/deep-learning/MLP 전체 구조.png' | relative_url }}" alt="MLP 전체 구조.png" loading="lazy">

각 원은 뉴런, 선은 가중치 연결을 의미

## 2. MLP란 무엇인가요?

MLP는 **Multi-Layer Perceptron**의 줄임말입니다. 한국어로는 다층 퍼셉트론이라고 부른다

이름 그대로 여러 층을 가진 퍼셉트론 구조

```
Perceptron 하나       -> 선형 경계 하나
Perceptron 여러 개    -> 여러 방향의 신호 계산
Layer 여러 개          -> 표현을 단계적으로 변환
MLP                  -> 입력을 여러 단계로 변환해 출력 생성
```

MLP의 핵심은 입력 데이터를 바로 정답으로 바꾸는 것이 아니라, 중간에 **은닉층(hidden layer)** 을 통해 더 유용한 표현으로 바꾸는 것

은닉층은 원본 feature를 모델이 이해하기 좋은 중간 feature로 바꾸는 공간이라고 생각하면 좋다. 사람으로 비유하면 원본 정보를 바로 판단하지 않고, 중간 메모를 만들고 그 메모를 바탕으로 판단하는 것과 비슷하다

## 3. 입력층, 은닉층, 출력층 역할

MLP는 보통 세 종류의 층으로 설명.

| 층 | 역할 | 예시 shape |
| --- | --- | --- |
| 입력층 | 원본 데이터를 받습니다. | `(batch_size, input_dim)` |
| 은닉층 | 입력을 새로운 표현으로 변환합니다. | `(batch_size, hidden_dim)` |
| 출력층 | 최종 예측 점수 또는 logits를 만듭니다. | `(batch_size, num_classes)` |

예를 들어 학생 4명의 데이터를 가지고 3개 feature로 2개 클래스를 분류한다고 가정

```
입력 X shape       : (4, 3)
은닉층 출력 shape  : (4, 5)
최종 출력 shape    : (4, 2)
```

여기서 `4`는 batch size입니다. MLP를 통과하더라도 batch size는 유지된다

MLP에서 각 층이 바꾸는 것은 보통 feature 차원입니다. batch 차원은 모델이 샘플 개수를 유지하기 위해 그대로 둔다

## 4. MLP에서 shape 흐름 읽기

```
input_dim = 3
hidden_dim = 5
num_classes = 2
```

입력 Tensor가 `(4, 3)`이면 shape 흐름

```
X                         : (4, 3)
Linear(3, 5) 통과 후      : (4, 5)
ReLU 통과 후              : (4, 5)
Linear(5, 2) 통과 후      : (4, 2)
```

이 흐름에서 각 값의 의미

| 값 | 의미 |
| --- | --- |
| 4 | batch size, 샘플 개수 |
| 3 | 입력 feature 수 |
| 5 | 은닉층 feature 수, hidden size |
| 2 | 출력 class 수 |

## 5. PyTorch로 2층 MLP 만들기

PyTorch에서는 `nn.Linear`를 이용해 선형층을 만들 수 있다.

```python
import torch
import torch.nn as nn

# 재현 가능한 예시를 위해 seed를 고정합니다.
torch.manual_seed(42)

# 샘플 4개, feature 3개인 입력입니다.
X = torch.randn(4, 3)

# MLP 구조를 정의합니다.
# Linear(3, 5): 입력 feature 3개를 hidden feature 5개로 변환합니다.
# ReLU(): 은닉층 결과에 비선형성을 추가합니다. 자세한 내용은 4장에서 다룹니다.
# Linear(5, 2): hidden feature 5개를 class 2개에 대한 점수로 변환합니다.
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 모델에 입력을 넣으면 출력 logits가 나옵니다.
logits = model(X)

print("X shape     :", X.shape)
print("logits shape:", logits.shape)
print(logits)
```

예상 출력

```
X shape     : torch.Size([4, 3])
logits shape: torch.Size([4, 2])
```

여기서 `logits`의 shape가 `(4, 2)`인 이유

```
4개 샘플 각각에 대해
2개 클래스의 점수를 출력
```

## 6. 중간 출력 shape 직접 확인하기

`nn.Sequential`은 편리하지만, 초반에는 각 층을 따로 실행해 shape를 확인하는 것이 좋다.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.randn(4, 3)

# 각 층을 변수로 분리해서 정의합니다.
fc1 = nn.Linear(3, 5)
relu = nn.ReLU()
fc2 = nn.Linear(5, 2)

# 1번째 Linear를 통과합니다.
hidden_linear = fc1(X)

# ReLU를 통과합니다.
hidden_activated = relu(hidden_linear)

# 마지막 Linear를 통과합니다.
logits = fc2(hidden_activated)

print("X               :", X.shape)
print("hidden_linear   :", hidden_linear.shape)
print("hidden_activated:", hidden_activated.shape)
print("logits          :", logits.shape)
```

출력

```
X               : torch.Size([4, 3])
hidden_linear   : torch.Size([4, 5])
hidden_activated: torch.Size([4, 5])
logits          : torch.Size([4, 2])
```

## 7. depth와 width 이해하기

MLP에서 자주 나오는 용어

| 용어 | 의미 |
| --- | --- |
| depth | 층을 얼마나 깊게 쌓았는지 |
| width | 한 은닉층에 뉴런이 얼마나 많은지 |
| hidden size | 은닉층 출력 feature 수 |

층의 수를 셀 때는 자료마다 입력층을 포함하는지, 학습 가능한 Linear 층만 세는지가 다를 수 가 있다<br>
이번 정리 내용은 학습 가능한 Linear 층 수를 기준으로 말한다.<br>
아래 모델은 Linear 층 2개, 은닉층 1개인 MLP

다음 모델은 은닉층이 1개이고 hidden size가 5, 아래 두 짧은 블록은 [앞 셀 실행]이며,  import torch.nn as nn을 사용

```python
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
```

다음 모델은 Linear 층 3개, 은닉층 2개

```python
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)
```

층을 많이 쌓거나 hidden size를 크게 하면 모델이 더 복잡한 패턴을 표현할 수 있다.<br>
하지만 그만큼 파라미터 수와 연산량이 증가하고, 데이터가 적으면 과적합이 발생할 수 있으므로 <br>무조건 크게 만드는 것이 좋은 것은 아니다.

## 8. hidden size가 parameter 수에 미치는 영향

hidden size가 커지면 parameter 수가 늘어난다

구조

```
input_dim = 3
hidden_dim = 5
num_classes = 2
```

첫 번째 Linear의 parameter 수

```
weight 수 = 3 * 5 = 15
bias 수   = 5
합계      = 20
```

두 번째 Linear의 parameter 수

```
weight 수 = 5 * 2 = 10
bias 수   = 2
합계      = 12
```

전체 parameter 수는 32개입니다.

코드로 확인

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# 모델의 학습 가능한 parameter 수를 계산합니다.
# p.numel()은 해당 parameter Tensor 안의 원소 개수를 의미합니다.
total_params = sum(p.numel() for p in model.parameters())

print("total parameters:", total_params)

# 각 parameter의 이름과 shape도 확인합니다.
for name, param in model.named_parameters():
    print(name, param.shape, "numel=", param.numel())
```

예상 출력

```
total parameters: 32
0.weight torch.Size([5, 3]) numel= 15
0.bias torch.Size([5]) numel= 5
2.weight torch.Size([2, 5]) numel= 10
2.bias torch.Size([2]) numel= 2
```

## 9. hidden size 변경 실험

hidden size를 바꾸면 parameter 수가 어떻게 달라지는지 확인

```python
import torch.nn as nn

input_dim = 3
num_classes = 2

for hidden_dim in [2, 5, 10, 20]:
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"hidden_dim={hidden_dim:2d} -> total_params={total_params}")
```

출력

```
hidden_dim= 2 -> total_params=14
hidden_dim= 5 -> total_params=32
hidden_dim=10 -> total_params=62
hidden_dim=20 -> total_params=122
```

hidden size는 모델의 표현력과 parameter 수를 함께 바꾼다<br> 
그래서 실험할 때 hidden size는 중요한 하이퍼파라미터입니다.

## 10. 모델 구조도에 shape 표시하기

아래 표처럼 모델 구조를 shape 흐름으로 정리

| 단계 | 연산 | 출력 shape | 설명 |
| --- | --- | --- | --- |
| 입력 | `X` | `(4, 3)` | 샘플 4개, feature 3개 |
| 은닉 선형층 | `Linear(3, 5)` | `(4, 5)` | feature 3개를 5개 표현으로 변환 |
| 활성화 | `ReLU()` | `(4, 5)` | shape는 유지 |
| 출력 선형층 | `Linear(5, 2)` | `(4, 2)` | class 2개 점수 출력 |

이런 표를 작성하면 모델을 구현하기 전에 shape 오류를 미리 줄일 수 있다.