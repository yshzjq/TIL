---
title: 3-4강 입출력 차원 계산과 flatten
date: 2026-08-20
updated: 2026-08-20
description: KANT 강의 '3-4강 입출력 차원 계산과 flatten' 정리
---

## 1. 이번에 배울것

MLP는 기본적으로 샘플 하나를 긴 feature 벡터로 받습니다.

```
MLP 입력 기본 형태: (batch_size, features)
```

그런데 이미지 데이터는 보통 다음 형태입니다.

```
이미지 batch 형태: (batch_size, channels, height, width)
```

따라서 이미지를 MLP에 넣으려면 채널, 높이, 너비 차원을 하나의 긴 feature 차원으로 펼쳐야 합니다. 이 과정을 **flatten**이라고 부른다

## 2. MLP가 기대하는 입력 구조

`nn.Linear`는 입력 Tensor의 마지막 차원이 `in_features`와 같아야 합니다.

가장 기본적인 MLP 입력

```
X shape = (batch_size, input_dim)
```

샘플 32개, 각 샘플의 feature가 10개

```python
import torch
import torch.nn as nn

X = torch.randn(32, 10)

model = nn.Linear(10, 3)

logits = model(X)

print("X shape     :", X.shape)
print("logits shape:", logits.shape)
```

출력

```
X shape     : torch.Size([32, 10])
logits shape: torch.Size([32, 3])
```

MLP에 넣기 전에는 항상 `X.shape[-1] == model의 첫 Linear in_features`인지 확인해야 한다

## 3. tabular 데이터의 입력 차원

표 데이터는 대부분 이미 `(batch_size, features)` 형태입니다.

고객 5명의 정보를 가지고 있다고 가정

| feature | 의미 |
| --- | --- |
| 나이 | 숫자 feature |
| 사용 개월 수 | 숫자 feature |
| 월 결제액 | 숫자 feature |
| 최근 접속 횟수 | 숫자 feature |

입력 Tensor

```python
import torch

# 고객 5명, feature 4개입니다.
X_tabular = torch.tensor([
    [25.0, 12.0, 30000.0, 10.0],
    [31.0, 24.0, 45000.0, 20.0],
    [42.0,  6.0, 20000.0,  5.0],
    [28.0, 18.0, 35000.0, 12.0],
    [36.0, 30.0, 50000.0, 25.0]
])

print(X_tabular.shape)
```

출력

```
torch.Size([5, 4])
```
이 경우 첫 번째 Linear는 `nn.Linear(4, hidden_dim)`이어야 한다.

```python
import torch.nn as nn

model = nn.Linear(4, 8)
out = model(X_tabular)

print(out.shape)
```
## 4. 이미지 데이터의 입력 차원

이미지 Tensor는 보통 4차원입니다.

```
(batch_size, channels, height, width)
```

예를 들어 흑백 이미지 16장이 있고, 각 이미지가 28×28 크기라고 가정

```python
import torch

# batch size 16, channel 1, height 28, width 28
images = torch.randn(16, 1, 28, 28)

print(images.shape)
```

출력

```
torch.Size([16, 1, 28, 28])
```

MLP는 샘플 하나를 긴 벡터로 받으므로, 샘플 하나의 feature 수는 다음과 같다.

```
1 * 28 * 28 = 784
```

따라서 flatten 후 shape

```
(16, 1, 28, 28) -> (16, 784)
```

## 5. flatten이 필요한 이유

이미지 한 장은 2차원 격자처럼 보입니다. 컬러 이미지라면 채널까지 포함해 3차원 구조입니다.

하지만 MLP의 첫 번째 `nn.Linear`는 일반적으로 다음과 같은 형태를 기대합니다.

```
샘플 하나 = 숫자들이 한 줄로 나열된 벡터
```

그래서 이미지 한 장의 픽셀들을 한 줄로 펼친다.

```
1 x 28 x 28 이미지 -> 길이 784 벡터
3 x 32 x 32 이미지 -> 길이 3072 벡터
```

## 6. `torch.flatten(start_dim=1)` 사용하기

`torch.flatten`을 사용할 때 가장 중요한 것은 batch 차원을 유지하는 것

```python
import torch

images = torch.randn(16, 1, 28, 28)

# start_dim=1은 1번째 차원부터 끝까지 펼친다는 뜻입니다.
# 0번째 차원인 batch size 16은 그대로 유지됩니다.
flat = torch.flatten(images, start_dim=1)

print("before:", images.shape)
print("after :", flat.shape)
```
출력

```
before: torch.Size([16, 1, 28, 28])
after : torch.Size([16, 784])
```

`torch.flatten(images)`처럼 `start_dim`을 지정하지 않으면 전체 Tensor가 한 줄로 펼쳐져 batch 차원까지 사라질 수 있다.

```python
wrong_flat = torch.flatten(images)
print(wrong_flat.shape)
```

출력.

```
torch.Size([12544])
```
이것은 샘플 16개가 모두 하나로 합쳐진 결과입니다. 학습 입력으로 사용하기 어렵다

## 7. `nn.Flatten()` 사용하기

모델 안에 flatten을 넣고 싶다면 `nn.Flatten()`을 사용할 수 있다.

```python
import torch
import torch.nn as nn

images = torch.randn(16, 1, 28, 28)

model = nn.Sequential(
    nn.Flatten(),          # (16, 1, 28, 28) -> (16, 784)
    nn.Linear(784, 10)     # (16, 784) -> (16, 10)
)

logits = model(images)

print("images shape:", images.shape)
print("logits shape:", logits.shape)
```

출력은 다음과 같습니다.

```
images shape: torch.Size([16, 1, 28, 28])
logits shape: torch.Size([16, 10])
```

`nn.Flatten()`은 기본적으로 batch 차원인 0번째 차원은 유지하고, 1번째 차원부터 펼친다.

## 8. 의도적 `in_features` 오류 수정

오류 코드

```python
import torch
import torch.nn as nn

images = torch.randn(16, 1, 28, 28)

# 잘못된 예시입니다.
# 이미지를 flatten하지 않고 Linear에 바로 넣었습니다.
model = nn.Linear(784, 10)

try:
    logits = model(images)
except RuntimeError as e:
    print("오류 발생!")
    print(e)
```

오류의 핵심은 `nn.Linear`가 입력의 마지막 차원을 봤다는 점이다

```
images shape = (16, 1, 28, 28)
마지막 차원 = 28
Linear가 기대한 in_features = 784
```

따라서 flatten을 먼저 해야 한다

```python
import torch
import torch.nn as nn

images = torch.randn(16, 1, 28, 28)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

logits = model(images)

print(logits.shape)
```

출력

```
torch.Size([16, 10])
```

## 9. 차원 계산표 만들기

MLP에 이미지를 넣을 때는 먼저 계산표를 만드는 것이 좋다.

| 데이터 | 원본 shape | flatten 후 feature 수 | MLP 첫 Linear |
| --- | --- | --- | --- |
| MNIST 흑백 이미지 | `(N, 1, 28, 28)` | `1*28*28 = 784` | `nn.Linear(784, hidden)` |
| CIFAR 형태 컬러 이미지 | `(N, 3, 32, 32)` | `3*32*32 = 3072` | `nn.Linear(3072, hidden)` |
| 작은 RGB 이미지 | `(N, 3, 64, 64)` | `3*64*64 = 12288` | `nn.Linear(12288, hidden)` |
| tabular 데이터 | `(N, 10)` | `10` | `nn.Linear(10, hidden)` |


`in_features`는 샘플 하나가 가진 feature 수입니다. 이미지에서는 `channels * height * width`가 됩니다.

## 10. `view`, `reshape`, `flatten` 비교

초반에는 `torch.flatten(images, start_dim=1)`을 가장 권장. 그래도 다른 코드에서 자주 보이는 형태를 알아두면 좋다.

```python
import torch

images = torch.randn(16, 1, 28, 28)

# 방법 1: flatten
flat1 = torch.flatten(images, start_dim=1)

# 방법 2: view
# images.size(0)은 batch size입니다.
flat2 = images.view(images.size(0), -1)

# 방법 3: reshape
flat3 = images.reshape(images.size(0), -1)

print(flat1.shape)
print(flat2.shape)
print(flat3.shape)
```

출력은 모두 같습니다.

```
torch.Size([16, 784])
torch.Size([16, 784])
torch.Size([16, 784])
```

초반 교안에서는 가독성이 좋은 `torch.flatten(x, start_dim=1)`과 모델 안에 넣기 쉬운 `nn.Flatten()`을 기본으로 사용한다.

`view`와 `reshape`의 차이
    
view() : 원본과 같은 메모리를 공유해서 shape만 바꿉니다. 그래서 Tensor가 메모리상 연속적(contiguous)이어야 합니다.
reshape() : 가능하면 같은 메모리를 공유하고, 안 되면 복사본을 만들어서라도 shape을 바꿉니다.