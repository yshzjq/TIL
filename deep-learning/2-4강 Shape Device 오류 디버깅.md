---
title: 2-4강 Shape Device 오류 디버깅
date: 2026-08-20
updated: 2026-08-20
description: KANT 강의 '2-4강 Shape Device 오류 디버깅' 정리
---

## 1. 딥러닝 오류를 읽는 기본 순서

```
1. RuntimeError가 있는 마지막 줄을 봅니다.
2. shape, dtype, device 중 무엇이 문제인지 분류합니다.
3. 오류가 난 연산 직전의 Tensor 정보를 출력합니다.
4. 모델이 기대한 값과 실제 Tensor 정보를 비교합니다.
5. 수정 후 다시 shape, dtype, device를 출력합니다.
```

## 2. 디버깅용 함수 준비하기

먼저 Tensor 정보를 출력하는 함수를 준비합니다.

```python
import torch

def describe_tensor(name, tensor):
    """
    디버깅을 위해 Tensor의 핵심 정보를 출력합니다.
    """
    print(f"[{name}]")
    print(f"  shape : {tensor.shape}")
    print(f"  ndim  : {tensor.ndim}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print()
```

## 3. 오류 1: `nn.Linear` 입력 차원 오류

`nn.Linear`는 입력 Tensor의 마지막 차원이 `in_features`와 같아야 합니다. 

출력은 마지막 차원이 `out_features`로 바뀝니다.

### 깨진 코드

```python
import torch
import torch.nn as nn

model = nn.Linear(in_features=4, out_features=2)

# 모델은 feature 4개를 기대합니다.
# 그런데 실제 입력은 feature가 5개입니다.
x_wrong = torch.randn(8, 5)

try:
    output = model(x_wrong)
except RuntimeError as e:
    print("오류 발생!")
    print(e)
```
오류 메시지는 다음과 비슷합니다.

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied

Linear 계층이 기대한 입력 feature 수와
실제 입력 Tensor의 마지막 차원이 맞지 않다는 뜻
```

### 오류 확인

```python
describe_tensor("x_wrong", x_wrong)

print("model expects in_features:", model.in_features)
print("actual input last dim 
```
예상 출력

```
[x_wrong]
  shape : torch.Size([8, 5])
  ndim  : 2
  dtype : torch.float32
  device: cpu

model expects in_features: 4
actual input last dim    : 5
```

```
모델 기대값: 4
실제 입력값: 5
```

### 수정 방법 1: 입력 데이터를 모델에 맞추기

```python
import torch
import torch.nn as nn

model = nn.Linear(in_features=4, out_features=2)

# feature 수를 4개로 맞춥니다.
x = torch.randn(8, 4)

output = model(x)

print(output.shape)
```

출력

```
torch.Size([8, 2])
```

### 수정 방법 2: 모델을 데이터에 맞추기

```python
import torch
import torch.nn as nn

# 실제 입력 feature가 5개라면 모델의 in_features를 5로 바꿉니다.
model = nn.Linear(in_features=5, out_features=2)

x = torch.randn(8, 5)

output = model(x)

print(output.shape)
```

출력은 다음과 같습니다.

```
torch.Size([8,2])
```

## 4. 오류 2: batch 차원 누락

샘플 하나를 모델에 넣을 때 batch 차원이 빠져도 `nn.Linear`는 동작할 수 있다.

하지만 학습 코드 전체에서는 shape가 헷갈릴 수 있으므로 보통 batch 차원을 유지하는 편이 좋다

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)

# 샘플 하나입니다.
sample = torch.randn(4)

# Linear는 이 입력도 처리할 수 있습니다.
output = model(sample)

print("sample:", sample.shape)
print("output:", output.shape)
```
출력은 다음과 같습니다.

```
sample: torch.Size([4])
output: torch.Size([2])
```
하지만 학습 루프에서는 보통 다음 형태가 더 안전합니다.

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)

sample = torch.randn(4)

# batch dimension을 추가합니다.
# (4,) -> (1, 4)
sample_batch = sample.unsqueeze(0)

output = model(sample_batch)

print("sample_batch:", sample_batch.shape)
print("output      :", output.shape)
```
```
sample_batch: torch.Size([1, 4])
output      : torch.Size([1, 2])
```
## 5. 오류 3: `CrossEntropyLoss` target dtype 오류

분류 문제에서는 `nn.CrossEntropyLoss`를 자주 사용합니다.

보통 모델 출력 shape

```
logits shape: (batch_size, num_classes)
```

정답 라벨 shape

```
target shape: (batch_size,)
```

target이 클래스 번호라면 dtype은 torch.long이어야 한다

### 깨진 코드

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# 4개 샘플, class 3개에 대한 모델 출력값입니다.
logits = torch.randn(4, 3)

# 정답 라벨입니다.
# 값은 0, 1, 2 클래스 번호처럼 보이지만 dtype이 float입니다.
target_wrong = torch.tensor([0.0, 1.0, 2.0, 1.0])

try:
    loss = criterion(logits, target_wrong)
except RuntimeError as e:
    print("오류 발생!")
    print(e)
```

```
RuntimeError: expected scalar type Long but found Float
```
### 오류 확인

```python
describe_tensor("logits", logits)
describe_tensor("target_wrong", target_wrong)
```

예상 출력은 다음과 같습니다.

```
[logits]
  shape : torch.Size([4, 3])
  ndim  : 2
  dtype : torch.float32
  device: cpu

[target_wrong]
  shape : torch.Size([4])
  ndim  : 1
  dtype : torch.float32
  device: cpu
```
문제는 `target_wrong.dtype`입니다.

```
현재 dtype: torch.float32
필요 dtype: torch.long
```
### 수정 코드

```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.randn(4, 3)

# target을 class index로 사용할 것이므로 dtype을 long으로 만듭니다.
target = torch.tensor([0, 1, 2, 1], dtype=torch.long)

loss = criterion(logits, target)

print(loss)
```
이미 만들어진 Tensor를 변환할 수 있다.

```python
target = target_wrong.long()
```

## 6. 오류 4: device mismatch

모델과 Tensor가 서로 다른 device에 있으면 오류가 난다

### 깨진 코드

```python
import torch
import torch.nn as nn

if torch.cuda.is_available():
    model = nn.Linear(4, 2).to("cuda")

    # 일부러 CPU Tensor를 만듭니다.
    x_cpu = torch.randn(8, 4)

    try:
        output = model(x_cpu)
    except RuntimeError as e:
        print("오류 발생!")
        print(e)
else:
    print("현재 환경에서는 CUDA를 사용할 수 없어 예시를 건너뜁니다.")
```
오류 확인
```python
if torch.cuda.is_available():
    print("model device:", next(model.parameters()).device)
    print("x_cpu device:", x_cpu.device)
```
출력
```
model device: cuda:0
x_cpu device: cpu
```

### 수정 코드

```python
import torch
import torch.nn as nn

# cuba 사용 가능 시 cuba로 통일
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(4, 2).to(device)

x = torch.randn(8, 4).to(device)

output = model(x)

print("model device :", next(model.parameters()).device)
print("x device     :", x.device)
print("output device:", output.device)
```

## 7. 오류 5: 의도치 않은 broadcasting으로 loss shape가 커지는 경우

```python
import torch

pred = torch.randn(4, 1)
target = torch.randn(4)

loss_wrong = ((pred - target) ** 2).mean()

print("pred shape  :", pred.shape)
print("target shape:", target.shape)
print("pred-target :", (pred - target).shape)
print("loss_wrong :", loss_wrong)
```

출력은 다음과 같습니다.

```
pred shape  : torch.Size([4, 1])
target shape: torch.Size([4])
pred-target : torch.Size([4, 4])
```

오류는 나지 않았지만, `pred - target`의 shape가 `(4, 4)`가 되었습니다.

이것은 샘플별 차이를 계산한 것이 아니라 모든 조합의 차이를 계산

### 수정 코드

```python
import torch

pred = torch.randn(4, 1)
target = torch.randn(4)

# target을 pred와 같은 shape로 바꿉니다.
target = target.unsqueeze(1)

assert pred.shape == target.shape, f"shape mismatch: pred={pred.shape}, target={target.shape}"

loss = ((pred - target) ** 2).mean()

print("pred shape  :", pred.shape)
print("target shape:", target.shape)
print("loss        :", loss)
```