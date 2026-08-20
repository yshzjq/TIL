---
title: 2-3강 CPU/GPU device와 .to(device)
date: 2026-08-20
updated: 2026-08-20
description: KANT 강의 '2-3강 CPU/GPU device와 .to(device)' 정리
---

## 1. device란?

`device`는 Tensor가 어디에 저장되어 있는지 나타낸다.

대표적으로 다음 두 가지가 있습니다.

| device | 의미 |
| --- | --- |
| `cpu` | CPU 메모리에 Tensor가 있음 |
| `cuda` 또는 `cuda:0` | NVIDIA GPU 메모리에 Tensor가 있음 |

PyTorch Tensor는 기본적으로 CPU에 생성됩니다. GPU에서 연산하려면 Tensor를 명시적으로 이동해야 한다

## 2. 현재 Tensor의 device 확인하기

```python
import torch

x = torch.randn(2, 3)

print(x.device)
```

예상 출력

```
cpu
```

Colab에서 GPU 런타임을 켰더라도, Tensor를 만들면 처음에는 보통 CPU에 있다.

GPU를 켰다는 것과 Tensor가 GPU에 올라갔다는 것은 다르다

Colab GPU를 켰다고 해서 모든 Tensor가 자동으로 GPU로 이동하지 않습니다. 직접 .to(device)를 사용해야 한다

## 3. GPU 사용 가능 여부 확인하기

```python
import torch

print(torch.cuda.is_available())
```

출력은 `True` 또는 `False`

`torch.cuda.is_available()`은 현재 CUDA 사용 가능 여부를 bool 값으로 반환

```python
import torch

if torch.cuda.is_available():
    print("GPU를 사용할 수 있습니다.")
else:
    print("GPU를 사용할 수 없습니다. CPU로 실행합니다.")
```

## 4. device 변수 만들기

실전 코드에서는 보통 다음 패턴을 사용합니다.

```python
import torch

# CUDA GPU가 사용 가능하면 cuda를 사용하고,
# 그렇지 않으면 cpu를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
```

## 5. Tensor를 device로 이동하기

Tensor는 `.to(device)`로 이동합니다.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(2, 3)

print("before:", x.device)

# x를 device로 이동합니다.
# Tensor의 .to()는 이동된 Tensor를 반환하므로,
# 반환값을 다시 x에 저장해야 합니다.
x = x.to(device)

print("after :", x.device)
```

Tensor의 `.to(device)`는 이동된 Tensor를 반환하므로 반환값을 저장해야 한다.<br>
반면 `nn.Module.to(device)`는 모듈의 파라미터와 버퍼를 이동시키고 모듈 자신을 반환<br>
 코드 스타일을 일관되게 유지하기 위해 둘 다 `x = x.to(device)`, `model = model.to(device)`처럼 작성하면 좋다.

```python
x = torch.randn(2, 3)

# 이렇게만 쓰면 반환된 Tensor를 저장하지 않습니다.
x.to(device)

print(x.device)
```

Tensor는 여전히 CPU에 있을 수 있어서 아래처럼 작성

```python
x = x.to(device)
```

## 6. 모델도 같은 device로 이동하기

데이터만 GPU로 보내면 안되고 <br>
모델도 같은 device에 있어야 한다.

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature 4개를 입력받아 class 2개를 출력하는 모델입니다.
model = nn.Linear(4, 2)

# 모델을 device로 이동합니다.
model = model.to(device)

# 입력 Tensor도 같은 device로 이동합니다.
x = torch.randn(8, 4).to(device)

# 모델과 입력이 같은 device에 있으므로 정상 실행됩니다.
output = model(x)

print(output.shape)
print(output.device)
```

예상 출력

```
torch.Size([8, 2])
cuda:0
```

GPU가 없을 시

```
torch.Size([8, 2])
cpu
```

## 7. 모델 파라미터의 device 확인하기

모델 자체에는 `.device`가 바로 보이지 않을 때가 있다

모델 파라미터 하나를 꺼내서 device를 확인할 수 있다.

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Linear(4, 2).to(device)

# 모델의 첫 번째 파라미터를 꺼내 device를 확인합니다.
print(next(model.parameters()).device)
```
## 8. device mismatch 오류 이해하기


모델은 GPU에 있고 입력 Tensor는 CPU에 있을때

이 상태에서 모델에 입력을 넣으면 오류가 발생한다.

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
    print("현재 환경에서는 CUDA를 사용할 수 없어 device mismatch 예시를 건너뜁니다.")
```

오류 메시지

```
모델과 입력 Tensor가 서로 다른 device에 있습니다.
```
device mismatch 오류 예시

학습 루프에서 `images`는 GPU로 보냈는데 `labels`를 CPU에 둔 경우, 또는 모델은 GPU인데 새로 만든 Tensor가 CPU에 남아 있는 경우 자주 발생.

## 9. 학습 루프에서 device 처리 패턴

앞으로 학습 루프를 만들 때는 다음 패턴을 거의 그대로 사용합니다.

**[의사코드 — 독립 실행 대상 아님]** 아래의 `model`, `dataloader`, `criterion`은 실제 과제에서 앞 셀에 정의되어 있다고 가정합니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model은 학습 시작 전에 한 번 device로 이동합니다.
model = model.to(device)

for inputs, labels in dataloader:
    # 입력과 정답을 모두 같은 device로 이동합니다.
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

모델·입력·정답을 같은 device에 두는 위치를 확인하는 것

## 10. 새 Tensor를 만들 때 device 맞추기

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(4, 3).to(device)

# 새 Tensor를 만들 때 x와 같은 device에 만들고 싶다면 device=x.device를 사용합니다.
bias = torch.zeros(3, device=x.device)

result = x + bias

print(x.device)
print(bias.device)
print(result.device)
```

또는 기존 Tensor와 같은 속성을 유지하려면 `torch.zeros_like`를 사용할 수 있다.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(4, 3).to(device)

# x와 같은 shape, dtype, device를 가진 0 Tensor를 만듭니다.
z = torch.zeros_like(x)

print(z.shape)
print(z.dtype)
print(z.device)
```
`ones_like`, `rand_like` 같은 함수는 기존 Tensor의 shape, dtype, device를 유지

