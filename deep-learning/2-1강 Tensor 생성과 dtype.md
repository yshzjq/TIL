---
title: 2-1강 Tensor 생성과 dtype
date: 2026-08-19
updated: 2026-08-19
description: KANT 강의 '2-1강 Tensor 생성과 dtype' 정리
---

## 1. 이번에 우리가 배울 것

딥러닝 모델은 숫자를 입력받고 숫자를 출력

이미지도 숫자, 텍스트도 숫자, 정답 라벨도 숫자로 바꿔서 모델에 넣는다

이때 PyTorch에서 숫자 데이터를 담는 기본 그릇이 **Tensor**다.

Tensor는 다음과 같이 생각하면 됩니다.

| 비유 | 딥러닝에서의 의미 |
| --- | --- |
| 엑셀 표 | 2차원 Tensor |
| 여러 장의 엑셀 표 | 3차원 Tensor |
| 여러 장의 컬러 이미지 묶음 | 4차원 Tensor |
| 모델이 읽는 숫자 상자 | Tensor |

딥러닝에서 Tensor를 볼 때는 값 자체보다 먼저 **shape, dtype, device**를 확인해야 한다

<img src="{{ '/assets/images/uploads/deep-learning/Tensor차원구조.png' | relative_url }}" alt="Tensor차원구조.png" loading="lazy">

이미지 설명: Scalar, Vector, Matrix, Tensor를 한 번에 비교하는 그림

## 2. Tensor의 차원 이해하기

Tensor의 차원은 “몇 겹으로 숫자가 감싸져 있는가”를 뜻합니다.

### 0차원 Tensor: Scalar

숫자 하나입니다.

```python
import torch

# 숫자 하나를 Tensor로 만듭니다.
# 이런 Tensor를 scalar라고 부릅니다.
scalar = torch.tensor(7)

print(scalar)
print(scalar.shape)
print(scalar.ndim)
```

출력 결과

```
tensor(7)
torch.Size([])
0
```

`torch.Size([])`는 비어 있는 것처럼 보이지만, 오류가 아닙니다.

값이 하나뿐이라서 행도 열도 없는 **0차원 Tensor**라는 뜻이다.


### 1차원 Tensor: Vector

숫자가 한 줄로 나열된 구조.

```python
# 숫자 3개가 한 줄로 들어 있는 Tensor입니다.
# 벡터(vector)라고 생각하면 됩니다.
vector = torch.tensor([1, 2, 3])

print(vector)
print(vector.shape)
print(vector.ndim)
```

예상 출력

```
tensor([1, 2, 3])
torch.Size([3])
1
```

`torch.Size([3])`은 값이 3개 있는 1차원 Tensor라는 뜻.


### 2차원 Tensor: Matrix

행과 열이 있는 표 형태

```python
# 2행 3열짜리 Tensor입니다.
# 엑셀 표처럼 생각하면 이해하기 쉽습니다.
matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix)
print(matrix.shape)
print(matrix.ndim)
```

예상 출력

```
tensor([[1, 2, 3],
        [4, 5, 6]])
torch.Size([2, 3])
2
```

`torch.Size([2, 3])`은 다음처럼 읽습니다.

```
2개의 행이 있고,
각 행마다 3개의 값이 있습니다.
```

즉, 2행 3열입니다.


### 3차원 Tensor: 여러 개의 행렬 묶음

3차원 Tensor는 2차원 표가 여러 장 쌓인 구조

```python
# 2개의 행렬이 들어 있는 3차원 Tensor입니다.
tensor_3d = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [7, 8, 9],
        [10, 11, 12]
    ]
])

print(tensor_3d)
print(tensor_3d.shape)
print(tensor_3d.ndim)
```

예상 출력

```
torch.Size([2, 2, 3])
3
```



```
`torch.Size([2, 2, 3])`

2개의 묶음이 있고, 각 묶음 안에는 2개의 행이 있고, 각 행에는 3개의 값이 있습니다.
```

## 3. shape를 읽는 기본 규칙

shape는 Tensor의 구조를 왼쪽에서 오른쪽으로 설명

| shape | 읽는 방법 | 예시 |
| --- | --- | --- |
| `torch.Size([])` | 값 하나 | scalar |
| `torch.Size([3])` | 값 3개 | vector |
| `torch.Size([2, 3])` | 2행 3열 | matrix |
| `torch.Size([10, 4])` | 샘플 10개, 특성 4개 | tabular batch |
| `torch.Size([32, 3, 224, 224])` | 이미지 32장, 채널 3개, 높이 224, 너비 224 | image batch |

## 4. Tensor 속성 확인하기

PyTorch Tensor는 값만 가지고 있는 것이 아닙니다.

Tensor는 다음 정보를 함께 가지고 있습니다.

| 속성 | 의미 | 예시 |
| --- | --- | --- |
| `shape` | Tensor의 모양 | `torch.Size([3, 4])` |
| `ndim` | Tensor의 차원 수 | `2` |
| `dtype` | 값의 자료형 | `torch.float32` |
| `device` | Tensor가 저장된 장치 | `cpu`, `cuda:0` |

```python
import torch

# 3행 4열짜리 무작위 Tensor를 만듭니다.
# torch.randn은 표준정규분포에서 무작위 값을 뽑습니다.
x = torch.randn(3, 4)

# shape: Tensor의 모양입니다.
print("shape:", x.shape)

# ndim: Tensor가 몇 차원인지 알려줍니다.
print("ndim:", x.ndim)

# dtype: Tensor 안에 들어 있는 값의 자료형입니다.
print("dtype:", x.dtype)

# device: Tensor가 CPU에 있는지 GPU에 있는지 알려줍니다.
print("device:", x.device)
```

예상 출력

```
shape: torch.Size([3, 4])
ndim: 2
dtype: torch.float32
device: cpu
```

## 5. dtype이 중요한 이유

`dtype`은 Tensor 안에 들어 있는 숫자의 자료형

| dtype | 주로 쓰는 곳 |
| --- | --- |
| `torch.float32` | 입력 데이터, 모델 가중치, 회귀 정답 |
| `torch.float64` | 더 정밀한 실수 연산이 필요한 경우 |
| `torch.int64` 또는 `torch.long` | `CrossEntropyLoss`에서 클래스 인덱스로 쓰는 라벨 |
| `torch.bool` | 조건 마스크 |

예를 들어 이미지 픽셀 값, 임베딩 벡터, 모델의 가중치는 보통 실수입니다.

반면 `CrossEntropyLoss`로 다중 클래스 분류를 할 때의 정답은 보통 정수 클래스 번호입니다. 이진 분류의 `BCEWithLogitsLoss` target은 예외로, 0/1 값을 가진 실수형 Tensor를 사용합니다.

```python
import torch

# 입력 데이터는 보통 실수형 Tensor로 다룹니다.
features = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
], dtype=torch.float32)

# 분류 라벨은 보통 정수형 Tensor로 다룹니다.
# 예: 0번 클래스, 2번 클래스
labels = torch.tensor([0, 2], dtype=torch.long)

print(features.dtype)
print(labels.dtype)
```

⛔ **주의사항**

분류 target의 dtype은 loss 함수에 따라 다릅니다.

```
CrossEntropyLoss + class index target: torch.long
BCEWithLogitsLoss + 0/1 target: torch.float32 등 입력과 같은 실수형
```

같은 ‘분류 라벨’이라도 loss가 기대하는 형식을 먼저 확인해야 합니다.

---