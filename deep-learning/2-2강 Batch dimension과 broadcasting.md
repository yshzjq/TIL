---
title: 2-2강 Batch dimension과 broadcasting
date: 2026-08-20
updated: 2026-09-01
description: KANT 강의 '2-2강 Batch dimension과 broadcasting' 정리
---

## 1. Batch dimension이란?

딥러닝 모델은 데이터를 여러 개를 묶어서 한 번에 처리한다

이 묶음을 **batch**라고 한다.

예를 들어 학생 한 명의 정보가 있으면

```plain
[키,몸무게,공부시간,수면시간]
```

학생 한 명은 feature가 4개인 1차원 Tensor입니다.

```python
import torch

# 학생 1명의 feature 4개
one_student = torch.tensor([170.0, 65.0, 5.0, 7.0])

print(one_student.shape)
```

출력

```plain
torch.Size([4])
```

하지만 학생 3명을 한 번에 모델에 넣으려면 만든다

```python
# 학생 3명, 각 학생마다 feature 4개
students = torch.tensor([
    [170.0, 65.0, 5.0, 7.0],
    [160.0, 50.0, 3.0, 6.0],
    [180.0, 80.0, 6.0, 8.0]
])

print(students.shape)
```

출력

```plain
torch.Size([3,4])
```

```plain
batch size = 3
feature 수 = 4
```

⭐ **중요**

딥러닝에서 2차원 입력 Tensor는 보통 `(batch_size, features)` 형태

## 2. 이미지 데이터의 batch dimension

표 데이터는 보통 `(batch_size, features)` 형태.

이미지 데이터는 보통 다음 형태를 사용

```plain
(batch_size, channels, height, width)
```

예를 들어 컬러 이미지 32장을 모델에 넣는다고 해봅시다.

```python
import torch

# 32장의 컬러 이미지
# 각 이미지는 RGB 채널 3개, 높이 224, 너비 224라고 가정합니다.
images = torch.randn(32, 3, 224, 224)

print(images.shape)
```

출력

```plain
torch.Size([32,3,224,224])
```

| 위치 | 값 | 의미 |
| --- | --- | --- |
| 0번째 차원 | 32 | 이미지 개수, batch size |
| 1번째 차원 | 3 | 채널 수, RGB |
| 2번째 차원 | 224 | 높이 |
| 3번째 차원 | 224 | 너비 |

## 3. `unsqueeze`로 batch 차원 추가하기

샘플 하나만 있을 때도 모델은 batch 형태를 기대할 수 있다.

이럴 때는 `unsqueeze`로 차원을 하나 추가한다

```python
import torch

# feature 4개를 가진 샘플 하나입니다.
sample = torch.tensor([170.0, 65.0, 5.0, 7.0])

print("before:", sample.shape)

# 0번째 위치에 차원을 하나 추가합니다.
# (4,) -> (1, 4)
sample_batch = sample.unsqueeze(0)

print("after :", sample_batch.shape)
```

출력

```plain
before: torch.Size([4])
after : torch.Size([1, 4])
```

## 4. `squeeze`로 크기가 1인 차원 제거하기

`squeeze`는 크기가 1인 차원을 제거.

```python
import torch

x = torch.randn(1, 4)

print("before:", x.shape)

# 크기가 1인 차원을 제거합니다.
y = x.squeeze(0)

print("after :", y.shape)
```

출력

```plain
before: torch.Size([1, 4])
after : torch.Size([4])
```

`squeeze()`를 인자 없이 사용하면 크기가 1인 모든 차원을 제거

예를 들어 `(1, 1, 4)`에서 모든 1 차원이 사라져 `(4,)`가 될 수 있습니다. <br>가능하면 `squeeze(0)`처럼 제거할 차원을 명시해야한다.

## 5. Broadcasting이란?

Broadcasting은 서로 다른 shape의 Tensor를 연산할 때, PyTorch가 가능한 경우 자동으로 크기를 맞춰주는 규칙.

```python
import torch

# 2행 3열 Tensor입니다.
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# feature별로 더할 bias입니다.
bias = torch.tensor([10.0, 20.0, 30.0])

result = x + bias

print(result)
print(result.shape)
```

출력

```plain
tensor([[11., 22., 33.],
        [14., 25., 36.]])
torch.Size([2, 3])
```

`x`의 shape는 `(2, 3)`이고, `bias`의 shape는 `(3,)`

PyTorch는 `bias`를 다음처럼 해석.

```plain
bias shape: (3,)
자동 해석: (1, 3)
x shape   : (2, 3)
결과 shape: (2, 3)
```

즉, `bias`가 각 행에 반복해서 더해진 것처럼 동작합니다.

두 Tensor가 broadcastable하려면 뒤쪽 차원부터 비교했을 때 각 차원이 같거나, 둘 중 하나가 1이거나, 한쪽 차원이 존재하지 않아야 한다<br> 결과 shape는 각 차원에서 더 큰 값을 사용합니다.

## 6. Broadcasting 규칙을 오른쪽부터 읽기

Broadcasting은 shape를 오른쪽부터 비교합니다.

```plain
x shape:    (2, 3, 4)
y shape:       (3, 1)
```

차원 수가 다르면 짧은 쪽 앞에 1이 있다고 생각한다.

```plain
x shape:    (2, 3, 4)
y shape:    (1, 3, 1)
```

이제 오른쪽부터 비교합니다.

| 차원 | x | y | 가능 여부 |
| --- | --- | --- | --- |
| 마지막 차원 | 4 | 1 | 가능, 1은 확장 가능 |
| 가운데 차원 | 3 | 3 | 가능, 같음 |
| 첫 번째 차원 | 2 | 1 | 가능, 1은 확장 가능 |

```plain
result shape: (2, 3, 4)
```

## 7. Broadcasting이 실패하는 경우

예시

```python
import torch

x = torch.randn(2, 3)
y = torch.randn(2)

# x shape: (2, 3)
# y shape: (2,)
# y는 자동으로 (1, 2)처럼 해석
# 마지막 차원에서 3과 2가 맞지 않으므로 오류가 발생한다
result = x + y
```

오류 메시지

```plain
RuntimeError: The size of tensor a (3) must match the size of tensor b (2)
```

이 문제를 해결하려면 `y`의 shape가 어느 차원에 맞아야 하는지 명확히 해야 한다

```python
import torch

x = torch.randn(2, 3)

# 행마다 하나씩 더하고 싶다면 y를 (2, 1)로 만들어야 한다.
y = torch.randn(2).unsqueeze(1)

result = x + y

print(x.shape)
print(y.shape)
print(result.shape)
```

출력

```plain
torch.Size([2,3])
torch.Size([2,1])
torch.Size([2,3])
```

## 8. 의도치 않은 Broadcasting

Broadcasting은 편리하지만 위험할 때가 있다.

```python
import torch

# 모델 예측값입니다.
# 4개의 샘플에 대해 각각 예측값 1개가 있습니다.
pred = torch.tensor([
    [0.1],
    [0.2],
    [0.3],
    [0.4]
])

# 정답값입니다.
# 4개의 샘플에 대해 각각 정답 1개가 있습니다.
target = torch.tensor([0.0, 1.0, 0.0, 1.0])

print("pred shape  :", pred.shape)
print("target shape:", target.shape)

diff = pred - target

print("diff shape:", diff.shape)
print(diff)
```

출력

```plain
pred shape  : torch.Size([4, 1])
target shape: torch.Size([4])
diff shape: torch.Size([4, 4])
tensor([[ 0.1, -0.9,  0.1, -0.9],
        [ 0.2, -0.8,  0.2, -0.8],
        [ 0.3, -0.7,  0.3, -0.7],
        [ 0.4, -0.6,  0.4, -0.6]])
```

첫 번째 예측값 `0.1`이 첫 번째 정답뿐 아니라 네 정답 모두와 비교되었다. 우리는 `pred`와 `target`을 샘플별로 하나씩 빼고 싶었습니다.

하지만 결과는 `(4, 4)`가 된다.

```plain
pred shape  : (4, 1)
target shape:    (4,)
자동 해석    : (1, 4)
결과 shape  : (4, 4)
```

`target`이 `(4,)`였기 때문에 PyTorch는 이것을 `(1, 4)`처럼 보고 broadcasting 하였다.

## 9. 의도한 shape로 고치기

정답 Tensor도 `(4, 1)`로 만들어야 한다.

```python
import torch

pred = torch.tensor([
    [0.1],
    [0.2],
    [0.3],
    [0.4]
])

target = torch.tensor([0.0, 1.0, 0.0, 1.0])

# target을 (4,)에서 (4, 1)로 바꾼다
target = target.unsqueeze(1)

diff = pred - target

print("pred shape  :", pred.shape)
print("target shape:", target.shape)
print("diff shape  :", diff.shape)
print(diff)
```

출력

```plain
pred shape  : torch.Size([4, 1])
target shape: torch.Size([4, 1])
diff shape  : torch.Size([4, 1])
tensor([[ 0.1],
        [-0.8],
        [ 0.3],
        [-0.6]])
```

Loss를 계산하기 전에는 항상 예측값과 정답값의 shape를 출력해서 봐야한다.

```python
print("pred:", pred.shape)
print("target:", target.shape)
```

## 10. shape 검증 습관 만들기

실전 코드에서는 `assert`를 사용하면 좋다.

```python
import torch

pred = torch.randn(4, 1)
target = torch.randn(4, 1)

# 두 Tensor의 shape가 같은지 확인.
# 다르면 여기서 바로 멈추고 오류를 낸다.
assert pred.shape == target.shape, f"shape mismatch: pred={pred.shape}, target={target.shape}"

loss = ((pred - target) ** 2).mean()

print(loss)
```

💡 **참고사항**

`assert`는 초반 실습에서 특히 유용하다. <br>
모델 코드가 길어질수록 오류가 뒤늦게 발견되면 원인을 찾기 어렵다.<br>
그래서 중요한 연산 직전에 shape를 확인하는 코드를 넣어두면 디버깅 시간이 줄어든다.
