---
title: 2장 1강 Gradient Descent와 파라미터 갱신
date: 2026-09-10
updated: 2026-09-10
description: KANT 강의 '2장 1강 Gradient Descent와 파라미터 갱신' 정리
---

## 1. Gradient Descent는 무엇인가

$$
\hat{y}=wx+b
$$

- `w`: 기울기
- `b`: 절편
- `x`: 입력
- `y_hat`: 모델 예측


정답 데이터가 `y = 2x + 1`이라면 최종적으로 `w`는 2, `b`는 1에 가까워져야 한다 .

```python
# [실습 목적] 1. Gradient Descent는 무엇인가요?에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 모델에 입력할 교육용 데이터입니다.
x = np.array([0.0, 1.0, 2.0, 3.0])
# 정답값 또는 정답 레이블입니다.
y_true = np.array([1.0, 3.0, 5.0, 7.0])
```

처음에는 `w=0`, `b=0`으로 시작

```python
# [실습 목적] 1. Gradient Descent는 무엇인가요?에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 입력 x에 곱해지는 가중치 파라미터입니다.
w = 0.0
# 모델 출력에 더해지는 편향 파라미터입니다.
b = 0.0
# 모델이 예측한 값입니다.
y_pred = w * x + b
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(y_pred)  # [0. 0. 0. 0.]
```
예측과 정답 차이가 크므로 MSE가 크다

```python
# [실습 목적] 1. Gradient Descent는 무엇인가요?에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
loss = np.mean((y_pred - y_true) ** 2)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(loss)
```

학습의 목표는 `w`와 `b`를 바꾸어 이 loss를 줄이는 것입니다.

## 2. 모델 학습의 큰 그림

모델 학습은 예측하고, 틀린 정도를 계산하고, 조금 고치는 과정을 반복하는 일이다

접선 기울기와 gradient 직관

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_tangent_gradient.png' | relative_url }}" alt="01_tangent_gradient.png" loading="lazy">

```
입력 x
  -> 현재 파라미터 w, b로 예측 y_hat 계산
  -> 정답 y와 비교해 loss 계산
  -> loss의 gradient 계산
  -> w, b 갱신
  -> 다시 예측
```


## 3. gradient와 내려가는 방향

산의 경사도를 생각해 보겠습니다.

- 양의 gradient: 오른쪽으로 갈수록 loss가 증가한다
- 음의 gradient: 오른쪽으로 갈수록 loss가 감소힌다.
- gradient 크기: 경사가 얼마나 가파른지 나타낸다

접선과 gradient 직관

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_tangent_gradient.png' | relative_url }}" alt="01_tangent_gradient.png" loading="lazy">

Gradient Descent의 갱신식은 다음과 같다


$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

코드

```python
# [실습 목적] Gradient Descent 갱신식이 실제 숫자에서 어떻게 동작하는지 확인합니다.
# [실행 방법] 현재 파라미터, learning rate, gradient를 먼저 정한 뒤 갱신식을 실행하세요.
# [확인 포인트] gradient가 양수이면 파라미터가 작아지고, 음수이면 파라미터가 커지는지 살펴봅니다.

# 현재 학습 중인 파라미터입니다.
parameter = 1.0

# 한 번의 갱신에서 gradient를 얼마나 반영할지 정하는 값입니다.
learning_rate = 0.1

# loss가 가장 빠르게 증가하는 방향을 나타냅니다.
# 이 예제에서는 gradient가 양수이므로, loss를 줄이려면 반대인 음의 방향으로 이동해야 합니다.
gradient = 2.0

# Gradient Descent의 핵심 갱신식입니다.
new_parameter = parameter - learning_rate * gradient

print("갱신 전 parameter:", parameter)
print("갱신 후 parameter:", new_parameter)

# 1.0 - 0.1 * 2.0 = 0.8이므로 아래 검사를 통과해야 합니다.
assert abs(new_parameter - 0.8) < 1e-12
```

- `parameter`: 현재 파라미터

- `gradient`: loss가 증가하는 방향

- `learning_rate`: 한 번에 이동하는 크기

- 앞의 마이너스 부호(-): 증가하는 방향의 반대로 이동

### 간단한 숫자 예시

현재 `w=1.0`, gradient가 `-4.0`, learning rate가 `0.1`이라고 가정한다

```python
# [실습 목적] 간단한 숫자 예시에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 입력 x에 곱해지는 가중치 파라미터입니다.
w = 1.0
# 손실을 w로 미분한 gradient입니다.
grad_w = -4.0
# 한 번의 갱신에서 gradient를 얼마나 반영할지 정하는 학습률입니다.
learning_rate = 0.1

new_w = w - learning_rate * grad_w
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(new_w)  # 1.4
```

## 4. 파라미터 한 step 갱신

선형 모델의 MSE를 사용하면 `w`와 `b`의 gradient를 다음처럼 계산할 수 있다.

$$
\frac{\partial L}{\partial w}=\frac{2}{N}\sum_i(\hat{y}_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}=\frac{2}{N}\sum_i(\hat{y}_i-y_i)
$$


```python
# [실습 목적] 4. 파라미터 한 step 갱신에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 입력 x에 곱해지는 가중치 파라미터입니다.
w = 0.0
# 모델 출력에 더해지는 편향 파라미터입니다.
b = 0.0
# 한 번의 갱신에서 gradient를 얼마나 반영할지 정하는 학습률입니다.
learning_rate = 0.05

# 1. 예측
y_pred = w * x + b

# 2. 오차
errors = y_pred - y_true

# 3. gradient
grad_w = 2.0 * np.mean(errors * x)
# 손실을 b로 미분한 gradient입니다.
grad_b = 2.0 * np.mean(errors)

# 4. 갱신
w = w - learning_rate * grad_w
# 모델 출력에 더해지는 편향 파라미터입니다.
b = b - learning_rate * grad_b

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("grad_w:", grad_w)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("grad_b:", grad_b)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("updated w:", w)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("updated b:", b)
```

출력

```
grad_w: -17.0
grad_b: -8.0
updated w: 0.8500000000000001
updated b: 0.4
```

한 step 파라미터 갱신표

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/step_parameters.png' | relative_url }}" alt="step_parameters.png" loading="lazy">


```python
# [실습 목적] 4. 파라미터 한 step 갱신에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

new_pred = w * x + b
new_loss = np.mean((new_pred - y_true) ** 2)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("new loss:", new_loss)
```
출력

```
ew loss: 7.05875
```
