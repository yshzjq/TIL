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

초기 loss보다 작아졌다면 첫 step은 올바른 방향으로 이동한 것이다

## 5. 학습을 이루는 핵심 구성요소

Gradient Descent

| 요소 | 의미 | 너무 크거나 잘못되었을 때 |
| --- | --- | --- |
| 파라미터 `w`, `b` | 모델이 학습해야 할 값 | 모델이 원하는 관계를 표현하지 못합니다. |
| loss | 현재 예측의 오차 | 잘못 정의하면 엉뚱한 목표를 최적화합니다. |
| gradient | loss가 증가하는 방향과 민감도 | 부호나 식이 틀리면 loss가 증가합니다. |
| learning rate | 한 번에 이동할 크기 | 너무 크면 발산하고, 너무 작으면 느립니다. |
| 반복 횟수 | 갱신을 수행하는 횟수 | 부족하면 수렴 전에 멈춥니다. |

Gradient Descent = loss를 줄이도록 parameter를 반복해서 업데이트하는 방법


갱신식

$$
\theta \leftarrow \theta - \eta \nabla_{\theta} L
$$

여기서 `θ`는 파라미터, `η`는 learning rate, `∇L`은 gradient입니다.

## 6. 입력/파라미터/출력 shape

선형 모델의 입력과 출력 shape

| 변수 | 예시 shape | 의미 |
| --- | --- | --- |
| `x` | `[B]` | 배치에 들어 있는 입력값 |
| `y` | `[B]` | 각 입력의 정답값 |
| `w` | scalar | 입력에 곱하는 가중치 |
| `b` | scalar | 출력에 더하는 편향 |
| `y_hat` | `[B]` | 모델의 예측값 |
| `loss` | scalar | 배치의 대표 오차 |

`w`와 `b`가 scalar이므로 NumPy broadcasting으로 배치 전체에 적용된다

더 복잡한 모델에서는 파라미터가 벡터나 행렬이 되지만 학습 원리는 같다.


## 7. 학습 루프와 loss 해석

```python
# [실습 목적] 7. 학습 루프와 loss 해석에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def train_linear_regression(
    x: np.ndarray,
    y_true: np.ndarray,
    learning_rate: float = 0.05,
    steps: int = 200,
):
    # 입력 x에 곱해지는 가중치 파라미터입니다.
    w = 0.0
    # 모델 출력에 더해지는 편향 파라미터입니다.
    b = 0.0
    history = []

    for step in range(steps):
        # forward
        y_pred = w * x + b

        # loss
        errors = y_pred - y_true
        # 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
        loss = np.mean(errors ** 2)

        # gradients
        grad_w = 2.0 * np.mean(errors * x)
        # 손실을 b로 미분한 gradient입니다.
        grad_b = 2.0 * np.mean(errors)

        # update
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        history.append(loss)

        if step % 40 == 0:
            # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
            print(
                f"step={step:3d} "
                f"loss={loss:.6f} "
                f"w={w:.4f} b={b:.4f}"
            )

    return w, b, np.array(history)
```

출력

```
step=  0 loss=21.000000 w=0.8500 b=0.4000
step= 40 loss=0.000087 w=2.0070 b=0.9850
step= 80 loss=0.000008 w=2.0021 b=0.9955
step=120 loss=0.000001 w=2.0006 b=0.9987
step=160 loss=0.000000 w=2.0002 b=0.9996
final w: 2.0000579170826387
final b: 0.9998763446184832
final loss: 5.890963230424943e-09
```

최종 `w`는 2, `b`는 1에 가까워진다

### loss curve를 확인

```python
# [실습 목적] loss curve를 확인합니다에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# Matplotlib은 loss 변화나 파라미터 이동을 그래프로 확인할 때 사용합니다.
import matplotlib.pyplot as plt

plt.plot(history)
plt.xlabel("step")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.show()
```

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/Training_Loss_Value.png' | relative_url }" alt="Training_Loss_Value.png" loading="lazy">

learning rate는 한 step의 이동 크기입니다.

- 너무 작음: 안정적이지만 매우 느리다.
- 적절함: 빠르고 안정적으로 줄어든다.
- 너무 큼: 최솟값을 지나치며 흔들리거나 발산할 수 있다.

learning rate에 따른 이동 경로

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_learning_rate_paths.png' | relative_url }" alt="01_learning_rate_paths.png" loading="lazy">

```python
# [실습 목적] loss curve를 확인합니다에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# overflow/nan을 예외로 승격시켜야 except 분기가 실제로 동작합니다.
np.seterr(over="raise", invalid="raise")

rates = [0.005, 0.05, 0.5]
results = {}

for lr in rates:
    try:
        w, b, losses = train_linear_regression(
            x,
            y_true,
            # 한 번의 갱신에서 gradient를 얼마나 반영할지 정하는 학습률입니다.
            learning_rate=lr,
            steps=80,
        )
        results[lr] = losses
        # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
        print(f"lr={lr}, final_loss={losses[-1]:.6f}")
    except FloatingPointError as exc:
        # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
        print(f"lr={lr} failed:{exc}")
```
출력
```
step=  0 loss=21.000000 w=0.0850 b=0.0400
step= 40 loss=0.677647 w=1.6733 b=0.7910
lr=0.005, final_loss=0.024384
step=  0 loss=21.000000 w=0.8500 b=0.4000
step= 40 loss=0.000087 w=2.0070 b=0.9850
lr=0.05, final_loss=0.000008
step=  0 loss=21.000000 w=8.5000 b=4.0000
step= 40 loss=578105839479201120061108951300990641897472.000000 w=1075661279084241354752.0000 b=503812793488637755392.0000
lr=0.5, final_loss=1551745137881676604039625909820584437733317502313878640695771169627922927703818240.000000
```


각 learning rate의 loss curve를 한 그래프에 표시합니다.

```python
# [실습 목적] loss curve를 확인합니다에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

for lr, losses in results.items():
    plt.plot(losses, label=f"lr={lr}")

plt.yscale("log")
plt.xlabel("step")
plt.ylabel("MSE loss (log scale)")
plt.legend()
plt.show()
```


학습률별 loss curve


<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_lr_curves.png' | relative_url }" alt="01_lr_curves.png" loading="lazy">



## 8. 연습 문제: 한 step 직접 계산

## 9. 연습 문제: learning rate 비교

같은 데이터로 learning rate만 바꾸어 loss 변화를 비교

```python
# [실습 목적] 9. 연습 문제: learning rate 비교에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 모델에 입력할 교육용 데이터입니다.
x = np.array([0.0, 1.0, 2.0, 3.0])
# 입력 x에 대응하는 정답 데이터입니다.
y = 2.0 * x + 1.0

def train_linear_model(learning_rate, steps=50):
    # 모든 실험을 같은 초기값에서 시작해 공정하게 비교합니다.
    w = 0.0
    # 모델 출력에 더해지는 편향 파라미터입니다.
    b = 0.0
    loss_history = []

    for step in range(steps):
        # 현재 파라미터로 배치 전체의 예측값을 계산합니다.
        y_hat = w * x + b

        # MSE의 평균값을 기록합니다.
        error = y_hat - y
        # 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
        loss = np.mean(error ** 2)
        loss_history.append(loss)

        # MSE를 w와 b로 미분한 gradient를 계산합니다.
        grad_w = 2.0 * np.mean(error * x)
        # 손실을 b로 미분한 gradient입니다.
        grad_b = 2.0 * np.mean(error)

        # gradient 반대 방향으로 파라미터를 갱신합니다.
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, np.array(loss_history)

for lr in [0.001, 0.05, 0.5]:
    w, b, losses = train_linear_model(lr)
    # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
    print(f'lr={lr} | w={w:.4f} | b={b:.4f} | final_loss={losses[-1]:.6f}')
```

출력

```
lr=0.001 | w=0.6962 | b=0.3279 | final_loss=9.183357
lr=0.05 | w=2.0054 | b=0.9885 | final_loss=0.000051
lr=0.5 | w=-38120093994113950396973056.0000 | b=-17854496965414991337357312.0000 | final_loss=726046123416057853265817957022568110185710729822208.000000
```


1. 가장 느리게 학습되는 설정은 무엇인가

2. loss가 커지거나 `nan`이 되면 가장 먼저 무엇을 의심해야 하나

정답

1. 일반적으로 `0.001`이 가장 느리다. 한 번의 이동이 작기 때문입니다.

2. learning rate가 너무 큰지, gradient 식과 데이터에 이상값이 없는지 먼저 확인합니다. 
<br>
실습 환경에 따라 `0.5`는 진동하거나 발산할 수 있다.


## 10. PyTorch 학습 루프와의 연결

PyTorch에서는 `autograd`가 gradient를 자동 계산하고 optimizer가 갱신을 수행합니다. 하지만 내부 흐름은 동일하다

```
optimizer.zero_grad()
-> y_hat = model(x)
-> loss = criterion(y_hat, y)
-> loss.backward()
-> optimizer.step()
```

NumPy 미니 구현을 이해하면 `backward()`와 `step()`이 각각 무엇을 대신하는지 분명해진다

## 12. 이해도 점검

1. Gradient Descent 갱신식에서 gradient를 빼는 이유는 무엇인가
2. learning rate가 너무 작은 경우와 너무 큰 경우의 현상을 각각 설명
3. `w`와 `b`는 모델의 무엇인가
4. 학습 루프의 네 단계를 순서
5. `loss.backward()`와 `optimizer.step()`의 역할은 각각 무엇인가

- 정답 

1. gradient는 loss가 증가하는 방향이므로 반대 방향으로 이동해야 loss를 줄일 수 있기 때문입니다.
2. 너무 작으면 수렴이 느리고, 너무 크면 최소점을 지나쳐 진동하거나 발산할 수 있다.
3. 모델 파라미터입니다.
4. 예측 -> loss 계산 -> gradient 계산 -> 파라미터 갱신입니다.
5. `backward()`는 gradient를 계산하고, `step()`은 계산된 gradient를 이용해 파라미터를 갱신한다

