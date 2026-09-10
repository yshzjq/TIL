---
title: 2장 2강 SGD,Momentum,Adam 직관 비교
date: 2026-09-10
updated: 2026-09-10
description: KANT 강의 '2장 2강 SGD,Momentum,Adam 직관 비교' 정리
---

## 1. Optimizer는 무엇인가

Optimizer는 계산된 gradient를 사용해 모델 파라미터를 갱신하는 규칙이다

모든 optimizer의 목표는 loss를 줄이는 것이지만, **현재 gradient만 사용하는지, 과거 방향과 크기를 기억하는지**가 다르다

## 2. SGD/Momentum/Adam의 큰 그림

Optimizer가 기억하는 정보

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_memory.png' | relative_url }}" alt="02_optimizer_memory.png" loading="lazy">

| Optimizer | 현재 gradient | 과거 방향 | gradient 크기 정보 |
| --- | --- | --- | --- |
| SGD | 사용 | 저장하지 않음 | 저장하지 않음 |
| Momentum | 사용 | velocity에 누적 | 직접 보정하지 않음 |
| Adam | 사용 | 1차 모멘트에 누적 | 2차 모멘트에 누적 |

복잡한 optimizer가 항상 더 좋은 것은 아니다.

 데이터, 모델, learning rate, regularization을 함께 비교해야한다

 ## 3. SGD의 작동 방식

 ```python
# [실습 목적] SGD가 현재 step의 gradient만 사용해 파라미터를 갱신하는 과정을 확인합니다.
# [실행 방법] 아래 셀을 그대로 실행한 뒤, 갱신 전후 값을 비교하세요.
# [확인 포인트] SGD에는 velocity나 1차/2차 모멘트 같은 추가 상태가 없다는 점을 확인합니다.

# 현재 파라미터 값입니다.
parameter = 1.0

# 현재 step에서 계산된 gradient입니다.
gradient = 2.0

# 한 번에 이동할 크기를 정하는 학습률입니다.
learning_rate = 0.1

# SGD는 현재 gradient를 그대로 사용해 파라미터를 갱신합니다.
parameter -= learning_rate * gradient

print("SGD 갱신 후 parameter:", parameter)

# 1.0 - 0.1 * 2.0 = 0.8인지 확인합니다.
assert abs(parameter - 0.8) < 1e-12
```

출력

```
SGD 갱신 후 parameter: 0.8
```

현재 step의 gradient만 사용하므로 이해하기 쉽고 추가 상태가 거의 필요하지 않는다

### 장점

- 구조가 단순하다
- 메모리 사용량이 적다.
- 잘 조정하면 좋은 일반화 성능을 보일 수 있다.

### 어려움

- 좁고 굽은 loss 지형에서 좌우로 흔들릴 수 있다.
- learning rate 조정이 중요하다

SGD와 다른 optimizer의 이동 경로

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_paths.png' | relative_url }}" alt="02_optimizer_paths.png" loading="lazy">

이 그림은 optimizer가 반드시 항상 이런 경로를 보인다는 뜻이 아니다

같은 gradient 정보를 서로 다르게 누적하는 직관을 보여 주는 도식

## 4. Momentum의 작동 방식

Momentum은 velocity라는 상태를 유지합니다.

$$
v_t=\beta v_{t-1}+g_t
$$

$$
\theta_t=\theta_{t-1}-\eta v_t
$$

- `g_t`: 현재 gradient
- `v_t`: 이전 방향이 섞인 velocity
- `β`: 과거 방향을 얼마나 유지할지 결정하는 값

### 카트가 경사면을 내려가는 비유

SGD는 매 순간의 경사만 보고 한 걸음 움직인다

Momentum은 이전에 움직이던 방향의 관성이 남아 있어, 같은 방향의 경사가 반복되면 더 빠르게 진행한다

### 숫자로 velocity를 추적합니다

```python
# [실습 목적] 숫자로 velocity를 추적합니다에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

gradients = [2.0, 1.5, 1.0, -0.5]
beta = 0.9
# Momentum이 최근 이동 방향을 누적해 보관하는 상태값입니다.
velocity = 0.0

for step, grad in enumerate(gradients, start=1):
    # Momentum이 최근 이동 방향을 누적해 보관하는 상태값입니다.
    velocity = beta * velocity + grad
    # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
    print(f"step={step}, grad={grad:5.2f}, velocity={velocity:7.3f}")
```

출력

```
step=1, grad= 2.00, velocity=  2.000
step=2, grad= 1.50, velocity=  3.300
step=3, grad= 1.00, velocity=  3.970
step=4, grad=-0.50, velocity=  3.073
```

gradient가 같은 방향으로 이어지면 velocity가 누적된다.
<br>
마지막에 gradient 방향이 바뀌어도 이전 관성이 일부 남아 있다.

optimizer step별 상태 비교

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_step_table.png' | relative_url }}" alt="02_optimizer_step_table.png" loading="lazy">

### 간단한 Momentum update 함수

```python
# [실습 목적] 간단한 Momentum update 함수에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def momentum_update(
    parameter: float,
    gradient: float,
    velocity: float,
    learning_rate: float = 0.05,
    beta: float = 0.9,
):
    new_velocity = beta * velocity + gradient
    new_parameter = parameter - learning_rate * new_velocity
    return new_parameter, new_velocity
```

## 5. Adam의 작동 방식

Adam은 gradient의 두 가지 이동 평균을 저장한다

- **1차 모멘트 `m`:** gradient의 평균적인 방향
- **2차 모멘트 `v`:** gradient 제곱의 평균, 즉 최근 크기 정보

직관적인 갱신 흐름은 다음과 같다.

```
현재 gradient
   ├─ 방향을 부드럽게 누적 -> m
   └─ 크기를 제곱해 누적   -> v

m / (sqrt(v) + epsilon)
        ↓
파라미터마다 조절된 이동량
```

### 왜 gradient 제곱을 저장할까

어떤 파라미터의 gradient가 계속 크다면 그대로 큰 step을 반복하지 않도록 분모를 키울 수 있다. 

반대로 gradient가 작은 파라미터는 상대적으로 더 큰 보정 이동을 할 수 있다.

### 교육용으로 단순화한 Adam 한 step

```python
# [실습 목적] 교육용으로 단순화한 Adam 한 step에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def adam_update(
    parameter: float,
    gradient: float,
    m: float,
    v: float,
    step: int,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    # Adam이 gradient의 이동 평균을 저장하는 1차 모멘트입니다.
    m = beta1 * m + (1.0 - beta1) * gradient
    # Adam이 gradient 제곱의 이동 평균을 저장하는 2차 모멘트입니다.
    v = beta2 * v + (1.0 - beta2) * (gradient ** 2)

    # 초기값 0에서 시작한 이동 평균의 편향을 보정합니다.
    m_hat = m / (1.0 - beta1 ** step)
    v_hat = v / (1.0 - beta2 ** step)

    parameter = parameter - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
    return parameter, m, v
```
`m_hat`, `v_hat`의 유도는 필수 범위가 아니다. 

초기 step에서 이동 평균이 0 쪽으로 치우치는 현상을 보정한다는 의미만 이해헌더


optimizer가 저장하는 상태

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_memory.png' | relative_url }}" alt="02_optimizer_memory.png" loading="lazy">

| Optimizer | 추가로 기억하는 값 |
| --- | --- |
| SGD | 거의 없음 |
| Momentum | velocity 1개 |
| Adam | 1차 모멘트와 2차 모멘트 |

## 6. Optimizer state와 입력값

optimizer는 파라미터 외에 내부 상태를 가질 수 있다.

| 항목 | SGD | Momentum | Adam |
| --- | --- | --- | --- |
| 파라미터 | 필요 | 필요 | 필요 |
| gradient | 매 step 필요 | 매 step 필요 | 매 step 필요 |
| velocity | 없음 | 필요 | 없음 |
| 1차 모멘트 `m` | 없음 | 없음 | 필요 |
| 2차 모멘트 `v` | 없음 | 없음 | 필요 |
| step 번호 `t` | 보통 불필요 | 보통 불필요 | bias correction에 필요 |

따라서 checkpoint를 저장할 때 optimizer state도 함께 저장해야 학습을 정확히 이어갈 수 있다.


## 7. 동일 문제에서 결과 비교

2-1강의 `y = 2x + 1` 문제를 사용합니다. 교육용으로 `w`, `b` 각각의 상태를 딕셔너리로 관리하겠습니다.

```python
# [실습 목적] 7. 동일 문제에서 결과 비교에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 모델에 입력할 교육용 데이터입니다.
x = np.array([0.0, 1.0, 2.0, 3.0])
# 정답값 또는 정답 레이블입니다.
y_true = np.array([1.0, 3.0, 5.0, 7.0])

def gradients(w: float, b: float):
    # 모델이 예측한 값입니다.
    y_pred = w * x + b
    errors = y_pred - y_true
    # 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
    loss = float(np.mean(errors ** 2))
    # 손실을 w로 미분한 gradient입니다.
    grad_w = float(2.0 * np.mean(errors * x))
    # 손실을 b로 미분한 gradient입니다.
    grad_b = float(2.0 * np.mean(errors))
    return loss, grad_w, grad_b
```

### SGD 학습

```python
# [실습 목적] SGD 학습에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def train_sgd(lr: float = 0.05, steps: int = 100):
    w, b = 0.0, 0.0
    # 샘플마다 계산된 손실값을 보관합니다.
    losses = []

    for _ in range(steps):
        loss, grad_w, grad_b = gradients(w, b)
        w -= lr * grad_w
        b -= lr * grad_b
        losses.append(loss)

    return w, b, np.array(losses)
```

### Momentum 학습

```python
# [실습 목적] Momentum 학습에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def train_momentum(
    lr: float = 0.02,
    beta: float = 0.9,
    steps: int = 100,
):
    w, b = 0.0, 0.0
    vw, vb = 0.0, 0.0
    # 샘플마다 계산된 손실값을 보관합니다.
    losses = []

    for _ in range(steps):
        loss, grad_w, grad_b = gradients(w, b)
        vw = beta * vw + grad_w
        vb = beta * vb + grad_b
        w -= lr * vw
        b -= lr * vb
        losses.append(loss)

    return w, b, np.array(losses)
```

### Adam 학습

```python
# [실습 목적] Adam 학습에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def train_adam(
    lr: float = 0.05,
    steps: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    w, b = 0.0, 0.0
    mw = mb = 0.0
    vw = vb = 0.0
    # 샘플마다 계산된 손실값을 보관합니다.
    losses = []

    for step in range(1, steps + 1):
        loss, grad_w, grad_b = gradients(w, b)

        mw = beta1 * mw + (1.0 - beta1) * grad_w
        mb = beta1 * mb + (1.0 - beta1) * grad_b
        vw = beta2 * vw + (1.0 - beta2) * grad_w ** 2
        vb = beta2 * vb + (1.0 - beta2) * grad_b ** 2

        mw_hat = mw / (1.0 - beta1 ** step)
        mb_hat = mb / (1.0 - beta1 ** step)
        vw_hat = vw / (1.0 - beta2 ** step)
        vb_hat = vb / (1.0 - beta2 ** step)

        w -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        losses.append(loss)

    return w, b, np.array(losses)
```

### 결과 비교

```python
# [실습 목적] 결과 비교에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

results = {
    "SGD": train_sgd(),
    "Momentum": train_momentum(),
    "Adam": train_adam(),
}

for name, (w, b, losses) in results.items():
    # 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
    print(
        f"{name:8s} | w={w:.4f} b={b:.4f} "
        f"final_loss={losses[-1]:.8f}"
    )
```

출력

```
SGD      | w=2.0012 b=0.9975 final_loss=0.00000247
Momentum | w=2.0025 b=1.0015 final_loss=0.00024169
Adam     | w=1.7916 b=1.3872 final_loss=0.06101481
```

```python
# [실습 목적] 결과 비교에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# Matplotlib은 loss 변화나 파라미터 이동을 그래프로 확인할 때 사용합니다.
import matplotlib.pyplot as plt

for name, (_, _, losses) in results.items():
    plt.plot(losses, label=name)

plt.yscale("log")
plt.xlabel("step")
plt.ylabel("MSE loss")
plt.legend()
plt.show()
```

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_loss_curves.png' | relative_url }}" alt="02_optimizer_loss_curves.png" loading="lazy">

이 작은 문제에서 한 optimizer가 빠르다고 해서 모든 딥러닝 문제에서 항상 더 좋다는 뜻은 아니다.

 데이터, 모델, learning rate, scheduler에 따라 결과가 달라진다

optimizer 기본 선택 흐름

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_optimizer_choice_flow.png' | relative_url }}" alt="02_optimizer_choice_flow.png" loading="lazy">

### 실습/빠른 baseline

Adam 또는 AdamW를 자주 사용한다. 

비교적 적은 튜닝으로 학습이 시작되는 경우가 많다.

### 단순하고 통제된 실험

SGD는 작동 원리를 확인하고 학습률 영향을 분석하기 좋다.

### 관성이 도움이 되는 경우

Momentum은 같은 방향의 gradient가 반복되는 구간에서 이동을 부드럽게 누적한다


optimizer 이름만 바꾸고 learning rate를 그대로 두면 공정한 비교가 아닐 수 있다. 

optimizer마다 적절한 기본 learning rate 범위가 다를 수 있다.

Transformer/LLM 학습에서는 AdamW 계열을 자주 보게 된다

AdamW의 weight decay 분리는 뒤의 fine-tuning 과정에서 다시 연결한다






## 10. PyTorch Optimizer와의 연결

PyTorch에서는 다음과 같이 optimizer를 선택한다

```python
# [실습 목적] 10. PyTorch Optimizer와의 연결에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# PyTorch는 NumPy 구현과 공식 연산 결과를 비교하거나 tensor를 다룰 때 사용합니다.
import torch

# 교육용으로 파라미터 하나를 만듭니다.
parameter = torch.nn.Parameter(torch.tensor([1.0]))

# 세 optimizer 중 하나를 선택해 같은 파라미터를 관리하게 합니다.
sgd = torch.optim.SGD([parameter], lr=0.1)
momentum = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
adam = torch.optim.Adam([parameter], lr=0.001)
```


## 12. 이해도 점검

1. SGD가 현재 step에서 사용하는 핵심 정보는 무엇인가
2. Momentum이 추가로 저장하는 상태는 무엇인가
3. Adam의 1차 모멘트와 2차 모멘트는 각각 무엇을 요약하나
4. Adam에서 step 번호가 필요한 이유는 무엇인가
5. optimizer를 비교할 때 함께 기록해야 할 설정 세 가지

- 정답 확인

1. 현재 gradient와 learning rate다.
2. 최근 이동 방향을 누적한 velocity다.
3. 1차 모멘트는 gradient의 이동 평균, 2차 모멘트는 gradient 제곱의 이동 평균
4. 학습 초기 이동 평균이 0 쪽으로 치우치는 편향을 보정하기 위해 사용한다.
5. learning rate, 초기값 또는 seed, 학습 step 수를 기록해야 하며 batch size와 weight decay도 함께 기록하는 편이 좋습니다.

weight decay는 가중치(weight)가 너무 크게 커지지 않도록 조금씩 줄여주는 규제(regularization) 방법
