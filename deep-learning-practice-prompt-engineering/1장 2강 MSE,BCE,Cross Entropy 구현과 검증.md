---
title: 1장 2강 MSE,BCE,Cross Entropy 구현과 검증
date: 2026-09-10
updated: 2026-09-10
description: KANT 강의 '1장 2강 MSE,BCE,Cross Entropy 구현과 검증' 정리
---

## 1. 손실 함수는 무엇인가

모델 학습은 다음 흐름으로 진행된다

```
입력 -> 모델 -> 예측 -> 손실 함수 -> loss -> 역전파 -> 파라미터 수정
```

손실 함수는 예측과 정답의 차이를 하나의 숫자로 요약한다

일반적으로 loss가 작을수록 정답과 더 가깝다

문제 유형과 대표 손실 함수

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_loss_family_map.png' | relative_url }}" alt="02_loss_family_map.png" loading="lazy">

| 문제 유형 | 모델 출력 예 | 정답 예 | 대표 손실 함수 |
| --- | --- | --- | --- |
| 회귀 | 온도 `23.7` | `24.0` | MSE |
| 이진 분류 | 스팸일 logit 또는 확률 | `0` 또는 `1` | BCE |
| 다중 분류 | 클래스별 logits | 정답 클래스 인덱스 | Cross Entropy |

확인해야할 사항

1. 모델 출력은 값 하나입니까, 클래스별 점수입니까?
2. 정답은 실수입니까, 0/1입니까, 클래스 번호입니까?



## 2. 문제 유형과 손실 함수의 큰 그림

손실 함수는 문제 유형에 따라 달라진다.

먼저 모델이 무엇을 예측하는지 확인한 뒤 손실 함수를 선택한다

문제 유형과 손실 함수 연결

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_loss_family_map.png' | relative_url }}" alt="02_loss_family_map.png" loading="lazy">

| 문제 | 모델 출력 예 | 정답 예 | 대표 손실 |
| --- | --- | --- | --- |
| 집값 예측 | 실수 하나 | `420.0` | MSE |
| 스팸 여부 | logit 하나 또는 확률 하나 | `0` 또는 `1` | BCE / BCE with logits |
| 동물 3종 분류 | 클래스별 logits 3개 | 클래스 인덱스 `2` | Cross Entropy |

손실 함수 이름을 외우기보다 **출력 형태와 정답 형태가 맞는지** 먼저 확인

## 3. MSE 계산 원리

MSE(Mean Squared Error)는 예측과 정답의 차이를 제곱한 뒤 평균 낸다

$$
\operatorname{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

```python
# [실습 목적] 3. MSE 계산 원리에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    # 모델이 예측한 값입니다.
    y_pred = np.asarray(y_pred, dtype=np.float64)
    # 정답값 또는 정답 레이블입니다.
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred와 y_true의 shape이 같아야 합니다.")
    return float(np.mean((y_pred - y_true) ** 2))
```

좋은 예측과 나쁜 예측을 비교

```python
# [실습 목적] 3. MSE 계산 원리에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 정답값 또는 정답 레이블입니다.
y_true = np.array([2.0, 4.0, 6.0])
good_pred = np.array([2.1, 3.9, 6.2])
bad_pred = np.array([0.0, 8.0, 3.0])

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("good MSE:", mse_loss(good_pred, y_true))
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("bad MSE:", mse_loss(bad_pred, y_true))
```
출력
```
good MSE: 0.020000000000000035
bad MSE: 9.666666666666666
```

MSE에서는 큰 오차가 제곱되므로 더 강하게 반영된다

```
오차 1 -> 제곱 오차 1
오차 2 -> 제곱 오차 4
오차 4 -> 제곱 오차 16
```



MSE는 회귀에서 자주 사용하지만, 이상치가 많으면 큰 오차에 지나치게 영향을 받을 수 있다. 

## 4. BCE 계산 원리

이진 분류에서는 정답이 `0` 또는 `1`입니다.

모델의 확률 예측 `p`

$$
\operatorname{BCE}=-\left[y\log(p)+(1-y)\log(1-p)\right]
$$

- 정답이 1이면 `log(p)`가 중요하다
- 정답이 0이면 `log(1-p)`가 중요하다

### 확률 입력을 받는 BCE

```python
# [실습 목적] 확률 입력을 받는 BCE에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def binary_cross_entropy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12,
) -> float:
    # 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # 예측과 비교할 정답값을 모은 배열입니다.
    targets = np.asarray(targets, dtype=np.float64)

    if probabilities.shape != targets.shape:
        raise ValueError("probabilities와 targets의 shape이 같아야 합니다.")

    safe_probs = np.clip(probabilities, eps, 1.0 - eps)
    # 샘플마다 계산된 손실값을 보관합니다.
    losses = -(
        targets * np.log(safe_probs)
        + (1.0 - targets) * np.log(1.0 - safe_probs)
    )
    return float(np.mean(losses))
```

```python
# [실습 목적] 확률 입력을 받는 BCE에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 예측과 비교할 정답값을 모은 배열입니다.
targets = np.array([1.0, 0.0, 1.0, 0.0])
good_probs = np.array([0.9, 0.1, 0.8, 0.2])
bad_probs = np.array([0.1, 0.9, 0.3, 0.8])

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("good BCE:", binary_cross_entropy(good_probs, targets))
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("bad BCE:", binary_cross_entropy(bad_probs, targets))
```

출력

```
good BCE: 0.164252033486018
bad BCE: 1.854645225687032
```


### 왜 `clip`이 필요할까

`log(0)`은 정의되지 않는다.

모델이 정확히 0 또는 1을 출력했다고 가정하면 계산이 `inf`가 될 수 있습니다.

```python
# [실습 목적] 왜 `clip`이 필요할까요?에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(np.log(0.0))  # -inf와 경고가 발생할 수 있습니다.
```

`np.clip`으로 매우 작은 양수와 1보다 조금 작은 값 사이로 제한하면 기본적인 수치 오류를 피할 수 있다.

BCE 곡선의 직관

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_bce_curve.png' | relative_url }" alt="02_bce_curve.png" loading="lazy">

실무 프레임워크에서는 보통 확률이 아니라 **logits를 직접 받는 BCEWithLogitsLoss**를 사용한다.

Sigmoid와 BCE를 수치적으로 안정적인 방식으로 묶어 주기 때문입니다


## 5. Cross Entropy 계산 원리

다중 분류에서는 모델이 클래스별 logits를 출력하고, 정답은 보통 클래스 인덱스로 주어집니다.

```python
# [실습 목적] 5. Cross Entropy 계산 원리에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([
    [2.0, 1.0, -0.5],  # 샘플 1
    [0.2, 0.3, 1.5],   # 샘플 2
])

# 예측과 비교할 정답값을 모은 배열입니다.
targets = np.array([0, 2])
```

- 샘플 1의 정답 클래스: 0
- 샘플 2의 정답 클래스: 2

Cross Entropy는 정답 클래스의 log-probability에 마이너스를 붙입니다.

$$
L_i=-\log p_{i,y_i}
$$

```python
# [실습 목적] 5. Cross Entropy 계산 원리에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 모델에 입력할 교육용 데이터입니다.
    x = np.asarray(x, dtype=np.float64)
    max_value = np.max(x, axis=axis, keepdims=True)
    # 수치 안정성을 위해 각 행의 최댓값을 뺀 logits입니다.
    shifted = x - max_value
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp

def cross_entropy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    # 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
    logits = np.asarray(logits, dtype=np.float64)
    # 예측과 비교할 정답값을 모은 배열입니다.
    targets = np.asarray(targets, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError("logits shape은 (batch, classes)여야 합니다.")
    if targets.shape != (logits.shape[0],):
        raise ValueError("targets shape은 (batch,)여야 합니다.")

    # 확률에 로그를 취한 log-probability입니다.
    log_probs = stable_log_softmax(logits, axis=-1)
    sample_indices = np.arange(logits.shape[0])
    correct_log_probs = log_probs[sample_indices, targets]
    sample_losses = -correct_log_probs
    return float(np.mean(sample_losses))
```

출력

```python
# [실습 목적] 5. Cross Entropy 계산 원리에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
loss = cross_entropy_from_logits(logits, targets)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("cross entropy:", loss)
```

### 중간값을 확인

```python
# [실습 목적] 중간값을 확인합니다에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 확률에 로그를 취한 log-probability입니다.
log_probs = stable_log_softmax(logits, axis=-1)
correct_log_probs = log_probs[np.arange(2), targets]
sample_losses = -correct_log_probs

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("log probabilities:\n", log_probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("correct class log probabilities:", correct_log_probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("sample losses:", sample_losses)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("mean loss:", sample_losses.mean())
```

Cross Entropy를 수동으로 계산하는 흐름

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_ce_manual_steps.png' | relative_url }" alt="02_ce_manual_steps.png" loading="lazy">

정답 클래스의 확률이 높을수록 정답 클래스 log-probability는 0에 가까워지고, loss도 작아진다

### one-hot과 클래스 인덱스

```
클래스 인덱스: [0, 2]
one-hot:
[[1, 0, 0],
 [0, 0, 1]]
```

프레임워크의 일반적인 다중 분류 loss도 클래스 인덱스를 받는 경우가 많다.

## 6. 정답 표현과 입력 shape

손실 함수에서 가장 자주 발생하는 오류는 값보다 **shape과 정답 표현**에서 나온다

| 손실 | 예측 shape | 정답 shape | 주의점 |
| --- | --- | --- | --- |
| MSE | `[B]`, `[B, 1]` 등 | 예측과 동일 | 불필요한 broadcasting을 피합니다. |
| BCE | `[B]` 또는 `[B, 1]` | 예측과 동일 | 확률 입력인지 logit 입력인지 구분합니다. |
| Cross Entropy | `[B, C]` logits | `[B]` class index | 정답은 보통 `0~C-1` 정수입니다. |

one-hot label은 클래스마다 0과 1을 적은 벡터이고, class index는 정답 클래스의 번호 하나다

두 표현은 같은 정보를 담을 수 있지만 사용하는 API의 기대 형식이 다르다

Cross Entropy 수동 계산 단계

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_ce_manual_steps.png' | relative_url }" alt="02_ce_manual_steps.png" loading="lazy">

## 7. 샘플별 loss와 배치 reduction

loss 함수는 샘플마다 하나의 loss를 먼저 만들고, 이를 평균 내는 경우가 많다

```python
# [실습 목적] 7. 샘플별 loss와 배치 reduction에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

sample_losses = np.array([0.2, 0.8, 0.5])

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("none:", sample_losses)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("mean:", sample_losses.mean())
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("sum:", sample_losses.sum())
```

reduction 방식과 shape

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_reduction_shapes.png' | relative_url }" alt="02_reduction_shapes.png" loading="lazy">

- `none`: 샘플별 loss를 그대로 유지한다
- `mean`: 평균 하나로 줄인다
- `sum`: 합 하나로 줄인다

학습률과 배치 크기를 비교할 때는 어떤 reduction을 사용했는지 확인해야 한다

이번 장의 기본 함수는 `mean`을 사용한다

reduction = 여러 loss 값을 어떻게 줄여서 정리할지

## 8. 연습 문제: 손실 함수 선택과 직접 계산

```python
# [실습 목적] 8. 연습 문제: 손실 함수 선택과 직접 계산에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 정답값 또는 정답 레이블입니다.
y_true = np.array([3.0, 5.0])
# 모델이 예측한 값입니다.
y_pred = np.array([2.0, 7.0])

#TODO: 각 샘플의 오차를 구합니다.
errors = __________________________

#TODO: 오차를 제곱합니다.
squared_errors = __________________

#TODO: 제곱 오차의 평균을 구합니다.
mse = ______________________________

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('errors:', errors)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('squared errors:', squared_errors)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('MSE:', mse)
```

정답과 해설

1. 영화 평점처럼 연속적인 실수를 예측하므로 MSE를 먼저 고려할 수 있다

2. 두 클래스 중 하나를 고르는 문제이므로 BCE 또는 BCE-with-logits가 적합하다

3. 세 클래스 중 하나를 고르는 다중 분류이므로 Cross Entropy가 적합하다

```python
# [실습 목적] 8. 연습 문제: 손실 함수 선택과 직접 계산에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 정답값 또는 정답 레이블입니다.
y_true = np.array([3.0, 5.0])
# 모델이 예측한 값입니다.
y_pred = np.array([2.0, 7.0])

# 예측에서 정답을 빼 각 샘플의 오차를 구합니다.
errors = y_pred - y_true

# 양수/음수 오차가 서로 상쇄되지 않도록 제곱합니다.
squared_errors = errors ** 2

# 배치 전체의 대표 손실로 평균을 사용합니다.
mse = np.mean(squared_errors)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('errors:', errors)                 # [-1.  2.]
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('squared errors:', squared_errors) # [1. 4.]

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('MSE:', mse)                       # 2.5
```

## 9. 연습 문제: 구현 검증과 오류 수정

다음 Cross Entropy 함수의 오류를 찾아 수정하세요

```python
# [실습 목적] 9. 연습 문제: 구현 검증과 오류 수정에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def cross_entropy_from_logits(logits, target_index):
    # 잘못된 구현: 큰 logits에서 overflow가 날 수 있습니다.
    shifted = np.max(logits)

    exp_values = np.exp(logits)
    # 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
    probabilities = exp_values / exp_values.sum()
    return -np.log(probabilities[target_index])

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([1000.0, 1001.0, 1002.0])
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(cross_entropy_from_logits(logits, target_index=2))
```

정답

```
# [실습 목적] 9. 연습 문제: 구현 검증과 오류 수정에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def cross_entropy_from_logits(logits, target_index):
    # 입력을 float64 배열로 통일해 계산합니다.
    logits = np.asarray(logits, dtype=np.float64)

    # 가장 큰 logit을 빼 exp overflow를 막습니다.
    shifted = logits - np.max(logits)

    # 안정화된 logits에서 softmax 확률을 계산합니다.
    exp_values = np.exp(shifted)
    # 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
    probabilities = exp_values / exp_values.sum()

    # log(0)을 피하기 위해 아주 작은 값을 더합니다.
    epsilon = 1e-12
    target_probability = probabilities[target_index]
    return -np.log(target_probability + epsilon)

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([1000.0, 1001.0, 1002.0])
# 현재 예측이 정답과 얼마나 다른지를 나타내는 스칼라 손실값입니다.
loss = cross_entropy_from_logits(logits, target_index=2)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('loss:', loss)

# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.isfinite(loss)

```

정답 클래스의 logit이 커지면 그 클래스 확률이 높아지고, `-log(p)`는 작아집니다.

## 10. 실제 모델 학습과의 연결

실제 프레임워크에서는 수치 안정성과 미분을 함께 처리하는 전용 손실 함수를 사용한다

```
회귀        -> MSELoss
이진 분류   -> BCEWithLogitsLoss
다중 분류   -> CrossEntropyLoss
```

특히 `BCEWithLogitsLoss`와 `CrossEntropyLoss`는 **logits를 직접 입력**으로 받는다

학습 코드에서 Sigmoid나 Softmax를 먼저 적용해야 하는지 공식 문서를 확인해야한다


## 12. 이해도 점검

1. MSE가 오차를 제곱하는 이유 두 가지를 적으세요.

2. BCE에서 `log(0)`을 피하기 위해 사용하는 대표적인 방법은 무엇인가

3. 다중 분류 logits shape이 `[B, C]`일 때 class index 정답 shape은 보통 무엇인가

4. 정답 클래스 확률이 높아지면 Cross Entropy는 어떻게 변하나

5. `reduction="none"`은 어떤 결과를 반환하나

정답 확인

1. 양수/음수 오차가 상쇄되는 것을 막고, 큰 오차에 더 큰 패널티를 주기 위해서다.

2. 확률을 `[epsilon, 1-epsilon]` 범위로 clipping하거나, logits 기반의 안정적인 전용 함수를 사용한다

3. 샘플마다 정답 클래스 번호 하나를 가지므로 `[B]`입니다.

4. 정답 확률이 높아질수록 `log(p)`가 0에 가까워지고, 그에 따라 loss(=-log(p))가 감소한다

5. 배치 평균이나 합을 내지 않고 샘플별 loss를 그대로 반환한다

