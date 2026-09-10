---
title: 1장 1강 logits softmax log-softmax와 수치 안정성
date: 2026-09-10
updated: 2026-09-10
description: KANT 강의 '1장 1강 logits softmax log-softmax와 수치 안정성' 정리
---

## 1. logit과 softmax는 무엇인가

고양이/강아지/토끼를 구분하는 모델이 다음 값을 출력했다고 가정

```python
# [실습 목적] 1. logit과 softmax는 무엇인가요?에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = [2.0, 1.0, -0.5]
```

이 값은 세 클래스에 대한 **원시 점수(raw score)**다.

- 첫 번째 값이 가장 크므로 모델은 첫 번째 클래스를 가장 선호한다
- 세 값의 합은 1이 아니다
- 음수가 포함될 수 있다.
- 따라서 아직 확률이라고 부를 수 없다.

logits에서 확률과 loss로 이어지는 흐름

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_logits_softmax_flow.png' | relative_url }}" alt="01_logits_softmax_flow.png" loading="lazy">

softmax는 원시 점수를 다음 조건을 만족하는 값으로 바꾼다

1. 모든 값이 0 이상이다.
2. 한 샘플 안에서 모든 값의 합이 1이다.
3. 더 큰 logit에는 더 큰 확률이 배정된다.

logit은 모델의 비교 점수이고, softmax 확률은 그 점수를 한 샘플 안에서 해석하기 쉽게 변환한 값이다

## 2. logits에서 확률로 가는 큰 그림

분류 모델의 마지막 선형층은 보통 클래스마다 하나의 점수를 출력한다

이 점수가 **logit**입니다. softmax는 logits를 서로 비교할 수 있는 확률 분포로 바꾼다


<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_logits_softmax_flow.png' | relative_url }}" alt="01_logits_softmax_flow.png" loading="lazy">

```
입력 데이터
  -> 모델의 마지막 선형층
  -> logits [클래스별 원시 점수]
  -> softmax
  -> probability [합이 1인 분포]
  -> 정답과 비교
  -> loss
```

이 흐름에서 softmax는 모델을 학습시키는 별도의 층이라기보다, **출력 점수를 해석 가능한 분포로 바꾸는 계산**이다. 

## 3. softmax를 한 단계씩 계산하기


softmax 수식은 다음과 같다.

$$
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

수식을 코드로 옮길 때는 세 단계로 나누면 된다

1. 각 logit에 지수 함수를 적용한다
2. 지수값을 모두 더한다
3. 각 지수값을 전체 합으로 나눈다

```python
# [실습 목적] 3. softmax를 한 단계씩 계산하기에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([2.0, 1.0, -0.5])

# 각 logit에 지수 함수를 적용해 모든 값을 양수로 바꿉니다.
exp_values = np.exp(logits)
# 지수값의 합을 구해 정규화 분모로 사용합니다.
exp_sum = np.sum(exp_values)
# 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
probabilities = exp_values / exp_sum

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("logits:", logits)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("exp(logits):", exp_values)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("sum(exp):", exp_sum)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("probabilities:", probabilities)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("probability sum:", probabilities.sum())
```

출력

```
logits: [ 2.   1.  -0.5]
exp(logits): [7.3890561  2.71828183 0.60653066]
sum(exp): 10.71386859
probabilities: [0.68967209 0.25371618 0.05661173]
probability sum: 1.0
```

첫 번째 logit이 가장 크므로 첫 번째 확률도 가장 크다
<br> 중요한 점은 **크기 순서는 유지되지만 값의 범위와 합이 바뀌었다는 것**이다



### 가장 먼저 할 검증

```python
# [실습 목적] 가장 먼저 할 검증에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert probabilities.shape == logits.shape
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.all(probabilities >= 0)
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.all(probabilities <= 1)
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.isclose(probabilities.sum(), 1.0)
```

`assert`는 "이 조건을 반드시 만족해야 한다"는 검사용 문장입니다.

 계산이 틀리면 그 자리에서 오류를 발생시킨다


## 4. 수치 안정성이 필요한 이유

지수 함수는 입력이 조금만 커져도 매우 빠르게 증가합니다.

```python
# [실습 목적] 4. 수치 안정성이 필요한 이유에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

large_logits = np.array([1000.0, 1001.0, 1002.0])

with np.errstate(over="ignore", invalid="ignore"):
    # 각 logit에 지수 함수를 적용해 모든 값을 양수로 바꿉니다.
    exp_values = np.exp(large_logits)
    naive_probs = exp_values / exp_values.sum()

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("exp values:", exp_values)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("naive probabilities:", naive_probs)
```

출력

```
exp values: [inf inf inf]
naive probabilities: [nan nan nan]
```

- `inf`: 컴퓨터가 표현할 수 있는 범위를 넘어선 값
- `nan`: 계산 결과를 숫자로 정의할 수 없는 상태

여기서는 `inf / inf`가 되어 `nan`이 발생한다

naive softmax와 stable softmax

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_softmax_stability.png' | relative_url }}" alt="01_softmax_stability.png" loading="lazy">

이 문제는 딥러닝 모델에서 실제로 중요하다

모델이 어떤 클래스를 강하게 선호하면 logits의 크기가 커질 수 있기 때문이다

softmax는 모든 logits에 같은 값을 더하거나 빼도 결과가 바뀌지 않는다
<br>
따라서 각 행에서 가장 큰 값을 뺀다

```python
# [실습 목적] 4. 수치 안정성이 필요한 이유에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 수치 안정성을 위해 각 행의 최댓값을 뺀 logits입니다.
shifted = large_logits - np.max(large_logits)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(shifted)  # [-2. -1.  0.]
```

가장 큰 값이 0이 되었고, 나머지는 0 이하가 된다

이제 가장 큰 지수값은 `exp(0) = 1`이므로 overflow를 피할 수 있다.



```python
# [실습 목적] 4. 수치 안정성이 필요한 이유에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 각 logit에 지수 함수를 적용해 모든 값을 양수로 바꿉니다.
exp_values = np.exp(shifted)
stable_probs = exp_values / exp_values.sum()
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(stable_probs)
```

출력

```
[0.09003057 0.24472847 0.66524096]
```


### 재사용 가능한 함수

```python
# [실습 목적] 재사용 가능한 함수에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 모델에 입력할 교육용 데이터입니다.
    x = np.asarray(x, dtype=np.float64)
    # 수치 안정성을 위해 각 행의 최댓값을 뺀 logits입니다.
    shifted = x - np.max(x, axis=axis, keepdims=True)
    # 각 logit에 지수 함수를 적용해 모든 값을 양수로 바꿉니다.
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)
```

검증

```python
# [실습 목적] 재사용 가능한 함수에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

small = np.array([2.0, 1.0, -0.5])
large = np.array([1000.0, 1001.0, 1002.0])

small_probs = stable_softmax(small)
large_probs = stable_softmax(large)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(small_probs, small_probs.sum())
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(large_probs, large_probs.sum())

# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.isclose(small_probs.sum(), 1.0)
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.isclose(large_probs.sum(), 1.0)
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.all(np.isfinite(large_probs))
```

## 5. stable softmax의 핵심 구성요소

stable softmax는 복잡한 기술이 아니다. 

네 요소만 순서대로 확인

| 구성요소 | 역할 | 확인할 값 |
| --- | --- | --- |
| `np.max(..., keepdims=True)` | 각 샘플의 가장 큰 logit을 찾습니다. | 결과가 나눗셈 가능한 shape인지 확인합니다. |
| `shifted = x - max_value` | 가장 큰 값을 0으로 이동합니다. | 모든 값이 0 이하인지 확인합니다. |
| `np.exp(shifted)` | 점수 차이를 양수 비율로 바꿉니다. | `inf`가 없는지 확인합니다. |
| 합으로 나누기 | 합이 1인 분포를 만듭니다. | `sum(axis=-1) == 1`인지 확인합니다. |

naive softmax와 stable softmax

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_softmax_stability.png' | relative_url }}" alt="01_softmax_stability.png" loading="lazy">

## 6. 배치 입력과 axis 해석

샘플 두 개와 클래스 세 개가 들어 있는 배열을 보겠습니다.

```python
# [실습 목적] 6. 배치 입력과 axis 해석에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 여러 샘플의 클래스별 logits를 한 배열에 모읍니다.
batch_logits = np.array([
    [2.0, 1.0, -0.5],
    [0.1, 0.1, 0.1],
])

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(batch_logits.shape)  # (2, 3)
```

- 첫 번째 축: 샘플 2개
- 마지막 축: 클래스 3개

각 샘플마다 확률 합이 1이 되어야 하므로 **마지막 축**을 따라 계산

```python
# [실습 목적] 6. 배치 입력과 axis 해석에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 각 logit에 지수 함수를 적용해 모든 값을 양수로 바꿉니다.
exp_values = np.exp(batch_logits)
row_sums = np.sum(exp_values, axis=-1, keepdims=True)
# 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
probabilities = exp_values / row_sums

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(probabilities)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("row sums:", probabilities.sum(axis=-1))
```

`keepdims=True`를 사용하면 `row_sums`의 shape이 `(2, 1)`로 유지된다

그러면 `(2, 3)` 배열과 나눗셈할 때 각 행의 합이 자동으로 적용된다


softmax 축 선택 비교

<img src="{{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_axis_comparison.png' | relative_url }}" alt="01_axis_comparison.png" loading="lazy">


`axis=0`으로 계산하면 각 클래스의 배치 방향 합이 1이 된다.

코드는 실행되지만 원하는 의미는 아니다.

"실행된다"와 "올바르게 계산된다"는 다르다

## 7. log-softmax 출력 해석

Cross Entropy는 정답 클래스의 확률에 로그를 취한다

따라서 softmax 뒤에 `np.log`를 적용할 수 있습니다.

```python
# [실습 목적] 7. log-softmax 출력 해석에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# softmax로 변환한 확률값을 저장합니다.
probs = stable_softmax(np.array([2.0, 1.0, -0.5]))
# 확률에 로그를 취한 log-probability입니다.
log_probs = np.log(probs)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print(log_probs)
```

하지만 실제 구현에서는 softmax와 log를 따로 계산하기보다 안정적인 형태로 묶어서 계산한다

$$
\log \operatorname{softmax}(z_i)
= z_i - \log\left(\sum_j e^{z_j}\right)
$$

max trick을 적용한 간단한 구현

```python
# [실습 목적] 7. log-softmax 출력 해석에서 설명한 흐름을 코드로 확인합니다.
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
```

softmax와 일관된지 확인합니다.

```python
# [실습 목적] 7. log-softmax 출력 해석에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([2.0, 1.0, -0.5])
# softmax로 변환한 확률값을 저장합니다.
probs = stable_softmax(logits)
# 확률에 로그를 취한 log-probability입니다.
log_probs = stable_log_softmax(logits)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("softmax:", probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print("exp(log-softmax):", np.exp(log_probs))

# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.allclose(np.exp(log_probs), probs)
```

### log 값이 음수인 이유

확률은 0과 1 사이입니다. 1보다 작은 양수에 로그를 취하면 0보다 작은 값이 된다

```
확률 1.0  -> log = 0
확률 0.5  -> log ≈ -0.693
확률 0.1  -> log ≈ -2.303
```

확률이 작을수록 log-probability는 더 작은 음수가 된다.

## 8. 연습 문제: stable softmax 구현

```python
# [실습 목적] 8. 연습 문제: stable softmax 구현에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # 입력을 부동소수점 NumPy 배열로 통일합니다.
    x = np.asarray(x, dtype=np.float64)

    #TODO 1: 지정한 축에서 최댓값을 구합니다.
    max_value = ______________________________

    #TODO 2: 모든 logit에서 최댓값을 뺍니다.
    shifted = ________________________________

    #TODO 3: 지수화한 뒤 같은 축의 합으로 나눕니다.
    exp_values = _____________________________
    # 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
    probabilities = __________________________
    return probabilities

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([1000.0, 1001.0, 1002.0])
# 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
probabilities = stable_softmax(logits)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('probabilities:', probabilities)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('sum:', probabilities.sum())
```

확인할 조건은 두 가지입니다.

1. 결과에 `nan` 또는 `inf`가 없어야 합니다.
2. 확률의 합이 1이어야 합니다.
    
```python
# [실습 목적] 8. 연습 문제: stable softmax 구현에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
   # 계산 중 정수 나눗셈이나 낮은 정밀도 문제가 생기지 않도록 dtype을 통일합니다.
   x = np.asarray(x, dtype=np.float64)

   # 각 샘플의 최댓값을 유지된 차원으로 구합니다.
   # keepdims=True 덕분에 원본 배열과 바로 뺄 수 있습니다.
   max_value = np.max(x, axis=axis, keepdims=True)

   # 가장 큰 logit이 0이 되도록 이동해 exp overflow를 막습니다.
   shifted = x - max_value

   # 이동된 값에 exp를 적용하고, 같은 축의 합으로 나눕니다.
   exp_values = np.exp(shifted)
   # 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
   probabilities = exp_values / np.sum(
      exp_values,
      axis=axis,
      keepdims=True,
   )
   return probabilities

# 모델이 출력한 정규화 전 점수입니다. 아직 확률이 아닙니다.
logits = np.array([1000.0, 1001.0, 1002.0])
# 각 값을 전체 합으로 나누어 합이 1인 확률 분포를 만듭니다.
probabilities = stable_softmax(logits)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('probabilities:', probabilities)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('sum:', probabilities.sum())

# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.all(np.isfinite(probabilities))
# 구현이 반드시 만족해야 하는 조건을 자동으로 검증합니다.
assert np.isclose(probabilities.sum(), 1.0)
```

최댓값을 빼면 `[1000, 1001, 1002]`가 `[-2, -1, 0]`이 된다

상대적인 차이는 그대로이므로 softmax 결과는 바뀌지 않고 overflow만 피한다



## 9. 연습 문제: 축과 logit 간격 비교

다음 배열을 사용해 `axis=-1`과 `axis=0`의 결과를 비교

```python
# [실습 목적] 9. 연습 문제: 축과 logit 간격 비교에서 설명한 흐름을 코드로 확인합니다.
# [실행 방법] 위에서 아래로 실행하고, print 출력과 assert 통과 여부를 함께 확인하세요.
# [확인 포인트] 값뿐 아니라 배열의 shape, 합계, 범위가 본문 설명과 일치하는지 살펴봅니다.

# NumPy는 벡터/행렬 계산과 수치 검증에 사용합니다.
import numpy as np

# 여러 샘플의 클래스별 logits를 한 배열에 모읍니다.
batch_logits = np.array([
    [2.0, 1.0, 0.0],
    [0.0, 1.0, 2.0],
])

row_probs = stable_softmax(batch_logits, axis=-1)
column_probs = stable_softmax(batch_logits, axis=0)

# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('axis=-1 결과:\n', row_probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('각 샘플의 합:', row_probs.sum(axis=-1))
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('axis=0 결과:\n', column_probs)
# 중간 결과를 출력해 계산 흐름을 사람이 직접 확인합니다.
print('각 열의 합:', column_probs.sum(axis=0))
```

1. 샘플별 클래스 확률을 구하려면 어느 축이 맞나요?

2. logits가 `[1, 1, 1]`이면 각 클래스 확률은 얼마인가요?

3. logits 간격이 커질수록 가장 큰 클래스의 확률은 어떻게 변하나요?
- 정답과 해설

1. 클래스가 마지막 축에 있으므로 `axis=-1`이 맞습니다. 각 행의 합이 1이 됩니다.

2. 세 클래스가 같은 점수를 가지므로 각각 `1/3`입니다.

3. 가장 큰 logit과 나머지 logit의 차이가 커질수록 가장 큰 클래스의 확률은 1에 가까워집니다. 다만 정확히 1이 되는 것은 아니다


## 10. 실제 학습 코드와의 연결

실제 딥러닝 프레임워크에서는 `CrossEntropyLoss`가 logits를 직접 받는 경우가 많다.

이때 사용자가 미리 softmax를 적용하면 같은 계산을 두 번 하거나 수치적으로 불리한 형태가 될 수 있다.

```
학습 시: logits -> 프레임워크의 안정적인 Cross Entropy
추론 시: logits -> softmax -> 확률 해석 또는 argmax
```

따라서 "softmax를 언제 적용하는가?"는 문제의 목적에 따라 답이 달라진다.

모델 학습 API의 입력 규칙을 먼저 확인해야한다


## 12. 이해도 점검

1. logit과 probability의 차이를 한 문장으로 설명

2. `max trick`이 softmax 결과를 바꾸지 않는 이유는 무엇인가

3. shape이 `[B, C]`인 분류 logits에서 일반적으로 softmax를 적용하는 축은 무엇인가요?

4. `log-softmax` 값이 대부분 음수인 이유는 무엇인가요?

5. 확률 합을 검사할 때 사용할 수 있는 NumPy 함수는 무엇인가요?

- 정답 확인
1. logit은 클래스별 원시 점수이고, probability는 그 점수를 정규화해 합이 1이 되도록 만든 값입니다.

2. 모든 항에 같은 상수를 더하거나 빼면 분자와 분모에 같은 배수가 생겨 약분되며, 상대적인 차이는 유지되기 때문입니다.

3. 클래스가 있는 마지막 축 `axis=-1`입니다.

4. 확률은 0과 1 사이이고, 1보다 작은 양수의 로그는 음수이기 때문입니다.

5. `np.isclose(probabilities.sum(axis=-1), 1.0)` 또는 여러 샘플에는 `np.allclose`를 사용할 수 있습니다.


max trick은 Softmax 계산 전에 가장 큰 logit 값을 모든 값에서 빼는 기법
