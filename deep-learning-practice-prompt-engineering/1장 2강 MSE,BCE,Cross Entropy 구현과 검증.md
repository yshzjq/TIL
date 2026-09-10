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


