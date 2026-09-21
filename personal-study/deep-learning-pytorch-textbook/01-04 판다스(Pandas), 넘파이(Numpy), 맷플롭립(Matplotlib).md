---
title: 01-04 판다스(Pandas), 넘파이(Numpy), 맷플롭립(Matplotlib)
date: 2026-09-21
updated: 2026-09-21
description: 딥 러닝 파이토치 교과서 - 입문부터 LLM 파인튜닝까지 '01-04 판다스(Pandas), 넘파이(Numpy), 맷플롭립(Matplotlib)' 정리
---

# Pandas, NumPy, Matplotlib

데이터 분석에서 주로 사용하는 대표적인 Python 패키지는 다음과 같다.

- **Pandas**: 데이터 처리
- **NumPy**: 수치 및 배열 연산
- **Matplotlib**: 데이터 시각화

---

# 1. Pandas

Pandas는 **Python에서 데이터를 처리하고 분석하기 위한 라이브러리**이다.

```python
import pandas as pd
```

Pandas는 다음과 같은 데이터 구조를 제공한다.

- Series
- DataFrame
- Panel

이 중 주로 **Series와 DataFrame**을 사용한다.

---

## 1) Series

Series는 **1차원 배열의 값(values)에 각각 인덱스(index)를 붙인 구조**이다.

```python
sr = pd.Series(
    [17000, 18000, 1000, 5000],
    index=["피자", "치킨", "콜라", "맥주"]
)
```

구조:

```text
인덱스     값
피자     17000
치킨     18000
콜라      1000
맥주      5000
```

값과 인덱스는 다음과 같이 확인한다.

```python
sr.values
sr.index
```

---

## 2) DataFrame

DataFrame은 **행과 열을 가지는 2차원 자료구조**이다.

구성 요소:

- `index`: 행 이름
- `columns`: 열 이름
- `values`: 실제 값

```python
values = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

index = ['one', 'two', 'three']
columns = ['A', 'B', 'C']

df = pd.DataFrame(
    values,
    index=index,
    columns=columns
)
```

결과:

```text
       A  B  C
one    1  2  3
two    4  5  6
three  7  8  9
```

각 구성 요소는 다음과 같이 확인한다.

```python
df.index
df.columns
df.values
```

---

## 3) DataFrame 생성

DataFrame은 여러 데이터 구조로 만들 수 있다.

- List
- Series
- Dictionary
- NumPy ndarray
- 다른 DataFrame

### 리스트로 생성

```python
data = [
    ['1000', 'Steve', 90.72],
    ['1001', 'James', 78.09],
    ['1002', 'Doyeon', 98.43]
]

df = pd.DataFrame(
    data,
    columns=['학번', '이름', '점수']
)
```

### 딕셔너리로 생성

```python
data = {
    '학번': ['1000', '1001', '1002'],
    '이름': ['Steve', 'James', 'Doyeon'],
    '점수': [90.72, 78.09, 98.43]
}

df = pd.DataFrame(data)
```

---

## 4) DataFrame 조회

```python
df.head(n)
```

앞부분을 `n`개 확인한다.

```python
df.tail(n)
```

뒷부분을 `n`개 확인한다.

```python
df['열이름']
```

특정 열을 확인한다.

예:

```python
df.head(3)
df.tail(3)
df['학번']
```

---

## 5) 외부 데이터 읽기

Pandas는 다양한 데이터 파일을 읽어 DataFrame으로 만들 수 있다.

- CSV
- Text
- Excel
- SQL
- HTML
- JSON

CSV 파일은 `read_csv()`를 사용한다.

```python
df = pd.read_csv('example.csv')
```

별도로 인덱스를 지정하지 않으면 인덱스가 자동으로 생성된다.

---

# 2. NumPy

NumPy는 **수치 데이터를 다루는 Python 패키지**이다.

핵심 자료구조는 **ndarray**이며 벡터와 행렬을 사용하는 계산에 주로 사용된다.

```python
import numpy as np
```

---

## 1) np.array()

`np.array()`는 리스트나 튜플 등을 이용하여 ndarray를 만든다.

### 1차원 배열

```python
vec = np.array([1, 2, 3, 4, 5])
```

```text
[1 2 3 4 5]
```

### 2차원 배열

```python
mat = np.array([
    [10, 20, 30],
    [60, 70, 80]
])
```

```text
[[10 20 30]
 [60 70 80]]
```

NumPy 배열의 타입은 다음과 같다.

```python
numpy.ndarray
```

---

## 2) ndim과 shape

NumPy 배열에는 **축의 개수(ndim)**와 **크기(shape)**가 있다.

```python
vec.ndim
vec.shape
```

1차원 배열:

```text
ndim  = 1
shape = (5,)
```

2차원 배열:

```python
mat.ndim
mat.shape
```

```text
ndim  = 2
shape = (2, 3)
```

배열의 크기를 이해하는 것은 딥러닝에서도 중요하다.

---

## 3) ndarray 초기화

### np.zeros()

모든 값을 `0`으로 만든다.

```python
np.zeros((2, 3))
```

### np.ones()

모든 값을 `1`로 만든다.

```python
np.ones((2, 3))
```

### np.full()

모든 값을 지정한 값으로 만든다.

```python
np.full((2, 2), 7)
```

### np.eye()

대각선은 `1`, 나머지는 `0`인 2차원 배열을 만든다.

```python
np.eye(3)
```

### np.random.random()

임의의 값으로 채워진 배열을 만든다.

```python
np.random.random((2, 2))
```

---

## 4) np.arange()

`np.arange(n)`은 `0`부터 `n-1`까지의 배열을 만든다.

```python
np.arange(10)
```

결과:

```text
[0 1 2 3 4 5 6 7 8 9]
```

시작값, 끝값, 증가값도 지정할 수 있다.

```python
np.arange(1, 10, 2)
```

결과:

```text
[1 3 5 7 9]
```

---

## 5) np.reshape()

`reshape()`은 **데이터는 그대로 유지하면서 배열의 구조를 변경**한다.

```python
np.array(np.arange(30)).reshape((5, 6))
```

30개의 값을 `5행 6열` 구조로 변경한다.

---

## 6) NumPy 슬라이싱

슬라이싱을 사용하면 배열의 특정 행이나 열을 가져올 수 있다.

```python
mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
```

첫 번째 행:

```python
mat[0, :]
```

결과:

```text
[1 2 3]
```

두 번째 열:

```python
mat[:, 1]
```

결과:

```text
[2 5]
```

---

## 7) NumPy 정수 인덱싱

정수 인덱싱을 사용하면 **원하는 위치의 원소를 선택**할 수 있다.

```python
mat = np.array([
    [1, 2],
    [4, 5],
    [7, 8]
])
```

특정 값:

```python
mat[1, 0]
```

결과:

```text
4
```

여러 위치를 선택할 수도 있다.

```python
mat[[2, 1], [0, 1]]
```

결과:

```text
[7 5]
```

---

## 8) NumPy 연산

배열끼리 다음과 같은 연산을 수행할 수 있다.

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
```

### 덧셈

```python
x + y
```

또는

```python
np.add(x, y)
```

### 뺄셈

```python
x - y
```

또는

```python
np.subtract(x, y)
```

### 곱셈

```python
x * y
```

또는

```python
np.multiply(x, y)
```

### 나눗셈

```python
x / y
```

또는

```python
np.divide(x, y)
```

`*` 연산은 **요소별 곱셈**이다.

벡터·행렬곱 또는 행렬곱에는 `dot()`을 사용한다.

```python
np.dot(mat1, mat2)
```

---

# 3. Matplotlib

Matplotlib은 **데이터를 차트나 플롯으로 시각화하는 패키지**이다.

데이터를 이해하거나 분석 결과를 시각화할 때 사용한다.

주로 `pyplot`을 `plt`라는 이름으로 불러온다.

```python
import matplotlib.pyplot as plt
```

---

## 1) 라인 플롯

`plot()`을 사용하여 라인 그래프를 그릴 수 있다.

```python
plt.title('test')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.show()
```

- `title()`: 그래프 제목
- `plot()`: 그래프 생성
- `show()`: 그래프 출력

---

## 2) 축 이름 설정

```python
plt.xlabel('hours')
plt.ylabel('score')
```

- `xlabel()`: x축 이름
- `ylabel()`: y축 이름

예:

```python
plt.title('test')
plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.xlabel('hours')
plt.ylabel('score')
plt.show()
```

---

## 3) 여러 라인과 범례

하나의 그래프에 여러 개의 `plot()`을 사용할 수 있다.

각 선이 무엇을 의미하는지 표시하려면 `legend()`를 사용한다.

```python
plt.title('students')

plt.plot([1, 2, 3, 4], [2, 4, 8, 6])
plt.plot([1.5, 2.5, 3.5, 4.5], [3, 5, 8, 10])

plt.xlabel('hours')
plt.ylabel('score')

plt.legend(['A student', 'B student'])

plt.show()
```

---

# 핵심 정리

| 라이브러리 | 주요 역할 |
|---|---|
| Pandas | 데이터 처리 및 DataFrame 관리 |
| NumPy | 배열과 수치 연산 |
| Matplotlib | 데이터 시각화 |

```text
Pandas
→ 데이터를 읽고 정리

NumPy
→ 배열과 수치를 계산

Matplotlib
→ 데이터를 그래프로 표현
```