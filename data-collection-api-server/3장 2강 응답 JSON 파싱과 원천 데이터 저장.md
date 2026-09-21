---
title: 3장 1강 API 조건 설정, pagination, rate limit 확인
date: 2026-09-21
updated: 2026-09-21
description: KANT 강의 '3장 1강 API 조건 설정, pagination, rate limit 확인' 정리
---

# 3장 1강 : API 조건 설정 / Pagination / Rate Limit 확인

## 1. Pagination이란?

API는 많은 데이터를 한 번에 전부 보내지 않고 **여러 페이지로 나누어 전달**하는 경우가 많다.

이 방식을 **Pagination(페이지네이션)**이라고 한다.

예를 들어 전체 데이터가 5건이고 한 페이지에 2건씩 가져온다면:

```text
1페이지 → 2건
2페이지 → 2건
3페이지 → 1건
```

처럼 나누어 가져온다.

기본 흐름은 다음과 같다.

```text
현재 페이지 요청
→ 데이터 저장
→ 다음 페이지가 있는지 확인
→ 있으면 page 증가
→ 다음 페이지 요청
```

---

## 2. `page`와 `page_size`

페이지 방식의 API에서는 보통 `page`와 `page_size` 같은 Query Parameter를 사용한다.

### `page`

가져올 **페이지 번호**이다.

```python
page = 1
```

API마다 시작 번호가 다를 수 있다.

- 어떤 API는 `0`부터 시작
- 어떤 API는 `1`부터 시작

따라서 **API 문서에서 시작 번호를 확인해야 한다.**

---

### `page_size`

한 페이지에서 요청할 **최대 데이터 개수**이다.

```python
page_size = 2
```

예를 들어:

```python
params = {
    "category": "news",
    "page": 1,
    "page_size": 2,
}
```

의 의미는:

```text
category = news
→ 뉴스 데이터 조회

page = 1
→ 1페이지 조회

page_size = 2
→ 한 페이지에서 최대 2건 요청
```

API마다 이름은 다를 수 있다.

```text
page_size
size
limit
per_page
numOfRows
```

따라서 실제 API 문서에서 **정확한 Parameter 이름과 최댓값**을 확인해야 한다.

---

## 3. 마지막 페이지 확인

Pagination에서는 **언제 반복을 멈출지** 알아야 한다.

API마다 마지막 페이지를 알려 주는 방법이 다르다.

### 방법 1. `has_next`

```python
{
    "items": [...],
    "has_next": False
}
```

`has_next=False`이면 다음 페이지가 없으므로 반복을 멈춘다.

---

### 방법 2. 전체 페이지 수

```python
{
    "page": 3,
    "total_pages": 3
}
```

현재 페이지와 전체 페이지가 같으면 마지막이다.

```text
page == total_pages
```

---

### 방법 3. 빈 배열

```python
{
    "items": []
}
```

빈 배열을 마지막 페이지 신호로 사용하는 API도 있다.

단, **빈 배열이 정말 마지막 페이지를 의미하는지는 API 문서에서 확인해야 한다.**

### 중요

한 API에서 종료 규칙을 임의로 섞지 않는다.

이번 실습에서는:

```text
has_next = False
```

를 마지막 페이지 신호로 사용한다.

---

## 4. 수집 조건 먼저 정하기

반복해서 API를 호출하기 전에 어떤 조건으로 수집할지 정한다.

예:

| 항목 | 값 |
| --- | --- |
| 목적 | 뉴스 데이터 수집 |
| 검색 조건 | `category=news` |
| 시작 page | `1` |
| page_size | `2` |
| 마지막 신호 | `has_next=False` |

한 번의 수집이 진행되는 동안 **검색 조건을 중간에 바꾸지 않는다.**

예를 들어:

```text
1페이지 → category=news
2페이지 → category=notice
```

처럼 바꾸면 하나의 연속된 결과라고 보기 어렵다.

`page_size` 역시 수집 중간에 바꾸지 않는다.

---

## 5. 페이지 반복 수집 흐름

Pagination의 기본 알고리즘은 다음과 같다.

```text
1. page = 1로 시작
2. 현재 page 요청
3. 성공하면 items 저장
4. 마지막 페이지인지 확인
5. 마지막이면 종료
6. 아니면 page += 1
7. 다음 page 요청
```

Python으로 표현하면:

```python
page = 1
collected = []

while True:
    result = fetch_page(page)

    collected.extend(result["items"])

    if not result["has_next"]:
        break

    page += 1
```

핵심은:

> **현재 페이지를 성공적으로 처리한 뒤에 page 번호를 올린다.**

---

## 6. 여러 페이지의 데이터 합치기

전체 결과를 저장할 빈 리스트를 만든다.

```python
collected = []
```

한 페이지에서 다음과 같은 데이터가 왔다고 하자.

```python
page_items = [
    {"source_id": "news-001"},
    {"source_id": "news-002"},
]
```

전체 목록에 이어 붙일 때는 `extend()`를 사용한다.

```python
collected.extend(page_items)
```

결과:

```python
[
    {"source_id": "news-001"},
    {"source_id": "news-002"},
]
```

### `append()`와 차이

```python
collected.append(page_items)
```

를 사용하면 리스트 전체가 하나의 값으로 들어간다.

```python
[
    [
        {"source_id": "news-001"},
        {"source_id": "news-002"}
    ]
]
```

즉:

> `append()` = 리스트 자체를 하나의 값으로 추가  
> `extend()` = 리스트 안의 항목들을 각각 이어 붙임

페이지별 데이터를 하나의 전체 목록으로 만들 때는 **`extend()`**를 사용한다.

---

## 7. Rate Limit과 `429`

API에는 일정 시간 동안 요청할 수 있는 횟수가 제한되어 있을 수 있다.

너무 많은 요청을 보내면:

```text
429 Too Many Requests
```

응답을 받을 수 있다.

`429`는 **일정 시간 동안 너무 많은 요청을 보냈다는 뜻**이다.

---

## 8. `429`를 받았을 때

기본 처리 흐름은 다음과 같다.

```text
429 발생
→ 현재 page 요청 실패
→ Retry-After 또는 API 문서 확인
→ 안내된 시간만큼 기다림
→ 같은 page 다시 요청
```

예:

```python
if status_code == 429:
    print("잠시 기다린 뒤 같은 page 재시도")
```

### 왜 같은 페이지를 다시 요청할까?

예를 들어 2페이지 요청에서 `429`가 발생했다고 하자.

이때 바로:

```text
2페이지 실패
→ 3페이지 요청
```

으로 넘어가면 **2페이지 데이터가 빠지게 된다.**

따라서:

> **실패한 페이지는 page 번호를 올리지 않고 다시 요청한다.**

---

## 9. Page 번호는 성공 후 증가

중요한 원칙은 다음과 같다.

```text
요청 성공
→ 데이터 저장
→ 다음 페이지 확인
→ page += 1
```

실패했는데도:

```python
page += 1
```

해버리면 데이터를 건너뛸 수 있다.

예를 들어 `429`나 `500`이 발생했는데 page 번호를 올리면 해당 페이지가 누락될 수 있다.

---

## 10. 반복 수집 함수

교안의 핵심 흐름은 다음과 같다.

```python
def collect_all_pages(page_size=2):
    page = 1
    collected = []
    attempts = {}
    waits = []

    while True:
        result = fetch_page(page, page_size, attempts)

        if result["status_code"] == 429:
            waits.append(result["retry_after"])
            result = fetch_page(page, page_size, attempts)

        if result["status_code"] != 200:
            raise RuntimeError(f"page{page} 수집 실패")

        collected.extend(result["items"])

        if not result["has_next"]:
            break

        page += 1

    return {
        "items": collected,
        "waits": waits
    }
```

전체 흐름은:

```text
page 요청
   ↓
429인가?
 ├─ Yes → 기다린 뒤 같은 page 재시도
 └─ No
   ↓
200 성공인가?
 ├─ No → 오류
 └─ Yes
   ↓
items를 전체 목록에 extend
   ↓
has_next 확인
 ├─ False → 종료
 └─ True
   ↓
page += 1
   ↓
다음 page 요청
```

---

## 11. Fixture로 Pagination 재현

이번 실습에서는 실제 API 대신 고정된 Fixture를 사용한다.

전체 데이터를 `ALL_ITEMS`에 미리 저장하고:

```python
start = (page - 1) * page_size
end = start + page_size

page_items = ALL_ITEMS[start:end]
```

처럼 현재 페이지에 해당하는 부분만 잘라서 반환한다.

예를 들어 `page_size=2`이면:

```text
1페이지 → [0:2]
2페이지 → [2:4]
3페이지 → [4:6]
```

범위로 데이터를 가져온다.

다음 데이터가 남아 있는지는:

```python
has_next = end < len(ALL_ITEMS)
```

으로 판단한다.

이번 Fixture에서는 **2페이지 첫 요청에만 `429`를 반환하도록 설정**되어 있다.

따라서:

```text
1페이지 성공
→ 2페이지 429
→ 같은 2페이지 재시도
→ 2페이지 성공
→ 3페이지 성공
→ 종료
```

흐름을 확인할 수 있다.

---

## 12. 자주 발생하는 실수

### 1. Page 시작 번호를 잘못 사용

API가 1부터 시작하는데 0부터 요청하면 오류나 빈 결과가 올 수 있다.

→ API 문서 확인

### 2. 실패했는데 Page 증가

`429`, `500` 등이 발생했는데 다음 페이지로 넘어가면 데이터가 누락될 수 있다.

→ 성공 후에만 증가

### 3. `page_size`를 중간에 변경

페이지 경계가 달라져 중복이나 누락이 생길 수 있다.

→ 한 번의 수집 동안 같은 값 사용

### 4. `append()`와 `extend()` 혼동

```python
append(page_items)
```

를 사용하면 중첩 리스트가 만들어질 수 있다.

→ 페이지 데이터를 합칠 때 `extend()`

### 5. 결과 건수를 확인하지 않음

수집 후 최소한 다음을 확인한다.

```python
print("건수:", len(items))
print("첫 ID:", items[0]["source_id"])
print("마지막 ID:", items[-1]["source_id"])
```

---

# 핵심 정리

- **Pagination**은 많은 데이터를 여러 페이지로 나누어 받는 방식이다.
- `page`는 가져올 페이지 번호이다.
- `page_size`는 한 페이지에서 가져올 최대 데이터 수이다.
- 시작 page와 page_size의 이름·최댓값은 API 문서에서 확인한다.
- 마지막 페이지는 `has_next`, `total_pages`, 빈 배열 등의 방식으로 확인할 수 있다.
- 페이지 데이터를 전체 목록에 붙일 때는 `extend()`를 사용한다.
- **현재 페이지를 성공적으로 처리한 뒤에만 `page += 1` 한다.**
- `429 Too Many Requests`가 발생하면 기다린 뒤 **같은 페이지를 다시 요청한다.**
- 검색 조건과 `page_size`는 한 번의 수집이 끝날 때까지 고정한다.

## 가장 중요한 흐름

```text
page = 1
   ↓
현재 page 요청
   ↓
429?
 ├─ Yes → 기다림 → 같은 page 재요청
 └─ No
   ↓
성공 확인
   ↓
items를 extend
   ↓
마지막 page?
 ├─ Yes → 종료
 └─ No
   ↓
page += 1
   ↓
다음 page 요청
```

## 가장 쉽게 기억하기

```text
page
= 몇 번째 페이지?

page_size
= 한 페이지에 몇 개?

has_next
= 다음 페이지가 있나?

429
= 너무 많이 요청함 → 기다렸다 같은 page 재시도

extend
= 현재 페이지 데이터를 전체 목록에 이어 붙이기
```