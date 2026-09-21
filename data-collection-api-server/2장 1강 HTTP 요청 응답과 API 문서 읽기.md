---
title: 2장 1강 HTTP 요청 응답과 API 문서 읽기
date: 2026-09-21
updated: 2026-09-21
description: KANT 강의 '2장 1강 HTTP 요청 응답과 API 문서 읽기' 정리
---

# 2장 1강 : HTTP 요청/응답과 API 문서 읽기

## 1. HTTP 요청과 응답

API 호출에서는 **클라이언트(Client)**가 요청을 보내고 **서버(Server)**가 요청을 처리한 뒤 응답을 반환한다.

- **클라이언트**: 필요한 데이터를 요청하는 프로그램
- **서버**: 요청을 해석하고 결과를 반환하는 프로그램

예를 들어 Python 수집 코드가 클라이언트가 되고, 공공데이터 API가 서버가 될 수 있다.

### 요청에서 확인할 것

1. 어느 주소로 요청할 것인가?
2. 어떤 HTTP Method를 사용할 것인가?
3. 어떤 조건과 인증 정보를 보낼 것인가?
4. Request Body가 필요한가?

### 응답에서 확인할 것

1. 상태 코드는 무엇인가?
2. 응답 형식은 무엇인가?
3. 응답 본문에 어떤 데이터가 있는가?

응답을 받으면 **상태 코드를 먼저 확인한 뒤 본문을 확인한다.**

---

## 2. URL과 Endpoint

URL은 인터넷에서 자원의 위치를 나타내는 주소이다.

```text
https://api.example.org/v1/books?q=python&page=1
```

| 부분 | 예시 | 역할 |
| --- | --- | --- |
| scheme | `https` | 통신 방식 |
| host | `api.example.org` | 서버 이름 |
| base path | `/v1` | API 버전 또는 공통 경로 |
| endpoint | `/books` | 특정 기능이나 자원 경로 |
| query | `q=python&page=1` | 요청 조건 |

### Base URL

여러 Endpoint가 공유하는 앞부분이다.

```python
base_url = "https://api.example.org/v1"
```

### Endpoint

특정 기능이나 자원을 나타내는 경로이다.

```python
books_endpoint = "/books"
book_detail_endpoint = "/books/{book_id}"
```

---

## 3. Path Parameter

**Path Parameter**는 URL의 **경로 안에 들어가 특정 자원을 지정하는 값**이다.

예를 들어 다음과 같은 Endpoint가 있다.

```text
/books/{book_id}
```

`{book_id}`는 실제 요청을 보낼 때 구체적인 값으로 바뀐다.

```python
book_id = 42
detail_path = f"/books/{book_id}"
```

결과:

```text
/books/42
```

즉, `42`라는 ID를 가진 특정 도서를 지정한다.

### Path Parameter와 Query Parameter 비교

```text
/books/42
```

여기서 `42`는 **Path Parameter**이다.

반면 다음처럼 URL 뒤에 조건을 붙이는 값은 **Query Parameter**이다.

```text
/books?q=python&page=1
```

| 구분 | 예시 | 역할 |
| --- | --- | --- |
| Path Parameter | `/books/42` | 특정 자원을 지정 |
| Query Parameter | `/books?page=1` | 조회 조건을 전달 |

즉,

> **Path Parameter = 무엇을 대상으로 할지 지정**  
> **Query Parameter = 어떤 조건으로 조회할지 지정**

---

## 4. HTTP Method

HTTP Method는 서버에 어떤 작업을 요청하는지 나타낸다.

| Method | 기본 의미 |
| --- | --- |
| `GET` | 자원 조회 |
| `POST` | 새 자원 생성 또는 작업 요청 |
| `PUT` | 자원 전체 교체 |
| `PATCH` | 자원 일부 수정 |
| `DELETE` | 자원 삭제 |

외부 데이터 수집에서는 주로 `GET`을 사용한다.

같은 URL이라도 Method가 다르면 다른 기능일 수 있으므로 **Endpoint와 Method를 함께 확인해야 한다.**

```python
request_plan = {
    "method": "GET",
    "endpoint": "/books",
}
```

---

## 5. Header, Query Parameter, Request Body

서버에 정보를 전달하는 대표적인 방법은 `Header`, `Query Parameter`, `Request Body`이다.

각각 역할이 다르다.

### 5.1 Header

**Header는 요청에 대한 부가 정보를 전달하는 영역**이다.

인증 정보가 들어가는 경우가 많지만, **권한이나 인증만 담당하는 것은 아니다.**

```python
headers = {
    "Accept": "application/json",
    "X-API-Key": "환경 변수에서 읽은 값",
}
```

예를 들어:

- `Authorization`, `X-API-Key`: 인증 정보
- `Accept`: 어떤 형식의 응답을 원하는지 전달

따라서 Header는 다음과 같이 이해하면 된다.

> **Header = 요청에 대한 부가 정보 + 인증 정보**

인증 정보는 로그에 그대로 출력하지 않는다.

---

### 5.2 Query Parameter

**Query Parameter는 데이터를 어떤 조건으로 조회할지 전달하는 값**이다.

```python
params = {
    "q": "파이썬",
    "page": 1,
    "limit": 10,
}
```

주로 다음과 같은 조건에 사용한다.

- 검색어
- 페이지 번호
- 정렬 조건

예를 들어:

```text
GET /books?page=2
```

여기서:

```text
page=2
```

는

> "책 목록의 2페이지를 조회해줘"

라는 **조회 조건**이다.

따라서:

> **Query Parameter = 조회 조건**

---

### 5.3 Request Body

**Request Body는 서버에 실제로 처리해 달라고 보내는 데이터**이다.

Body를 단순히 **수정할 데이터**라고 생각하면 정확하지 않다.

수정할 때도 사용할 수 있지만, **새로운 데이터를 생성하거나 처리할 데이터를 보낼 때도 사용한다.**

예를 들어 새로운 데이터를 생성할 때:

```text
POST /books
```

```json
{
  "title": "파이썬 입문",
  "category": "education"
}
```

이 경우 Body의 의미는:

> "이 데이터를 가지고 처리해줘"

라고 볼 수 있다.

수정에서도 Body를 사용할 수 있다.

```text
PATCH /books/42
```

```json
{
  "title": "수정된 제목"
}
```

여기서는 Body에 **수정할 데이터**를 담는다.

따라서:

> **Request Body = 서버가 생성·수정·처리할 실제 데이터**

GET 요청은 일반적으로 Query Parameter로 조회 조건을 전달하고, POST 요청은 JSON Body를 사용하는 경우가 많다.

하지만 항상 그런 것은 아니므로 **실제 사용 방법은 API 문서를 확인해야 한다.**

---

### Header, Query, Body 비교

| 요소 | 쉽게 말하면 | 대표 용도 |
| --- | --- | --- |
| Header | 요청에 대한 추가 정보 | 인증, 원하는 응답 형식 |
| Path Parameter | 무엇을 대상으로 할지 | 특정 자원 지정 |
| Query Parameter | 어떤 조건으로 찾을지 | 검색, 필터, 페이지 |
| Request Body | 서버가 처리할 실제 데이터 | 생성, 수정, 처리 |

### 한 번에 이해하기

예를 들어 다음과 같은 요청이 있다고 하자.

```text
PATCH /books/42
```

Header:

```text
Authorization: API_KEY
```

Body:

```json
{
  "title": "수정된 제목"
}
```

각 부분의 역할은 다음과 같다.

```text
/books/42
    ↓
Path Parameter
→ 42번 책을 대상으로

Authorization
    ↓
Header
→ 인증 정보를 보내고

"title": "수정된 제목"
    ↓
Body
→ 이 데이터로 수정해줘
```

즉 다음처럼 기억하면 된다.

> **Header = 요청에 필요한 부가 정보·인증**  
> **Path = 어떤 자원을 대상으로 할지**  
> **Query = 어떤 조건으로 조회할지**  
> **Body = 서버가 생성·수정·처리할 데이터**

---

## 6. HTTP 응답 확인

응답을 받으면 다음 순서로 확인한다.

### 1. Status Code

요청이 성공했는지 또는 문제가 있는지를 나타낸다.

### 2. Content-Type

응답 본문의 형식을 나타낸다.

```text
application/json; charset=utf-8
```

JSON을 예상했는데 HTML이 반환된다면 오류 안내 페이지일 수 있다.

### 3. Response Body

실제 데이터 또는 오류 정보가 들어 있다.

```python
payload = {
    "items": [
        {"id": 1, "title": "처음 만나는 파이썬"}
    ],
    "page": 1,
}
```

응답 구조에서 필요한 데이터가 `items`, `data`, `results` 등 어디에 있는지 확인해야 한다.

오류 응답도 JSON으로 반환될 수 있으므로 **JSON 파싱에 성공했다고 해서 API 호출이 성공한 것은 아니다.**

---

## 7. 주요 HTTP 상태 코드

상태 코드의 첫 번째 숫자로 큰 범주를 구분할 수 있다.

| 범위 | 의미 |
| --- | --- |
| `2xx` | 성공 |
| `4xx` | 요청 또는 인증 문제 |
| `5xx` | 서버 측 문제 |

자주 사용하는 상태 코드는 다음과 같다.

| 코드 | 의미 | 확인할 내용 |
| --- | --- | --- |
| `200` | 성공 | 응답 형식과 본문 |
| `201` | 생성 성공 | 생성된 자원 |
| `400` | 잘못된 요청 | Parameter 이름과 자료형 |
| `401` | 인증 필요 또는 실패 | 인증 정보 |
| `403` | 요청은 이해했지만 허용되지 않음 | 권한 또는 정책 |
| `404` | 자원 또는 경로 없음 | Endpoint 또는 식별자 |
| `429` | 너무 많은 요청 | 기다린 뒤 재시도 |
| `500` | 서버 내부 오류 | 요청 정보와 서버 상태 |

상태 코드만으로 모든 원인을 알 수 없으므로 **응답 본문과 API 문서의 오류 설명도 함께 확인한다.**

---

## 8. API 문서 읽는 방법

API 문서를 볼 때는 다음 내용을 확인한다.

### 1. Base URL

개발, 테스트, 운영 주소가 구분되어 있는지 확인한다.

### 2. Endpoint와 Method

```text
GET /books
```

처럼 경로와 Method를 함께 확인한다.

### 3. 인증 방식

인증 정보를 어디에 넣는지 확인한다.

- Header
- Query
- 별도 Token 방식

### 4. Parameter

다음 내용을 확인한다.

- 이름
- 위치
- 자료형
- 필수 여부
- 허용 범위

### 5. 성공 응답 구조

상태 코드와 JSON 응답 예제에서 필요한 데이터의 위치를 확인한다.

### 6. 오류와 호출 제한

- 주요 오류 코드
- 일일 호출 한도
- 초당 호출 한도
- 재시도 방법

### 문서 읽는 순서

```text
기능 설명
→ Endpoint / Method
→ 인증
→ 필수 Parameter
→ 요청 예제
→ 성공 응답 예제
→ 오류 응답 / 호출 한도
```

---

## 9. 요청 구조 분석표

API 문서를 읽은 뒤 다음과 같이 정리할 수 있다.

| 항목 | 예시 |
| --- | --- |
| 목적 | 도서 제목 검색 |
| Method | `GET` |
| Base URL | `https://api.example.org/v1` |
| Endpoint | `/books` |
| 인증 | `X-API-Key` Header |
| Query | `q` 필수, `page`, `limit` 선택 |
| 성공 | `200`, JSON |
| 데이터 위치 | `items` |
| 주요 오류 | `400`, `401`, `429` |

Parameter도 따로 정리할 수 있다.

| 이름 | 위치 | 자료형 | 필수 |
| --- | --- | --- | --- |
| `q` | Query | 문자열 | O |
| `page` | Query | 정수 | X |
| `limit` | Query | 정수 | X |
| `X-API-Key` | Header | 문자열 | O |

이 표는 API 문서의 내용을 실제 HTTP 요청 코드로 옮길 때 기준이 된다.

---

## 핵심 정리

- HTTP는 **클라이언트의 요청 → 서버의 응답**으로 동작한다.
- **Base URL**은 여러 Endpoint가 공유하는 앞부분이다.
- **Endpoint**는 특정 기능이나 자원을 나타내는 경로이다.
- **Path Parameter**는 특정 자원을 지정한다.
- **Header**는 요청에 대한 부가 정보와 인증 정보를 전달한다.
- **Query Parameter**는 검색, 필터, 페이지 같은 조회 조건을 전달한다.
- **Request Body**는 서버가 생성·수정·처리할 실제 데이터를 담는다.
- 응답은 **Status Code → Content-Type → Response Body** 순서로 확인한다.
- API 문서에서는 **주소, Method, 인증, Parameter, 응답 구조, 오류와 호출 제한**을 확인한다.

### 가장 쉽게 기억하기

```text
Header = 인증·부가 정보
Path   = 어떤 자원?
Query  = 어떤 조건?
Body   = 어떤 데이터로 처리할까?
```