---
title: 2장 2강 Python API 호출과 인증, 환경변수
date: 2026-09-21
updated: 2026-09-21
description: KANT 강의 '2장 2강 Python API 호출과 인증, 환경변수' 정리
---

# 2장 2강 : Python API 호출과 인증/환경변수

## 1. Python에서 API를 호출하는 흐름

API를 호출할 때는 다음 순서로 진행한다.

```text
환경 변수에서 API 키 읽기
→ Header / Query Parameter 만들기
→ Timeout을 지정하고 요청 보내기
→ HTTP 상태 확인
→ JSON 응답 확인
```

단계를 나누는 이유는 **어디에서 문제가 발생했는지 확인하기 쉽기 때문**이다.

---

## 2. API 키와 환경 변수

API 키는 API를 호출하는 주체와 사용량을 식별하는 **비밀값**일 수 있다.

따라서 실제 API 키를 코드에 직접 작성하지 않는다.

### 좋지 않은 예

```python
hardcoded_key = "real-secret-key"
```

코드에 직접 작성하면 Git 이력, 공유 노트북, 화면 캡처 등에 키가 남을 수 있다.

### 환경 변수에서 읽기

```python
import os

def load_api_key(variable_name: str = "COURSE_API_KEY") -> str:
    raw_value = os.getenv(variable_name)

    if raw_value is None or not raw_value.strip():
        raise RuntimeError(f"{variable_name} 환경 변수를 설정하세요.")

    return raw_value.strip()
```

`os.getenv()`은 환경 변수 값을 가져온다.

환경 변수가 존재하지 않으면 `None`을 반환한다.

```python
api_key = os.getenv("COURSE_API_KEY")
```

`strip()`은 값의 앞뒤 공백을 제거한다.

API 키는 **코드와 분리해서 관리하고 로그나 오류 메시지에도 원문을 출력하지 않는다.**

---

## 3. HTTPX

**HTTPX**는 Python에서 HTTP 요청을 보낼 수 있는 라이브러리이다.

이번 강의에서는 동기식 `httpx.Client`를 사용한다.

```python
import httpx

with httpx.Client(
    base_url="https://api.example.org/v1"
) as client:
    print(client.base_url)
```

### Base URL

공통으로 사용하는 API 주소를 Client에 지정할 수 있다.

```python
base_url = "https://api.example.org/v1"
```

그러면 실제 요청에서는 Endpoint만 전달할 수 있다.

```python
endpoint = "/news"
```

즉,

```text
Base URL
https://api.example.org/v1

+

Endpoint
/news
```

를 합쳐 요청하게 된다.

### Client를 사용하는 이유

- 공통 Base URL을 한곳에 둘 수 있다.
- 여러 요청에서 연결을 재사용할 수 있다.
- 공통 Header와 Timeout을 관리하기 쉽다.
- `with` 블록이 끝나면 연결 자원이 자동으로 정리된다.

---

## 4. Header와 Query Parameter 전달

2-1강에서 배운 Header와 Query Parameter를 Python 딕셔너리로 만든다.

### Header

```python
headers = {
    "X-API-Key": api_key,
    "Accept": "application/json",
}
```

- `X-API-Key`: 인증 정보
- `Accept`: 원하는 응답 형식

API마다 인증 방식은 다를 수 있다.

예:

```text
X-API-Key
Authorization: Bearer ...
serviceKey Query Parameter
OAuth Token
```

따라서 **실제 API 문서의 인증 방식을 확인해야 한다.**

---

### Query Parameter

```python
params = {
    "q": "데이터",
    "limit": 2,
}
```

`params`에는 검색이나 필터 같은 **조회 조건**을 넣는다.

HTTPX를 사용하면 다음과 같은 URL 문자열을 직접 만들 필요가 없다.

```text
?q=데이터&limit=2
```

대신 딕셔너리를 전달하면 HTTPX가 URL에 맞게 처리한다.

```python
params = {
    "q": "데이터",
    "limit": 2,
}
```

---

## 5. GET 요청 보내기

Header와 Query Parameter를 함께 전달한다.

```python
with httpx.Client(
    base_url="https://api.example.org/v1",
    timeout=5.0
) as client:

    response = client.get(
        "/news",
        headers=headers,
        params=params,
    )
```

각 부분의 역할은 다음과 같다.

```text
client.get("/news")
        ↓
Endpoint

headers=headers
        ↓
인증·응답 형식 등의 부가 정보

params=params
        ↓
검색·필터 등의 조회 조건

timeout=5.0
        ↓
네트워크 대기 제한
```

---

## 6. Timeout

서버나 네트워크가 응답하지 않으면 프로그램이 오래 기다릴 수 있다.

이를 방지하기 위해 `timeout`을 지정한다.

```python
timeout_seconds = 5.0
```

또는:

```python
httpx.Client(timeout=5.0)
```

`timeout`은 네트워크 단계에서 **무기한 기다리는 것을 막는 기본 설정**이다.

---

## 7. 상태 코드 확인

API 요청 후에는 JSON을 바로 처리하지 않고 먼저 HTTP 상태를 확인한다.

```python
response.raise_for_status()
```

`raise_for_status()`는 `4xx`, `5xx` 상태를 정상 응답과 구분한다.

예를 들어:

```text
200 → 성공
401 → 인증 오류
404 → 경로 또는 자원 없음
429 → 너무 많은 요청
500 → 서버 오류
```

### 왜 JSON보다 먼저 확인할까?

오류 응답도 JSON일 수 있기 때문이다.

예:

```json
{
  "detail": "invalid api key"
}
```

따라서 다음 순서로 확인한다.

```text
요청
→ raise_for_status()
→ response.json()
```

---

## 8. JSON 응답 확인

상태 코드가 정상이라면 JSON 응답을 Python 값으로 변환한다.

```python
payload = response.json()
```

예를 들어 응답이 다음과 같다고 하자.

```python
payload = {
    "items": [
        {
            "source_id": "news-001",
            "title": "첫 번째 소식"
        }
    ]
}
```

필요한 데이터는 `items` 안에 있다.

```python
items = payload.get("items")
```

그리고 원하는 자료형인지 확인한다.

```python
if not isinstance(items, list):
    raise ValueError("응답의 items는 배열이어야 합니다.")
```

### `.get()`을 사용하는 이유

```python
payload["items"]
```

는 `items`가 존재하지 않으면 `KeyError`가 발생한다.

반면:

```python
payload.get("items")
```

으로 가져온 뒤 자료형을 검사하면 문제를 구분하기 쉽다.

응답 필드 이름은 API마다 다를 수 있다.

```text
items
data
results
documents
```

따라서 API 문서의 **성공 응답 예제**를 확인해야 한다.

---

## 9. 전체 API 호출 흐름

지금까지 배운 내용을 연결하면 다음과 같다.

```python
import os
import httpx

api_key = os.getenv("COURSE_API_KEY")

headers = {
    "X-API-Key": api_key,
    "Accept": "application/json",
}

params = {
    "q": "데이터",
    "limit": 2,
}

with httpx.Client(
    base_url="https://api.example.org/v1",
    timeout=5.0
) as client:

    response = client.get(
        "/news",
        headers=headers,
        params=params,
    )

    response.raise_for_status()

    payload = response.json()

    items = payload.get("items")
```

흐름으로 보면:

```text
환경 변수
    ↓
API Key

Header / Params 생성
    ↓
HTTPX Client

GET 요청
    ↓
상태 확인

raise_for_status()
    ↓
JSON 변환

response.json()
    ↓
필요한 데이터 확인
```

---

## 10. API 호출 실패 위치

API 호출은 여러 단계에서 실패할 수 있다.

| 실패 위치 | 예시 | 확인할 내용 |
| --- | --- | --- |
| 설정 | API 키 없음 | 환경 변수 |
| 연결 | 서버 연결 실패 | URL, 네트워크, Timeout |
| HTTP 상태 | `401`, `404`, `429`, `500` | 상태 코드 |
| JSON 파싱 | HTML 응답 | Content-Type, 본문 |
| 데이터 구조 | `items` 없음 | API 문서, 응답 구조 |

모든 오류를 단순히 `"API 호출 실패"`라고 처리하면 원인을 찾기 어렵다.

최소한 다음 문제는 구분한다.

```text
환경 변수 문제
HTTP 상태 문제
응답 구조 문제
```

---

## 11. Requests와 HTTPX

Requests와 HTTPX 모두 Python에서 HTTP 요청을 보낼 수 있다.

Requests:

```python
requests.get(
    url,
    headers=headers,
    params=params,
    timeout=5.0
)
```

HTTPX:

```python
client.get(
    "/news",
    headers=headers,
    params=params,
    timeout=5.0
)
```

둘 다 다음 기능을 사용할 수 있다.

- `headers`
- `params`
- `timeout`
- `response.status_code`
- `response.json()`
- `raise_for_status()`

이번 과정에서는 **HTTPX를 중심으로 사용한다.**

---

## 12. MockTransport와 Fixture

실제 API는 다음과 같은 외부 상황의 영향을 받을 수 있다.

- 네트워크
- API 키
- API 제공자의 서버 상태

그래서 교안에서는 실제 외부 서버 대신 HTTPX의 **`MockTransport`**를 사용한다.

### MockTransport

`MockTransport`는 실제 서버에 요청을 보내지 않고 **미리 정해 둔 응답을 반환하도록 만드는 기능**이다.

흐름을 비교하면 다음과 같다.

```text
실제 API

Python 코드
   ↓
인터넷
   ↓
실제 API 서버
   ↓
응답
```

MockTransport를 사용하면:

```text
Python 코드
   ↓
MockTransport
   ↓
미리 정해 둔 응답
```

이 된다.

즉,

> **MockTransport = 실제 외부 서버 대신 요청을 받아주는 가짜 서버 역할**

이라고 이해하면 된다.

---

### Fixture

이 강의에서 **Fixture**는 실습을 반복할 때 같은 결과를 확인할 수 있도록 **미리 정해 둔 학습용 요청 조건과 응답 데이터**를 의미한다.

예를 들어 학습용 API 키를 다음처럼 정해 둔다.

```python
os.environ["COURSE_API_KEY"] = "lesson-key"
```

그리고 MockTransport가 이 값을 검사한다.

```python
if request.headers.get("X-API-Key") != "lesson-key":
    return httpx.Response(
        401,
        json={"detail": "invalid api key"}
    )
```

Query Parameter도 검사한다.

```python
if request.url.params.get("q") != "데이터":
    return httpx.Response(
        400,
        json={"detail": "q is required"}
    )
```

요청이 올바르면 미리 정해 둔 데이터를 반환한다.

```python
return httpx.Response(
    200,
    json={
        "items": [
            {
                "source_id": "news-001",
                "title": "첫 번째 소식"
            },
            {
                "source_id": "news-001",
                "title": "중복 소식"
            }
        ]
    }
)
```

따라서 이 강의에서는 다음과 같이 이해하면 된다.

> **MockTransport = 가짜 서버처럼 요청을 받아주는 기능**  
> **Fixture = 그 실습에서 사용할 미리 정해 둔 조건과 응답 데이터**

이를 이용하면 **인터넷이나 실제 API 키 없이도 같은 요청과 응답을 반복해서 재현할 수 있다.**

---

## 핵심 정리

- API 키는 코드에 직접 작성하지 않고 **환경 변수**에서 읽는다.
- `os.getenv()`으로 환경 변수 값을 가져올 수 있다.
- HTTPX는 Python에서 HTTP 요청을 보내는 라이브러리이다.
- `headers`에는 인증·응답 형식 같은 부가 정보를 넣는다.
- `params`에는 검색·필터 같은 Query Parameter를 넣는다.
- `timeout`을 지정해 네트워크가 무한히 기다리는 것을 막는다.
- 응답을 받으면 **`raise_for_status()` → `response.json()`** 순서로 확인한다.
- JSON을 받은 뒤 필요한 필드와 자료형도 확인한다.
- API 호출 실패는 **설정 → 연결 → HTTP 상태 → JSON → 데이터 구조**로 나누어 확인한다.
- **MockTransport**는 실제 외부 서버 대신 미리 정한 응답을 반환한다.
- **Fixture**는 같은 결과를 재현하기 위해 사용하는 학습용 요청 조건과 응답 데이터이다.

### 가장 중요한 흐름

```text
환경 변수에서 API Key 읽기
        ↓
Header / Params 만들기
        ↓
HTTPX Client 생성
        ↓
GET 요청
        ↓
raise_for_status()
        ↓
response.json()
        ↓
필요한 데이터 확인
```

### MockTransport와 Fixture

```text
MockTransport
= 실제 서버 대신 요청을 받아주는 역할

Fixture
= 실습에서 사용할 미리 정해 둔 조건과 응답
```