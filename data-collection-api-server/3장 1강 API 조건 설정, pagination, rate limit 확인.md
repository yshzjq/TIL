---
title: 3장 2강 응답 JSON 파싱과 원천 데이터 저장
date: 2026-09-21
updated: 2026-09-21
description: KANT 강의 '3장 2강 응답 JSON 파싱과 원천 데이터 저장' 정리
---

# 3장 2강 : 응답 JSON 파싱과 원천 데이터 저장

## 1. 원본 데이터와 가공 데이터를 나누는 이유

API 응답에는 지금 당장 사용하지 않는 필드도 포함될 수 있다.

예:

```python
raw_response = {
    "pages_collected": [1, 2, 3],
    "items": [
        {
            "source_id": "news-001",
            "title": "첫 번째 소식",
            "body_html": "<p>데이터를 안전하게 정제합니다.</p>",
            "published_at": "2026/06/01",
            "source_url": "https://example.com/news/001",
        }
    ],
}
```

이 중 필요한 필드만 따로 뽑아 사용할 수 있다.

하지만 **원본 응답은 별도로 보존한다.**

### 원본을 남기는 이유

- 필드 추출이 잘못됐는지 다시 확인할 수 있다.
- 나중에 다른 필드가 필요할 때 API를 다시 호출하지 않아도 될 수 있다.
- API 응답 구조가 바뀌었는지 비교할 수 있다.
- 정제 전 데이터를 확인할 수 있다.

### 파일 역할

| 파일 | 역할 |
| --- | --- |
| `raw_response.json` | API 응답 원본 |
| `items.json` | 필요한 필드만 추출한 데이터 |
| `items.csv` | 표 형태로 보기 쉬운 데이터 |
| `collection_log.json` | 수집 실행 정보 |

즉:

```text
원본 응답
   ↓
raw_response.json

   ↓ 필요한 필드 추출

items.json
items.csv
```

원본과 가공 결과는 서로 다른 파일로 저장한다.

---

## 2. JSON 객체와 배열

JSON의 객체는 Python에서 보통 `dict`, 배열은 `list`가 된다.

### JSON 객체 → Python dict

```python
news = {
    "source_id": "news-001",
    "title": "첫 번째 소식",
}
```

값을 가져올 때:

```python
news["title"]
```

### JSON 배열 → Python list

```python
items = [
    {"source_id": "news-001"},
    {"source_id": "news-002"},
]
```

배열 길이는:

```python
len(items)
```

으로 확인한다.

---

## 3. 응답에서 `items` 배열 찾기

API 응답 전체가 객체라면 필요한 배열을 먼저 꺼낸다.

```python
payload = {
    "page": 1,
    "items": [
        {"source_id": "news-001"}
    ],
}
```

```python
raw_items = payload.get("items")
```

그다음 실제로 리스트인지 확인한다.

```python
if not isinstance(raw_items, list):
    raise ValueError("payload.items는 배열이어야 합니다.")
```

API마다 배열 이름은 다를 수 있으므로 성공 응답 예제를 확인해야 한다.

예:

```text
items
data
results
documents
```

---

## 4. JSON의 `null`과 Python의 `None`

JSON의:

```json
null
```

은 Python에서:

```python
None
```

으로 읽힌다.

다음 값들은 서로 다르다.

```text
""
0
None
```

정제 규칙은 이후 단계에서 따로 처리한다.

---

## 5. 필요한 필드만 추출하기

원본 `items`에서 필요한 필드만 새 딕셔너리로 만든다.

```python
def extract_items(raw_response):
    raw_items = raw_response.get("items")

    if not isinstance(raw_items, list):
        raise ValueError("raw_response.items는 배열이어야 합니다.")

    extracted = []

    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("items의 각 원소는 객체여야 합니다.")

        extracted.append(
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "body_html": item["body_html"],
                "published_at": item["published_at"],
                "source_url": item["source_url"],
            }
        )

    return extracted
```

이 과정은:

```text
원본 item
   ↓
필요한 필드만 선택
   ↓
새로운 dict 생성
```

이다.

### 원본을 직접 수정하지 않는다

```python
raw_item = {
    "source_id": "news-001",
    "title": "첫 소식",
}
```

새로운 객체를 만든다.

```python
clean_item = {
    "source_id": raw_item["source_id"],
    "title": raw_item["title"],
}
```

즉:

> **원본은 그대로 두고, 필요한 데이터는 새로 만든다.**

---

## 6. Raw JSON 저장

Python에서는 `json` 모듈을 사용해 JSON 파일을 저장한다.

### 저장 폴더 만들기

```python
from pathlib import Path

output_dir = Path("artifacts")
output_dir.mkdir(parents=True, exist_ok=True)
```

파일 경로:

```python
raw_path = output_dir / "raw_response.json"
```

### JSON 문자열 만들기

```python
import json

text = json.dumps(
    raw_response,
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
) + "\n"
```

파일에 저장:

```python
raw_path.write_text(
    text,
    encoding="utf-8"
)
```

### 옵션 의미

| 옵션 | 역할 |
| --- | --- |
| `ensure_ascii=False` | 한글을 그대로 저장 |
| `indent=2` | 들여쓰기를 넣어 읽기 쉽게 함 |
| `sort_keys=True` | 키 순서를 일정하게 함 |
| `encoding="utf-8"` | UTF-8 인코딩 사용 |

---

## 7. 필요한 데이터만 JSON으로 저장

필요한 필드만 추출한 데이터도 별도 JSON 파일로 저장한다.

```python
items_path = output_dir / "items.json"
```

```python
items_text = json.dumps(
    items,
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
) + "\n"
```

```python
items_path.write_text(
    items_text,
    encoding="utf-8"
)
```

### 두 JSON 파일의 차이

`raw_response.json`

```python
{
    "pages_collected": [...],
    "items": [...]
}
```

→ 최상위가 **객체(dict)**

`items.json`

```python
[
    {...},
    {...}
]
```

→ 최상위가 **배열(list)**

따라서 나중에 파일을 읽을 때 구조가 다르다는 점을 알아야 한다.

---

## 8. JSON을 사용하는 경우

JSON은 다음 경우에 적합하다.

- 중첩된 데이터를 보존해야 할 때
- 자료형을 비교적 자연스럽게 유지할 때
- 다른 Python 프로그램에서 다시 읽을 때

JSON과 CSV 중 하나만 선택할 필요는 없고, 목적에 따라 둘 다 만들 수 있다.

---

## 9. CSV 저장

CSV는 데이터를 **행과 열 형태**로 저장한다.

Python의 `csv.DictWriter`를 사용한다.

```python
import csv

def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_id",
                "title",
                "body_html",
                "published_at",
                "source_url",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)
```

### `fieldnames`

CSV의 열 이름과 순서를 정한다.

```python
fieldnames = [
    "source_id",
    "title",
    "body_html",
    "published_at",
    "source_url",
]
```

### `writeheader()`

첫 줄에 열 이름을 작성한다.

### `writerow()`

딕셔너리 한 개를 CSV의 한 행으로 저장한다.

---

## 10. CSV 저장 시 주의점

CSV는 값 안에 다음과 같은 문자가 있을 수 있다.

```text
쉼표 ,
줄바꿈
따옴표 "
```

따라서 문자열을 직접 이어 붙이지 않는다.

좋지 않은 방식:

```python
",".join(...)
```

대신 Python의 `csv` 모듈에 처리를 맡긴다.

또한 CSV는 모든 행이 같은 열 구조를 가지는 것이 이해하기 쉽다.

---

## 11. 파일 저장 구조

이번 실습의 파일 구조는 다음과 같다.

```text
3-2강/
├── code/
│   ├── storage.py
│   └── run_all.py
│
├── artifacts/
│   ├── raw_response.json
│   ├── items.json
│   ├── items.csv
│   └── collection_log.json
│
└── expected_output/
    └── run_all.txt
```

---

## 12. 실행 위치와 저장 경로

단순히:

```python
Path("artifacts")
```

를 사용하면 실행 위치에 따라 저장되는 위치가 달라질 수 있다.

그래서 현재 Python 파일 위치를 기준으로 경로를 만든다.

```python
from pathlib import Path

lesson_dir = Path(__file__).resolve().parent.parent
output_dir = lesson_dir / "artifacts"
```

`__file__`은 현재 실행 중인 Python 파일의 위치를 의미한다.

이렇게 하면 어디에서 실행해도 같은 폴더에 결과를 저장할 수 있다.

---

## 13. 수집 로그 저장

데이터만 저장하면 나중에:

```text
언제 수집했는지
어디에서 가져왔는지
몇 건을 가져왔는지
성공했는지
```

알기 어렵다.

그래서 별도의 수집 로그를 만든다.

```python
collection_log = {
    "collected_at": "2026-07-16T09:00:00+09:00",
    "source": "fixture://news?page=all",
    "item_count": 5,
    "status": "success",
}
```

### 기본 로그 항목

| 필드 | 의미 |
| --- | --- |
| `collected_at` | 데이터를 수집한 시각 |
| `source` | 데이터 출처 |
| `item_count` | 수집한 데이터 수 |
| `status` | 수집 성공 여부 |

---

## 14. 로그에 넣지 않는 정보

수집 로그에는 다음과 같은 민감한 정보를 넣지 않는다.

```text
API Key 원문
Authorization Header 전체
개인정보가 포함된 전체 응답
API Key가 포함된 전체 URL
```

원본 응답은 로그에 넣지 않고 별도의 파일로 저장한다.

---

## 15. 수집 시각과 데이터 시각

`collected_at`은 **내가 데이터를 받은 시각**이다.

```text
collected_at
= API 응답을 수집한 시각
```

데이터 자체의 날짜나 관측 시각은 별도의 값이다.

예:

```text
published_at
observed_at
data_date
```

즉:

> **수집한 시각과 데이터가 만들어진 시각은 서로 다를 수 있다.**

---

## 16. 한 번에 저장하기

원본, JSON, CSV, 로그를 한 번에 저장하는 함수로 만들 수 있다.

```python
def save_collection_bundle(
    output_dir,
    raw_response
):
    items = extract_items(raw_response)

    save_json(
        output_dir / "raw_response.json",
        raw_response
    )

    save_json(
        output_dir / "items.json",
        items
    )

    save_csv(
        output_dir / "items.csv",
        items
    )

    log = {
        "collected_at": "2026-07-16T09:00:00+09:00",
        "source": "fixture://news?page=all",
        "item_count": len(items),
        "status": "success",
    }

    save_json(
        output_dir / "collection_log.json",
        log
    )

    return len(items)
```

전체 흐름은 다음과 같다.

```text
API 응답
   ↓
raw_response
   ↓
필요한 필드 추출
   ↓
items
   ↓
┌──────────────────────┐
│ raw_response.json    │
│ items.json           │
│ items.csv            │
│ collection_log.json  │
└──────────────────────┘
```

---

## 17. 저장 결과 확인

같은 입력과 같은 저장 규칙을 사용하면 결과도 같아야 한다.

확인할 내용:

- 파일 이름이 같은가?
- JSON 구조와 건수가 같은가?
- CSV 열 순서가 같은가?
- 한글이 깨지지 않았는가?
- 로그에 API 키가 들어가지 않았는가?

이번 실습에서는 결과가 일정하게 나오도록 다음 값을 고정한다.

- 입력 데이터
- 수집 시각
- JSON 키 순서
- CSV 열 순서
- 파일 이름

---

# 핵심 정리

- API 응답에서 먼저 필요한 `items` 배열을 찾는다.
- JSON 객체는 Python의 `dict`, 배열은 `list`가 된다.
- 원본 데이터는 `raw_response.json`으로 보존한다.
- 필요한 필드만 추출한 데이터는 별도 `items.json`으로 저장한다.
- 표 형태로 사용할 데이터는 `items.csv`로 저장할 수 있다.
- JSON 저장 시 `ensure_ascii=False`, `indent=2`, UTF-8 등을 사용한다.
- CSV는 `csv.DictWriter`를 사용하고 열 순서를 `fieldnames`로 정한다.
- 수집 로그에는 출처, 수집 시각, 건수, 상태를 남긴다.
- API 키나 인증 정보는 로그에 넣지 않는다.
- 원본과 가공 데이터는 서로 다른 파일로 관리한다.

## 가장 중요한 흐름

```text
API 응답
   ↓
JSON 파싱
   ↓
items 배열 확인
   ↓
원본 저장
raw_response.json
   ↓
필요한 필드 추출
   ↓
items.json
items.csv
   ↓
수집 정보 기록
collection_log.json
```

## 가장 쉽게 기억하기

```text
raw_response.json
= API에서 받은 원본

items.json
= 필요한 필드만 추출한 JSON

items.csv
= 표 형태로 보기 쉬운 데이터

collection_log.json
= 언제, 어디서, 몇 건 수집했는지 기록
```