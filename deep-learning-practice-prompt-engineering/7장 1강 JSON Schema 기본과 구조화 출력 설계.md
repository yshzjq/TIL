---
title: 7장 1강 JSON Schema 기본과 구조화 출력 설계
date: 2026-09-11
updated: 2026-09-11
description: KANT 강의 '7장 1강 JSON Schema 기본과 구조화 출력 설계' 정리
---

## 1. JSON Schema란 무엇인가

JSON Schema는 JSON 데이터가 따라야 할 구조와 규칙을 표현하는 문서다

다음 JSON은 고객 문의 분석 결과

```json
{
  "category": "delivery",
  "priority": 2,
  "needs_review": false
}
```

이 값이 프로그램에 안전하게 들어오려면 다음을 확인해야 합니다.

- `category`가 문자열인가
- 허용된 범주 중 하나인가
- `priority`가 정수인가
- `priority`가 1~3 범위인가
- `needs_review`가 boolean인가
- 필수 필드가 빠지지 않았나
- 정의하지 않은 필드가 추가되지 않았나

JSON Schema는 이 질문을 기계가 검사할 수 있는 형태로 적는다

```json
{
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["delivery", "refund", "account"]
    },
    "priority": {
      "type": "integer",
      "minimum": 1,
      "maximum": 3
    },
    "needs_review": {
      "type": "boolean"
    }
  },
  "required": ["category", "priority", "needs_review"],
  "additionalProperties": false
}
```

Structured Output 전체 흐름

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_structured_output_flow.png' | relative_url }" alt="01_structured_output_flow.png" loading="lazy">

JSON Schema는 모델을 더 똑똑하게 만드는 수식이 아니다

**모델과 애플리케이션 사이의 출력 계약**이다.

모델 = 답을 만드는 쪽
애플리케이션 = 그 답을 받아서 사용하는 프로그램
JSON Schema = 둘 사이에서 "이 형식으로 주고받자"라고 정한 규칙

## 2. Structured Output의 큰 그림

Structured Output = 모델의 출력을 프로그램이 쉽게 사용할 수 있도록 일정한 구조로 받는 방식

자연어 출력은 사람이 읽기 쉽지만 프로그램이 안정적으로 사용하기 어렵다

```
모델 응답: "이 문의는 배송 문제이고 우선순위는 2 정도입니다."
```

이 문장에서 분류와 우선순위를 꺼내려면 정규식이나 문자열 규칙이 필요하다. 표현이 조금만 바뀌어도 파서가 실패할 수 있다.

Structured Output은 간략한 흐름
```
업무 요구사항
-> JSON Schema 설계
-> 모델에 구조 전달
-> 구조화된 출력 수신
-> 로컬 검증
-> 비즈니스 로직 사용
```

Schema는 출력의 **형태**를 안정화한다

그러나 내용의 사실성이나 업무 판단이 맞는지까지 자동으로 보장하지는 않는다

예를 들어 `confidence`가 Schema상 0~1 범위를 만족해도 모델이 계산한 신뢰도가 실제 통계적 확률이라는 보장은 없다. 따라서 구조 검증과 의미 검증을 구분해야 합니다.

모델이 출력한 값이 형식에 맞는다고 해서, 내용까지 무조건 옳다는 뜻이 아니다

| 검증 종류 | 질문 | 예시 |
| --- | --- | --- |
| 구조 검증 | 필드와 자료형이 맞는가? | `priority`가 정수인가? |
| 범위 검증 | 허용 범위인가? | 1~3인가? |
| 비즈니스 규칙 | 업무 규칙에 맞는가? | 환불 요청인데 주문번호가 필요한가? |
| 사실 검증 | 실제 데이터와 같은가? | 주문 상태가 실제로 배송 중인가? |

## 3. JSON과 JSON Schema의 차이

### JSON

JSON은 실제 데이터를 표현한다

```json
{
  "document_type": "receipt",
  "language": "ko",
  "pages": 1
}
```

### JSON Schema

JSON Schema는 데이터가 따라야 할 규칙을 표현한다

```json
{
  "type": "object",
  "properties": {
    "document_type": {
      "type": "string"
    },
    "language": {
      "type": "string"
    },
    "pages": {
      "type": "integer"
    }
  },
  "required": ["document_type", "language", "pages"]
}
```

Valid JSON과 Invalid JSON

Valid = 형식과 규칙이 맞다<br>
Invalid = 형식이나 규칙이 틀리다<br>
Valid JSON = JSON 문법이 맞다<br>
Invalid JSON = JSON 문법이 틀리다<br>

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/03_valid_invalid_json.png' | relative_url }" alt="03_valid_invalid_json.png" loading="lazy">

```
JSON 문법이 맞다
≠ Schema를 만족한다
≠ 내용이 사실이다
```

예를 들어 아래 값은 JSON 문법은 맞지만 `pages`가 문자열이므로 위 Schema를 만족하지 않는다

```json
{
  "document_type": "receipt",
  "language": "ko",
  "pages": "한 장"
}
```

## 4. 핵심 Schema 키워드

JSON Schema 핵심 키워드

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_json_schema_keywords.png' | relative_url }" alt="02_json_schema_keywords.png" loading="lazy">

### `type`

값의 자료형을 제한한다

| JSON Schema type | Python에서 대응되는 대표 값 |
| --- | --- |
| `object` | `dict` |
| `array` | `list` |
| `string` | `str` |
| `integer` | 소수점이 없는 `int` |
| `number` | `int` 또는 `float` 계열 숫자 |
| `boolean` | `True`, `False` |
| `null` | `None` |


### `properties`

객체 안에 어떤 필드가 있는지 정의한다

### `required`

반드시 존재해야 하는 필드 이름을 배열로 적는다

### `enum`

허용된 값의 목록

```json
"category": {
  "type": "string",
  "enum": ["delivery", "refund", "account"]
}
```


### `items`

배열의 각 원소가 따라야 할 규칙이다

```json
"keywords": {
  "type": "array",
  "items": {
    "type": "string"
  }
}
```

### `additionalProperties`

`false`로 설정하면 `properties`에 정의하지 않은 필드를 허용하지 않는다

```
정의된 필드: category, priority
출력에 extra_note가 추가됨
-> additionalProperties=false이면 검증 실패
```

### 범위와 길이

- `minimum`, `maximum`: 숫자 범위
- `minLength`, `maxLength`: 문자열 길이
- `minItems`, `maxItems`: 배열 원소 개수

OpenAI Structured Outputs의 strict Schema는 일반 JSON Schema의 모든 기능을 지원하는 것은 아니다.

## 5. 출력 계약을 설계하는 순서

Schema를 코드부터 작성하면 필드가 불필요하게 많아질 수 있습니다.

### 5-1. 출력이 사용될 위치를 정한다

```
상담 화면에 표시
데이터베이스에 저장
다음 Tool의 인자로 전달
자동 라우팅에 사용
사람 검토 큐에 전송
```


### 5-2. 최소 필드를 고른다

고객 문의 라우팅이라면 다음 정도로 시작할 수 있다.

```
category
priority
summary
needs_review
```

라우팅 = 입력을 보고 적절한 처리 경로로 보내는 것


### 5-3. 각 필드의 자료형과 허용값을 정한다

```
category: enum 문자열
priority: 1~3 정수
summary: 문자열
needs_review: boolean
```
### 5-4. 정보가 없을 때의 정책을 정한다


선택지 세 가지

1. 빈 문자열을 사용한다
2. `null`을 허용한다
3. `needs_review=true`와 이유 필드를 사용한다

모델이 임의 값을 채우지 않게 하려면 **정보 부족 표현 방식**을 Schema와 Prompt에 함께 적어야 한다

needs_review = 사람 검토 필요 여부를 나타내는 true/false 값 입니다.

```
  "needs_review": true,
  "review_reason": "문서 제목이 잘려 있어 유형을 확정하기 어렵습니다."
```
Structured Output이나 JSON Schema를 사용하면<br>
모델이 실제로 저런 구조의 JSON 형태로 응답하도록 만들 수 있다

### 5-5. 알 수 없는 필드를 막습니다

`additionalProperties: false`를 사용해 예상하지 못한 키가 downstream 코드로 들어오는 것을 줄인다


## 6. 단순 객체와 중첩 객체 해석

### 단순 객체

```json
{
  "category": "delivery",
  "priority": 2,
  "needs_review": false
}
```


### 중첩 객체

영수증처럼 항목 배열이 있으면 중첩 구조가 필요하다

```json
{
  "merchant": "Sparta Cafe",
  "items": [
    {
      "name": "Americano",
      "quantity": 2,
      "unit_price": 4500
    }
  ],
  "total": 9000
}
```

중첩 Schema 구조

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/04_nested_schema.png' | relative_url }" alt="04_nested_schema.png" loading="lazy">

중첩 Schema에서는 각 객체마다 `properties`, `required`, `additionalProperties`를 별도로 생각해야 한다

```json
{
  "type": "object",
  "properties": {
    "merchant": {"type": "string"},
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "quantity": {"type": "integer", "minimum": 1},
          "unit_price": {"type": "number", "minimum": 0}
        },
        "required": ["name", "quantity", "unit_price"],
        "additionalProperties": false
      }
    },
    "total": {"type": "number", "minimum": 0}
  },
  "required": ["merchant", "items", "total"],
  "additionalProperties": false
}
```

중첩이 깊어질수록 출력과 검증이 어려워진다.

실제로 사용하지 않는 필드는 넣지 말고, 한 번의 출력이 너무 복잡하면 태스크를 나눈다

## 7. 기본 실습: 문서 분류 Schema 만들기

다음 업무를 구조화

```
입력: 문서의 첫 페이지 텍스트
출력:
- 문서 유형: invoice, receipt, contract, other
- 언어 코드
- 한 문장 요약
- 사람 검토 필요 여부
- 검토 이유
```

```python
# [실습 목적]
# Python dict로 JSON Schema를 작성합니다.
# 이 단계에서는 API를 호출하지 않고 출력 계약 자체를 검토합니다.

import json

DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": ["invoice", "receipt", "contract", "other"],
            "description": "문서의 대표 유형입니다.",
        },
        "language": {
            "type": "string",
            "minLength": 2,
            "description": "ko, en과 같은 짧은 언어 코드입니다.",
        },
        "summary": {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
            "description": "문서 핵심을 한 문장으로 요약합니다.",
        },
        "needs_review": {
            "type": "boolean",
            "description": "입력이 흐리거나 유형이 애매하면 true입니다.",
        },
        "review_reason": {
            # strict 출력에서는 선택 필드를 null 허용 필수 필드로 만드는 방식을
            # 자주 사용합니다. 정보가 없으면 null을 반환합니다.
            "type": ["string", "null"],
            "description": "검토가 필요하지 않으면 null입니다.",
        },
    },
    # 필드 누락을 막기 위해 모든 필드를 required에 포함합니다.
    "required": [
        "document_type",
        "language",
        "summary",
        "needs_review",
        "review_reason",
    ],
    # 정의되지 않은 임의 필드를 허용하지 않습니다.
    "additionalProperties": False,
}

# indent=2를 사용해 사람이 읽기 좋은 형태로 출력합니다.
print(json.dumps(DOCUMENT_SCHEMA, ensure_ascii=False, indent=2))

# Schema의 최상위가 object인지, 필수 필드가 들어 있는지 확인합니다.
assert DOCUMENT_SCHEMA["type"] == "object"
assert "document_type" in DOCUMENT_SCHEMA["required"]
assert DOCUMENT_SCHEMA["additionalProperties"] is False
```

출력

```

{
  "type": "object",
  "properties": {
    "document_type": {
      "type": "string",
      "enum": [
        "invoice",
        "receipt",
        "contract",
        "other"
      ],
      "description": "문서의 대표 유형입니다."
    },
    "language": {
      "type": "string",
      "minLength": 2,
      "description": "ko, en과 같은 짧은 언어 코드입니다."
    },
    "summary": {
      "type": "string",
      "minLength": 1,
      "maxLength": 200,
      "description": "문서 핵심을 한 문장으로 요약합니다."
    },
    "needs_review": {
      "type": "boolean",
      "description": "입력이 흐리거나 유형이 애매하면 true입니다."
    },
    "review_reason": {
      "type": [
        "string",
        "null"
      ],
      "description": "검토가 필요하지 않으면 null입니다."
    }
  },
  "required": [
    "document_type",
    "language",
    "summary",
    "needs_review",
    "review_reason"
  ],
  "additionalProperties": false
}

```

### 정상 출력 예시

```python
valid_document = {
    "document_type": "contract",
    "language": "ko",
    "summary": "서비스 제공 조건과 계약 해지 기준을 정한 계약서입니다.",
    "needs_review": False,
    "review_reason": None,
}
```

### 검토가 필요한 출력 예시

```python
review_document = {
    "document_type": "other",
    "language": "ko",
    "summary": "문서 일부만 보여 유형을 확정하기 어렵습니다.",
    "needs_review": True,
    "review_reason": "제목과 발행 주체가 잘려 있습니다.",
}
```


## 8. 연습 문제: 유효한 출력과 잘못된 출력 구분

```json
{
  "type": "object",
  "properties": {
    "label": {"type": "string", "enum": ["positive", "negative"]},
    "score": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "required": ["label", "score"],
  "additionalProperties": false
}
```


### 문제 1

아래 값은 유효한가요?

```json
{"label": "positive", "score": 0.82}
```

유효, `label`은 허용 enum 중 하나이고, `score`는 0~1 범위의 숫자이며 필수 필드가 모두 있다.

### 문제 2

아래 값이 실패하는 이유

```json
{"label": "neutral", "score": "높음", "note": "애매함"}
```

1. `label`의 `neutral`은 허용 enum에 없다.
2. `score`는 number여야 하지만 문자열이다
3. `note`는 Schema에 정의되지 않은 추가 필드이며 `additionalProperties=false`다.

### 문제 3

`score`를 알 수 없을 때 `null`을 허용하려면 어떻게 수정해야 하나

```json
"score": {
  "type": ["number", "null"],
  "minimum": 0,
  "maximum": 1
}
```

다만 `null`이 허용되는 이유와 이후 코드의 처리 정책도 함께 정해야 한다

## 9. 추가 실습: 로컬 Schema 검증하기

`jsonschema` 라이브러리를 사용하면 API 호출 없이 Schema를 테스트할 수 있다.

```python
# Google Colab 또는 터미널에서 한 번 실행합니다.
# !pip install -q jsonschema
```
```python
# [실습 목적]
# DOCUMENT_SCHEMA와 실제 Python dict를 비교해 구조 오류를 찾습니다.

from jsonschema import Draft202012Validator

# Validator를 한 번 만들어 여러 출력에 재사용합니다.
validator = Draft202012Validator(DOCUMENT_SCHEMA)

def collect_schema_errors(data: dict) -> list[str]:
    """모든 Schema 오류를 사람이 읽기 좋은 문자열로 반환합니다."""
    messages: list[str] = []

    # iter_errors()는 첫 오류에서 멈추지 않고 가능한 오류를 모두 반환합니다.
    # path를 기준으로 정렬하면 출력 순서가 비교적 일정해집니다.
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.path))

    for error in errors:
        # error.path는 중첩 데이터에서 오류가 발생한 위치를 나타냅니다.
        path = ".".join(str(part) for part in error.path) or "<root>"
        messages.append(f"{path}:{error.message}")

    return messages

invalid_document = {
    "document_type": "memo",        # enum에 없는 값입니다.
    "language": "k",                # minLength=2를 만족하지 않습니다.
    "summary": "",                  # minLength=1을 만족하지 않습니다.
    "needs_review": "yes",          # boolean이 아니라 문자열입니다.
    # review_reason 필드가 누락되었습니다.
    "extra": "허용되지 않은 필드",  # additionalProperties=false입니다.
}

errors = collect_schema_errors(invalid_document)

for message in errors:
    print("-", message)

assert len(errors) >= 5
```

예상 출력은 라이브러리 버전에 따라 표현이 조금 다를 수 있지만, 다음 위치를 확인할 수 있다.

```
- <root>:'review_reason' is a required property
- <root>:Additional properties are not allowed ('extra' was unexpected)
- document_type:'memo' is not one of ['invoice', 'receipt', 'contract', 'other']
- language:'k' is too short
- needs_review:'yes' is not of type 'boolean'
- summary:'' should be non-empty
```

### 정상 데이터 검사

```python
valid_errors = collect_schema_errors(valid_document)
print(valid_errors)

assert valid_errors == []
```

Schema를 모델에 연결하기 전에 정상 예시, 필드 누락, 잘못된 enum, 추가 필드, 중첩 배열 오류를 로컬 테스트로 먼저 확인해야한다

## 12. 이해도 점검

1. JSON과 JSON Schema의 차이를 설명
2. `properties`와 `required`는 각각 무엇을 정의하나
3. `enum`은 어떤 문제를 줄이나요?
4. `additionalProperties=false`의 효과는 무엇인가
5. 구조 검증과 사실 검증이 다른 이유는 무엇인가
6. 중첩 배열의 각 원소 구조는 어느 키워드 아래에 작성하나

- 정답 확인

1. JSON은 실제 데이터이고, JSON Schema는 그 데이터가 따라야 할 구조와 규칙
2. `properties`는 가능한 필드와 각 필드 규칙을 정의하고, `required`는 반드시 존재해야 할 필드를 정한다
3. 정해진 범주 밖의 임의 문자열이 들어오는 문제를 줄인다
4. Schema에 정의하지 않은 필드가 들어오면 검증 실패로 처리한다
5. 구조가 올바른 값도 실제 데이터와 다를 수 있기 때문이다
6. 배열 Schema의 `items` 아래에 작성한다

---


