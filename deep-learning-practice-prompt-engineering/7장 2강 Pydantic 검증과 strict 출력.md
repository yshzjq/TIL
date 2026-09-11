---
title: 7장 2강 Pydantic 검증과 strict 출력
date: 2026-09-11
updated: 2026-09-11
description: KANT 강의 '7장 2강 Pydantic 검증과 strict 출력' 정리
---

## 1. Pydantic은 무엇인가

Pydantic은 Python type hint를 바탕으로 입력 데이터를 검증하고, <br>
일정한 Python 객체로 변환하는 라이브러리다

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
```

다음 값은 검증에 성공한다

```python
user = User.model_validate({"name": "민지", "age": 27})
print(user.name)
print(user.age)
```

```python
user = User.model_validate({"name": "민지", "age": 27})
print(user.name)
print(user.age)
```

자료형이 맞지 않으면 `ValidationError`가 발생합니다.

```python
User.model_validate({"name": "민지", "age": "스물일곱"})
```

Pydantic의 장점은 다음 세 가지를 같은 모델에서 관리할 수 있다는 점이다

```
Python 타입 정의
-> 입력 검증
-> JSON Schema 생성
```

Pydantic 검증 흐름

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/05_pydantic_validation.png' | relative_url }" alt="05_pydantic_validation.png" loading="lazy">

## 2. Pydantic 검증의 큰 그림

구조화 출력 파이프라인에서 Pydantic은 API 앞과 뒤 모두에 사용할 수 있습니다.

```
Pydantic 모델 정의
-> JSON Schema 생성
-> 모델에 출력 구조 전달
-> 구조화된 응답 수신
-> Pydantic 객체로 파싱
-> 비즈니스 규칙 검증
-> 저장/Tool/UI에 전달
```

OpenAI Python SDK의 `responses.parse()`는 Pydantic 모델을 `text_format`으로 전달하고,
<br>
파싱된 결과를 `response.output_parsed`에서 읽을 수 있게 한다.

Responses parse 흐름

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/07_responses_parse.png' | relative_url }" alt="07_responses_parse.png" loading="lazy">

구조화 출력에서 구분해야 할 세 경계가 있다

1. **API 출력 구조**: 필드와 자료형이 Schema를 따르는가?
2. **Pydantic 검증**: Python 객체로 안전하게 읽을 수 있는
3. **업무 규칙**: 필드 사이 관계가 실제 업무 규칙에 맞는가

예를 들어 영수증의 `total`이 숫자여도 `items` 합계와 다를 수 있다.

이 오류는 자료형 검증만으로 잡히지 않는다

## 3. BaseModel과 Field

`BaseModel`을 상속해 데이터 구조를 정의한다
<br>
`Field`에는 설명, 범위, 길이 같은 추가 규칙을 적는다

```python
from pydantic import BaseModel, Field

class SupportTicket(BaseModel):
    # min_length와 max_length는 문자열 길이 규칙입니다.
    summary: str = Field(
        min_length=1,
        max_length=200,
        description="고객 문의를 한 문장으로 요약합니다.",
    )

    # ge는 greater than or equal, le는 less than or equal입니다.
    priority: int = Field(
        ge=1,
        le=3,
        description="1은 낮음, 3은 높음입니다.",
    )

    needs_review: bool = Field(
        description="사람 검토가 필요한지 표시합니다.",
    )
```

### 인스턴스 만들기

```python
# model_validate()는 dict처럼 외부에서 들어온 값을 검증할 때 사용합니다.
ticket = SupportTicket.model_validate(
    {
        "summary": "배송 완료로 표시되지만 상품을 받지 못했습니다.",
        "priority": 3,
        "needs_review": True,
    }
)

print(ticket)
print(ticket.priority)

# model_dump()는 Pydantic 객체를 일반 Python dict로 바꿉니다.
print(ticket.model_dump())
```

### 정의하지 않은 필드 막기

기본 설정에서는 모델에 따라 추가 필드가 무시될 수 있다.

출력 계약을 엄격하게 관리하려면 `extra="forbid"`를 사용한다

```python
from pydantic import ConfigDict

class StrictTicket(BaseModel):
    # 정의하지 않은 필드가 들어오면 ValidationError를 발생시킵니다.
    model_config = ConfigDict(extra="forbid")

    summary: str
    priority: int = Field(ge=1, le=3)
    needs_review: bool
```

## 4. Literal과 중첩 모델

### Literal로 허용값 제한하기

`Literal`은 정해진 값만 허용합니다. JSON Schema의 `enum`과 비슷한 역할입니다.

```python
from typing import Literal

Category = Literal["delivery", "refund", "account", "other"]

class ClassifiedTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    summary: str = Field(min_length=1, max_length=200)
    priority: int = Field(ge=1, le=3)
    needs_review: bool
```

### 중첩 모델

영수증 항목처럼 객체 안에 객체 배열이 들어갈 때 별도 모델을 만든다

```python
class ReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)

class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merchant: str = Field(min_length=1)
    items: list[ReceiptItem] = Field(min_length=1)
    total: float = Field(ge=0)
    currency: Literal["KRW", "USD", "EUR"]
```

```
Receipt
├── merchant: str
├── items: list[ReceiptItem]
│   ├── name: str
│   ├── quantity: int
│   └── unit_price: float
├── total: float
└── currency: enum
```

## 5. 타입 변환과 strict 검증

Pydantic은 편의를 위해 일부 값을 자동 변환할 수 있다.

```python
class Quantity(BaseModel):
    value: int

converted = Quantity.model_validate({"value": "3"})
print(converted.value, type(converted.value))
# 문자열 "3"이 정수 3으로 변환될 수 있습니다.
```

이 동작은 사용자 폼처럼 변환이 필요한 곳에서는 편리하지만,

LLM 출력의 자료형 오류를 그대로 발견하고 싶을 때는 주의해야 한다

일반 검증과 strict 검증

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/06_strict_coercion.png' | relative_url }" alt="06_strict_coercion.png" loading="lazy">

### StrictInt 사용하기

```python
from pydantic import StrictInt

class StrictQuantity(BaseModel):
    value: StrictInt
```

```python
from pydantic import ValidationError

try:
    StrictQuantity.model_validate({"value": "3"})
except ValidationError as error:
    print(error)
```

출력

```
1 validation error for StrictQuantity
value
  Input should be a valid integer [type=int_type, input_value='3', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/int_type
```

### 모델 전체 strict 설정

```python
class StrictRecord(BaseModel):
    # strict=True는 가능한 자동 형 변환을 줄입니다.
    # extra="forbid"는 정의되지 않은 필드를 막습니다.
    model_config = ConfigDict(strict=True, extra="forbid")

    count: int
    active: bool
```

“strict”라는 단어는 두 문맥에서 나온다. Pydantic strict는 Python 입력 변환 정책

OpenAI function/Structured Output의 strict는 Schema 준수 정책이다

목적이 비슷하지만 같은 설정은 아니다


Pydantic의 strict : 입력 타입을 엄격하게 검사

OpenAI Structured Output : strict는
모델 출력이 지정한 JSON Schema를 엄격하게 따르게 한다

## 6. Pydantic 모델에서 JSON Schema 읽기

`model_json_schema()`를 사용하면 Pydantic 정의에서 JSON Schema를 생성할 수 있다.

```python
import json

schema = ClassifiedTicket.model_json_schema()
print(json.dumps(schema, ensure_ascii=False, indent=2))
```

출력

```
{
  "additionalProperties": false,
  "properties": {
    "category": {
      "enum": [
        "delivery",
        "refund",
        "account",
        "other"
      ],
      "title": "Category",
      "type": "string"
    },
    "summary": {
      "maxLength": 200,
      "minLength": 1,
      "title": "Summary",
      "type": "string"
    },
    "priority": {
      "maximum": 3,
      "minimum": 1,
      "title": "Priority",
      "type": "integer"
    },
    "needs_review": {
      "title": "Needs Review",
      "type": "boolean"
    }
  },
  "required": [
    "category",
    "summary",
    "priority",
    "needs_review"
  ],
  "title": "ClassifiedTicket",
  "type": "object"
}
```

출력에는 다음 정보가 포함된다

- 객체 type
- properties
- required
- Literal에서 생성된 enum
- Field의 description
- 숫자와 문자열 제약

### Schema를 읽을 때 확인할 것

```python
# 최상위 properties에 category가 있는지 확인합니다.
assert "category" in schema["properties"]

# category의 enum이 원하는 네 값인지 확인합니다.
category_schema = schema["properties"]["category"]
assert set(category_schema["enum"]) == {
    "delivery",
    "refund",
    "account",
    "other",
}
```

Pydantic이 생성한 Schema를 그대로 외부 API에서 지원하는지 확인해야 한다.

API가 JSON Schema의 일부 기능만 지원한다면 호환되지 않는 제약이 포함될 수 있다.

OpenAI Python SDK의 Pydantic helper를 사용하면 지원되는 Schema 형태로 연결하는 과정을 단순화할 수 있다.

## 7. 기본 실습: 문서 분석 모델 만들기

```python
# [실습 목적]
# 문서 분석 결과를 Pydantic 모델로 정의하고 정상/오류 데이터를 검증합니다.

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DocumentType = Literal["invoice", "receipt", "contract", "other"]

class DocumentAnalysis(BaseModel):
    """문서 한 건의 분류/요약/검토 여부를 표현합니다."""

    # 정의되지 않은 필드를 막아 출력 계약을 명확하게 유지합니다.
    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType = Field(
        description="invoice, receipt, contract, other 중 하나입니다."
    )
    language: str = Field(
        min_length=2,
        max_length=10,
        description="ko, en과 같은 언어 코드입니다.",
    )
    summary: str = Field(
        min_length=1,
        max_length=200,
        description="문서 핵심을 한 문장으로 요약합니다.",
    )
    needs_review: bool = Field(
        description="문서 유형이나 내용이 불확실하면 true입니다."
    )
    review_reason: str | None = Field(
        description="검토가 필요하지 않으면 null입니다."
    )

valid_data = {
    "document_type": "contract",
    "language": "ko",
    "summary": "서비스 이용 조건과 해지 절차를 정의한 계약서입니다.",
    "needs_review": False,
    "review_reason": None,
}

analysis = DocumentAnalysis.model_validate(valid_data)
print(analysis)
print(analysis.model_dump())

assert analysis.document_type == "contract"
assert analysis.review_reason is None
```

출력

```
document_type='contract' language='ko' summary='서비스 이용 조건과 해지 절차를 정의한 계약서입니다.' needs_review=False review_reason=None
{'document_type': 'contract', 'language': 'ko', 'summary': '서비스 이용 조건과 해지 절차를 정의한 계약서입니다.', 'needs_review': False, 'review_reason': None}
```

### 오류 데이터 확인

```python
invalid_data = {
    "document_type": "memo",  # Literal에 없는 값입니다.
    "language": "k",          # 최소 길이를 만족하지 않습니다.
    "summary": "",            # 빈 문자열입니다.
    "needs_review": "yes",    # bool 대신 문자열입니다.
    "review_reason": None,
    "extra_field": 123,        # extra="forbid"로 막힙니다.
}

try:
    DocumentAnalysis.model_validate(invalid_data)
except ValidationError as error:
    # errors()는 오류 위치, 종류, 메시지, 입력값을 dict 목록으로 반환합니다.
    for item in error.errors():
        print(item)
```

출력

```
{'type': 'literal_error', 'loc': ('document_type',), 'msg': "Input should be 'invoice', 'receipt', 'contract' or 'other'", 'input': 'memo', 'ctx': {'expected': "'invoice', 'receipt', 'contract' or 'other'"}, 'url': 'https://errors.pydantic.dev/2.13/v/literal_error'}
{'type': 'string_too_short', 'loc': ('language',), 'msg': 'String should have at least 2 characters', 'input': 'k', 'ctx': {'min_length': 2}, 'url': 'https://errors.pydantic.dev/2.13/v/string_too_short'}
{'type': 'string_too_short', 'loc': ('summary',), 'msg': 'String should have at least 1 character', 'input': '', 'ctx': {'min_length': 1}, 'url': 'https://errors.pydantic.dev/2.13/v/string_too_short'}
{'type': 'extra_forbidden', 'loc': ('extra_field',), 'msg': 'Extra inputs are not permitted', 'input': 123, 'url': 'https://errors.pydantic.dev/2.13/v/extra_forbidden'}
```

### 필드 사이 관계 확인하기

구조는 맞지만 다음 값은 업무상 이상하다
```python
inconsistent_data = {
    "document_type": "other",
    "language": "ko",
    "summary": "문서 일부가 잘려 유형을 확정하기 어렵습니다.",
    "needs_review": False,
    "review_reason": "문서 제목이 잘려 있습니다.",
}
```
`needs_review=False`인데 검토 이유가 있다. 

 model validator나 별도 비즈니스 규칙 함수로 검사한다

 ## 8. 연습 문제: 잘못된 입력 수정하기

 ### 문제 1

다음 모델에서 허용하지 않은 category가 들어오지 않도록 수정한다

```python
class Ticket(BaseModel):
    category: str
    priority: int
```

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["delivery", "refund", "account"]
    priority: int = Field(ge=1, le=3)
```


`Literal`로 허용값을 제한하고, `Field`로 우선순위 범위를 제한했다

### 문제 2

`count="5"`가 정수 5로 자동 변환되지 않게 하려면 어떻게 해야 하나

```python
from pydantic import BaseModel, ConfigDict

class CountRecord(BaseModel):
    model_config = ConfigDict(strict=True)

    count: int
```

또는 필드 타입을 `StrictInt`로 지정할 수 있다

### 문제 3

Pydantic 검증에 성공했는데도 별도 비즈니스 검증이 필요한 예를 하나 작성하세요.

영수증의 `total`이 숫자이고 0 이상이어도, `sum(item.quantity * item.unit_price)`와 다를 수 있다.<br>
자료형과 범위는 Pydantic이 확인하지만 합계 일치 여부는 별도 규칙으로 확인해야 한다


## 9. 추가 실습: Responses API parse 사용하기

다음 실습은 실제 API 호출 비용이 발생할 수 있다. 

API key는 환경 변수 `OPENAI_API_KEY` 또는 Colab Secrets에 저장하고 코드에 직접 적지 않는다

```python
# [실습 목적]
# Pydantic 모델을 Responses API의 text_format으로 전달하고,
# 파싱된 결과를 response.output_parsed에서 읽습니다.

import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
client = OpenAI()

source_text = """
문서 제목: 서비스 이용 계약서
본 계약은 2026년 8월 1일부터 1년간 유효합니다.
중도 해지 시 30일 전에 서면 통지가 필요합니다.
""".strip()

response = client.responses.parse(
    model=MODEL,
    input=[
        {
            "role": "system",
            "content": (
                "입력 문서를 분류하고 요약하세요. "
                "근거가 부족하면 document_type은 other로 두고 "
                "needs_review를 true로 설정하세요."
            ),
        },
        {
            "role": "user",
            "content": source_text,
        },
    ],
    # SDK가 이 Pydantic 모델을 Structured Output 형식으로 연결합니다.
    text_format=DocumentAnalysis,
)

# output_parsed는 검증을 통과한 DocumentAnalysis 객체입니다.
parsed = response.output_parsed

if parsed is None:
    # 안전상 거절이나 incomplete 상태 등에서는 정상 객체가 없을 수 있습니다.
    print("파싱된 결과가 없습니다.")
    print("status:", response.status)
    print("raw output:", response.output)
else:
    print("Pydantic 객체:", parsed)
    print("문서 유형:", parsed.document_type)
    print("요약:", parsed.summary)
    print("dict 변환:", parsed.model_dump())
```

### 실행 후 확인할 것

- `type(parsed)`가 `DocumentAnalysis`인가
- `parsed.document_type`을 점 표기법으로 읽을 수 있나
- `response.output_text` 대신 `output_parsed`를 사용하는 이유는 무엇인가
- 불확실한 입력에서 `needs_review` 정책이 지켜지나

### 여러 입력 테스트하기

```python
# [실습 목적]
# 정상 문서와 정보가 부족한 문서를 같은 모델/Schema로 비교합니다.

test_documents = [
    "카페 영수증: 아메리카노 2잔, 총액 9,000원",
    "사진 아래쪽 일부만 남아 있으며 제목과 발행처를 확인할 수 없음",
]

for index, text in enumerate(test_documents, start=1):
    result = client.responses.parse(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "문서를 분류하고 한 문장으로 요약하세요. "
                    "판단 근거가 부족하면 needs_review=true로 표시하세요."
                ),
            },
            {"role": "user", "content": text},
        ],
        text_format=DocumentAnalysis,
    )

    print(f"case-{index}", result.output_parsed)
```

출력

```
case-1 document_type='receipt' language='ko' summary='카페에서 아메리카노 2잔을 구매했으며 총 결제금액은 9,000원입니다.' needs_review=False review_reason=None
case-2 document_type='other' language='ko' summary='사진의 아래쪽 일부만 남아 있어 문서의 제목과 발행처를 확인할 수 없습니다.' needs_review=True review_reason='문서의 핵심 식별 정보와 전체 내용이 누락되어 유형을 정확히 분류하기 어렵습니다.'
```

## 12. 이해도 점검

1. `BaseModel`과 `Field`의 역할을 설명
2. `Literal`은 JSON Schema의 어떤 개념과 비슷한가
3. `extra="forbid"`는 어떤 입력을 막나
4. Pydantic strict와 OpenAI strict가 같은 설정인가
5. `model_dump()`와 `model_json_schema()`의 차이는 무엇인가
6. `responses.parse()` 결과에서 파싱된 객체는 어디에서 읽나

- 정답 확인

1. `BaseModel`은 데이터 구조와 검증 모델을 정의하고, `Field`는 설명/범위/길이 같은 필드 제약을 추가한다
2. 허용값 목록을 정하는 `enum`과 비슷하다
3. 모델에 정의되지 않은 추가 필드가 들어오는 것을 막는다
4. 아닙니다. Pydantic strict는 Python 형 변환 정책이고, OpenAI strict는 API 출력의 Schema 준수 정책
5. `model_dump()`는 객체를 데이터 dict로 바꾸고, `model_json_schema()`는 모델 규칙을 JSON Schema로 만든다
6. `response.output_parsed`에서 읽는다

## 13. 이번 강의 요약

- Pydantic은 Python 타입 정의, 입력 검증, 직렬화, JSON Schema 생성을 연결한다
- `BaseModel`, `Field`, `Literal`, `ConfigDict(extra="forbid")`가 핵심
- 자동 형 변환이 필요한지, strict 검증이 필요한지 경계마다 결정해야 한다
- 중첩 객체는 작은 모델을 조합해 표현한다
- Pydantic 성공은 구조 검증 성공이며, 사실성과 비즈니스 규칙은 별도로 확인한다
- `client.responses.parse(..., text_format=Model)`로 Structured Output을 요청하고 `response.output_parsed`에서 객체를 읽을 수 있다.
- 거절과 incomplete 상태에서는 파싱 객체가 없을 수 있으므로 응답 상태도 함께 확인해야 한다.

