---
title: 7장 3강 Validation Error 처리와 출력 복구
date: 2026-09-11
updated: 2026-09-11
description: KANT 강의 '7장 3강 Validation Error 처리와 출력 복구' 정리
---


## 1. 출력 복구란 무엇인가

출력 복구는 모델 결과가 기대한 조건을 만족하지 못했을 때, 오류를 분류하고 정해진 정책으로 처리하는 과정이다

```
모델 출력
-> JSON/Pydantic 구조 확인
-> 비즈니스 규칙 확인
-> 정상 저장 또는 오류 분기
```

Validation 복구 흐름

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/08_validation_recovery.png' | relative_url }" alt="08_validation_recovery.png" loading="lazy">

복구 결과는 다음 중 하나가 될 수 있다.

- 그대로 사용합니다.
- 안전한 로컬 변환으로 수정합니다.
- 한 번 더 모델에 명확하게 요청합니다.
- 입력 정보를 사용자에게 다시 요청합니다.
- 사람 검토 큐로 보냅니다.
- 요청을 중단하고 오류를 반환합니다.

“복구”라는 말 때문에 모든 출력을 정상값으로 만들어야 한다고 생각하면 안 된다

실제 정보가 부족한데 임의 값을 채우는 것은 복구가 아니라 데이터 오염이다


## 2. 오류 처리의 큰 그림

출력 파이프라인은 다음 순서로 점검한다

```
API 요청 성공?
├─ 아니오 -> 네트워크/인증/Rate Limit 등 API 오류 처리
└─ 예
   ↓
응답 status 정상?
├─ refusal -> 거절 정책 처리
├─ incomplete -> incomplete reason 확인
└─ completed
   ↓
파싱 객체 존재?
├─ 아니오 -> 출력 item과 SDK 상태 확인
└─ 예
   ↓
Pydantic 구조 검증 통과?
├─ 아니오 -> 오류 위치 기록, 제한된 재시도/중단
└─ 예
   ↓
비즈니스 규칙 통과?
├─ 아니오 -> 사람 검토/입력 재요청
└─ 예 -> 저장/Tool/UI로 전달
```

Structured Outputs를 사용하면 Schema 준수 가능성이 크게 높아지지만, 
<br>
안전상 거절이나 출력 예산 부족, 업무 규칙 불일치까지 없어지는 것은 아니다

### 오류 정책 표

| 오류 | 대표 처리 |
| --- | --- |
| 인증 실패 | 재시도하지 않고 설정 확인 |
| Rate Limit | backoff 후 재시도 |
| refusal | 안전 안내 또는 허용 범위 재설명 |
| incomplete | 원인 확인 후 출력 예산/태스크 조정 |
| JSON 문법 오류 | legacy 출력이면 제한적 repair 또는 재요청 |
| Pydantic 타입 오류 | 오류 정보를 포함해 한 번 재요청 또는 중단 |
| 비즈니스 규칙 오류 | 원본 근거 재확인/사람 검토 |

## 3. 세 종류의 검증 오류

### 3-1. 문법 오류

문자열이 JSON으로 파싱되지 않는다

```
{"category": "delivery", "priority": 2,
```

닫는 중괄호가 없으므로 `json.loads()`에서 `JSONDecodeError`가 발생한다.

### 3-2. 구조 오류

JSON 문법은 맞지만 Pydantic 규칙을 어긴다

```json
{
  "category": "배송",
  "priority": "높음",
  "needs_review": false
}
```

- `category`가 허용 Literal에 없다.
- `priority`가 정수가 아니다.

### 3-3. 비즈니스 규칙 오류

구조는 맞지만 필드 관계가 모순된다

```json
{
  "needs_review": false,
  "review_reason": "문서가 너무 흐려 확인이 필요합니다."
}
```
자료형은 맞지만 업무 의미가 충돌한다

구조 검증과 업무 검증의 경계

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/05_pydantic_validation.png' | relative_url }" alt="05_pydantic_validation.png" loading="lazy">

## 4. ValidationError 읽기

Pydantic `ValidationError.errors()`는 오류 목록을 dict 형태로 반환한다

```python
# [실습 목적]
# ValidationError에서 필드 위치, 오류 유형, 메시지를 추출합니다.

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

class TicketResult(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    category: Literal["delivery", "refund", "account", "other"]
    priority: int = Field(ge=1, le=3)
    needs_review: bool
    review_reason: str | None

bad_data = {
    "category": "배송",   # 허용 Literal에 없습니다.
    "priority": "3",     # strict=True이므로 문자열 숫자를 허용하지 않습니다.
    "needs_review": False,
    # review_reason이 누락되었습니다.
    "memo": "추가 필드",  # extra="forbid"로 막힙니다.
}

try:
    TicketResult.model_validate(bad_data)
except ValidationError as error:
    for item in error.errors():
        # loc는 오류가 발생한 중첩 경로입니다.
        location = ".".join(str(part) for part in item["loc"])
        error_type = item["type"]
        message = item["msg"]

        print(f"위치={location} | 유형={error_type} | 설명={message}")
```

오류

```
위치=category | 유형=literal_error | 설명=Input should be 'delivery', 'refund', 'account' or 'other'
위치=priority | 유형=int_type | 설명=Input should be a valid integer
위치=review_reason | 유형=missing | 설명=Field required
위치=memo | 유형=extra_forbidden | 설명=Extra inputs are not permitted
```

### 오류를 작은 구조로 정리하기

```python
from dataclasses import dataclass, asdict

@dataclass
class FieldError:
    """UI나 로그에 전달할 최소 오류 정보입니다."""

    path: str
    error_type: str
    message: str


# 오류가 날시
def flatten_validation_error(error: ValidationError) -> list[FieldError]:
    """Pydantic 오류 목록을 읽기 쉬운 dataclass 목록으로 바꿉니다."""
    result: list[FieldError] = []

    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"]) or "<root>"
        result.append(
            FieldError(
                path=path,
                error_type=item["type"],
                message=item["msg"],
            )
        )

    return result

try:
    TicketResult.model_validate(bad_data)
except ValidationError as error:
    flattened = flatten_validation_error(error)
    print([asdict(item) for item in flattened])
```

출력

```
[{'path': 'category', 'error_type': 'literal_error', 'message': "Input should be 'delivery', 'refund', 'account' or 'other'"}, {'path': 'priority', 'error_type': 'int_type', 'message': 'Input should be a valid integer'}, {'path': 'review_reason', 'error_type': 'missing', 'message': 'Field required'}, {'path': 'memo', 'error_type': 'extra_forbidden', 'message': 'Extra inputs are not permitted'}]
```

코드에서 error에 오류 정보가 들어가게 되고 그 변수를 이용해서 오류 내용들을 출력

로그에는 전체 사용자 입력을 그대로 남기기보다 오류 필드, 유형, 요청 ID, Prompt 버전처럼 필요한 정보만 남기면 좋다

## 5. 비즈니스 규칙 검증

Pydantic의 `model_validator`로 필드 사이 관계를 검사할 수 있다

```python
# [실습 목적]
# needs_review와 review_reason의 관계를 모델 수준에서 검증합니다.

from typing import Self
from pydantic import model_validator

class ReviewedTicket(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    category: Literal["delivery", "refund", "account", "other"]
    priority: int = Field(ge=1, le=3)
    needs_review: bool
    review_reason: str | None

    @model_validator(mode="after")
    def check_review_relationship(self) -> Self:
        """검토 여부와 이유가 서로 모순되지 않는지 확인합니다."""
        if self.needs_review and not self.review_reason:
            raise ValueError(
                "needs_review=true이면 review_reason이 필요합니다."
            )

        if not self.needs_review and self.review_reason is not None:
            raise ValueError(
                "needs_review=false이면 review_reason은 null이어야 합니다."
            )

        return self
```

복잡한 외부 조회가 필요한 규칙은 Pydantic validator 안에 넣기보다 <br>
별도 service 함수로 분리하는 편이 테스트하기 쉽다.

## 6. refusal과 incomplete 처리

Structured Output 요청에서도 안전상 거절이 발생할 수 있다. 

거절은 사용자가 제공한 Schema와 다른 형태로 표현될 수 있으므로 응답 item과 상태를 확인해야 한다

### 응답 상태 검사 함수

```python
# [실습 목적]
# parse 응답이 정상 객체인지, refusal/incomplete인지 분류합니다.

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")

@dataclass
class ParsedResponseResult(Generic[T]):
    status: str
    data: T | None
    refusal: str | None
    incomplete_reason: str | None
    request_id: str | None

def inspect_parsed_response(response: Any) -> ParsedResponseResult[Any]:
    """OpenAI Response 객체에서 대표 상태를 안전하게 읽습니다."""
    refusal_text: str | None = None

    # output은 text, refusal, reasoning 등 여러 item을 포함할 수 있습니다.
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                refusal_text = getattr(content, "refusal", None)

    incomplete_reason: str | None = None
    details = getattr(response, "incomplete_details", None)
    if getattr(response, "status", None) == "incomplete" and details is not None:
        incomplete_reason = getattr(details, "reason", None)

    return ParsedResponseResult(
        status=getattr(response, "status", "unknown"),
        data=getattr(response, "output_parsed", None),
        refusal=refusal_text,
        incomplete_reason=incomplete_reason,
        request_id=getattr(response, "_request_id", None),
    )
```

### 처리 원칙

```
refusal 있음
-> 거절 내용을 사용자에게 안전하게 전달
-> 같은 요청을 문구만 바꿔 반복하지 않음

status=incomplete
-> incomplete_reason 확인
-> max_output_tokens라면 출력 구조/예산 조정

output_parsed=None
-> output item과 상태를 로그
-> 정상값이라고 가정하지 않음
```

`response.output[0].content[0]`처럼 위치를 고정해서 읽지 않는다
<br>
응답은 여러 종류의 item을 포함할 수 있다.

## 7. 기본 실습: 검증 결과 객체 만들기

애플리케이션이 성공과 실패를 동일한 형식으로 다루도록 결과 객체를 만든다

```python
# [실습 목적]
# 외부 JSON 문자열을 파싱하고, Pydantic 검증 결과를 한 구조로 반환합니다.

import json
from dataclasses import dataclass
from typing import Any

@dataclass
class ValidationResult:
    ok: bool
    data: dict[str, Any] | None
    error_code: str | None
    errors: list[dict[str, str]]

def validate_ticket_json(raw_text: str) -> ValidationResult:
    """JSON 문법 오류와 Pydantic 오류를 구분해 반환합니다."""
    try:
        # legacy 모델 출력이나 외부 API 문자열을 Python 값으로 변환합니다.
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return ValidationResult(
            ok=False,
            data=None,
            error_code="invalid_json",
            errors=[
                {
                    "path": f"line{error.lineno}, column{error.colno}",
                    "message": error.msg,
                }
            ],
        )

    try:
        validated = ReviewedTicket.model_validate(loaded)
    except ValidationError as error:
        flattened = flatten_validation_error(error)
        return ValidationResult(
            ok=False,
            data=None,
            error_code="validation_error",
            errors=[asdict(item) for item in flattened],
        )

    return ValidationResult(
        ok=True,
        data=validated.model_dump(),
        error_code=None,
        errors=[],
    )

valid_json = """
{
  "category": "delivery",
  "priority": 3,
  "needs_review": true,
  "review_reason": "배송 완료 표시와 실제 수령 상태가 다릅니다."
}
"""

result = validate_ticket_json(valid_json)
print(result)

assert result.ok is True
assert result.data is not None
assert result.data["category"] == "delivery"
```

출력

```
ValidationResult(ok=True, data={'category': 'delivery', 'priority': 3, 'needs_review': True, 'review_reason': '배송 완료 표시와 실제 수령 상태가 다릅니다.'}, error_code=None, errors=[])
```


### 문법 오류 테스트

```python
broken_json = '{"category": "delivery",'
broken_result = validate_ticket_json(broken_json)

print(broken_result)
assert broken_result.error_code == "invalid_json"
```
출력
```
ValidationResult(ok=False, data=None, error_code='invalid_json', errors=[{'path': 'line1, column25', 'message': 'Expecting property name enclosed in double quotes'}])
```


### 업무 규칙 오류 테스트

```python
inconsistent_json = """
{
  "category": "other",
  "priority": 1,
  "needs_review": false,
  "review_reason": "정보가 부족합니다."
}
"""

inconsistent_result = validate_ticket_json(inconsistent_json)
print(inconsistent_result)

assert inconsistent_result.error_code == "validation_error"
```

출력

```
ValidationResult(ok=False, data=None, error_code='validation_error', errors=[{'path': '<root>', 'error_type': 'value_error', 'message': 'Value error, needs_review=false이면 review_reason은 null이어야 합니다.'}])
```

## 8. 연습 문제: 재시도 가능한 오류 구분


API key가 잘못된 인증 오류는 같은 요청을 즉시 재시도해야 하나


아니다

설정을 바꾸지 않은 재시도는 같은 오류를 반복한다 요청을 중단하고 secret 설정과 권한을 확인해야 한다

### 문제 2

`priority=5`가 들어와 Pydantic 범위 검증에 실패했습니다. 가능한 처리 두 가지를 적으세요.

1. 오류 메시지와 허용 범위 1~3을 포함해 모델에 한 번 재요청
2. 원본 근거가 애매하면 사람 검토 큐로 보낸다( "needs_review": True)

`5`를 임의로 `3`으로 잘라 저장하는 것은 의미를 바꿀 수 있으므로 신중해야 합니다.

### 문제 3

`needs_review=true`인데 `review_reason=null`인 결과는 구조 오류인가요, 비즈니스 규칙 오류인가


각 필드의 자료형만 보면 구조는 맞다.

두 필드의 관계 규칙을 어겼으므로 비즈니스 규칙 오류다.
<br>
model validator로 검출할 수 있다.

## 9. 추가 실습: 제한된 복구 파이프라인

재시도는 무한 반복하지 않고 횟수와 조건을 제한한다

```python
# [실습 목적]
# Structured Output 호출을 최대 두 번 시도하고,
# 성공/거절/incomplete/예외를 서로 다른 결과로 반환합니다.

import os
from dataclasses import dataclass
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
client = OpenAI()

@dataclass
class ExtractionOutcome:
    ok: bool
    data: dict | None
    error_code: str | None
    message: str
    attempts: int
    request_id: str | None

def extract_ticket_with_retry(text: str, max_attempts: int = 2) -> ExtractionOutcome:
    """구조화된 문의 분석을 제한된 횟수만 시도합니다."""
    last_message = "알 수 없는 오류"
    last_request_id: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.responses.parse(
                model=MODEL,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "고객 문의를 분류하세요. 정보가 부족하면 "
                            "category=other, needs_review=true로 설정하고 "
                            "review_reason에 부족한 정보를 적으세요."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                text_format=ReviewedTicket,
                max_output_tokens=400,
            )
        except Exception as error:
            # 10장에서 API 예외 유형별 재시도를 더 정확히 구현합니다.
            # 여기서는 예외를 삼키지 않고 마지막 메시지만 기록합니다.
            last_message = f"API 호출 실패:{type(error).__name__}"
            continue

        last_request_id = getattr(response, "_request_id", None)
        inspected = inspect_parsed_response(response)

        if inspected.refusal:
            return ExtractionOutcome(
                ok=False,
                data=None,
                error_code="refusal",
                message=inspected.refusal,
                attempts=attempt,
                request_id=last_request_id,
            )

        if inspected.status == "incomplete":
            last_message = f"응답 미완료:{inspected.incomplete_reason}"
            # 같은 출력 예산으로 반복하면 같은 실패가 날 수 있습니다.
            # 실제 운영에서는 태스크를 줄이거나 max_output_tokens를 조정합니다.
            continue

        if inspected.data is not None:
            return ExtractionOutcome(
                ok=True,
                data=inspected.data.model_dump(),
                error_code=None,
                message="검증 완료",
                attempts=attempt,
                request_id=last_request_id,
            )

        last_message = "파싱 객체가 없습니다."

    return ExtractionOutcome(
        ok=False,
        data=None,
        error_code="exhausted",
        message=last_message,
        attempts=max_attempts,
        request_id=last_request_id,
    )
```

### 이 코드의 한계

- 모든 Exception을 같은 방식으로 처리한다 - 재시도가 의미 없는 오류도 있다는 것
- Rate Limit과 인증 실패를 구분하지 않는다. - <br>
오류별로 재시도 여부를 다르게 결정해야 한다<br>
429, 401 같은 오류 코드들은 시간이 지나면 되는 것도 있고 입력값이 틀려서 안되는 것들 을 구분하지 않는다.
- backoff가 없다.
- API SDK의 기본 재시도와 중복될 수 있다.



서버 오류였다면 기다렸다가 요청하는 게 더 적절할 수 있다.
이 전략이 backoff


Rate Limit은 쉽게 말하면 API를 너무 많이 또는 너무 빠르게 호출해서 걸리는 사용량 제한



## 12. 이해도 점검

1. JSON 문법 오류와 Pydantic 구조 오류의 차이는 무엇인가
2. `ValidationError.errors()`의 `loc`는 무엇을 나타내나
3. 비즈니스 규칙 오류 예시를 하나 적어 보세요.
4. refusal과 validation error는 같은 오류인가
5. `status=incomplete`일 때 무엇을 확인해야 하나
6. 재시도 횟수를 제한해야 하는 이유는 무엇인가

- 정답 확인

1. 문법 오류는 JSON 자체를 파싱할 수 없고, 구조 오류는 JSON은 읽히지만 자료형/필수 필드/범위 규칙을 어긴다
2. 중첩 데이터에서 오류가 발생한 필드 경로를 나타낸다
3. `needs_review=false`인데 검토 이유가 존재하거나, 영수증 항목 합계와 total이 다른 경우가 있다.
4. 아닙니다. refusal은 안전 정책에 따른 모델 거절이고, validation error는 받은 데이터가 모델 규칙을 어긴 경우다
5. `incomplete_details.reason`, 특히 `max_output_tokens` 도달 여부를 확인한다
6. 무한 반복과 비용 증가를 막고, 설정을 바꾸지 않은 같은 실패를 반복하지 않기 위해서디
