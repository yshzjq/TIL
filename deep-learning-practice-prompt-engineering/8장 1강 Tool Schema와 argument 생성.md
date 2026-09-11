---
title: 8장 1강 Tool Schema와 argument 생성
date: 2026-09-11
updated: 2026-09-11
description: KANT 강의 '8장 1강 Tool Schema와 argument 생성' 정리
---

## 1. Function Calling이란 무엇인가

Function Calling은 모델이 애플리케이션에 등록된 도구 중 어떤 도구를 사용할지 선택하고,<br>
그 도구에 필요한 인자를 구조화해 반환하도록 하는 기능입니다.

예를 들어 사용자가 다음과 같이 묻습니다.

```
주문 ORD-2026-0713 배송 상태를 알려 주세요.
```

모델은 자연어 답을 지어내는 대신 다음과 같은 Tool Call을 만들 수 있다.

```json
{
  "name": "get_delivery_status",
  "arguments": {
    "order_id": "ORD-2026-0713",
    "include_history": false
  }
}
```

중요한 점은 이 JSON이 **배송 조회 결과가 아니라 조회 요청**이라는 것이다

```
모델: 어떤 함수와 인자가 필요한지 결정
애플리케이션: 인자를 검증하고 실제 함수 실행
외부 시스템: 주문 데이터 반환
모델: Tool 결과를 사용해 최종 답변 작성
```

Tool Schema 구성요소

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/02_tool_schema_parts.png' | relative_url }" alt="02_tool_schema_parts.png" loading="lazy">

## 2. Tool Calling의 다섯 단계

Function Calling은 다음 다섯 단계로 진행된다

Tool Calling 다섯 단계

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/01_tool_calling_five_steps.png' | relative_url }" alt="01_tool_calling_five_steps.png" loading="lazy">

1. 애플리케이션이 사용 가능한 Tool Schema를 모델에 전달한다
2. 모델이 사용자 요청을 보고 `function_call` item을 생성한다
3. 애플리케이션이 arguments를 파싱·검증하고 실제 함수를 실행한다.
4. 애플리케이션이 `function_call_output`으로 실행 결과를 모델에 전달한다.
5. 모델이 Tool 결과를 바탕으로 최종 사용자 응답을 작성한다.

### 1차 응답과 2차 응답

```
1차 모델 응답
-> function_call

애플리케이션 함수 실행
-> function_call_output

2차 모델 응답
-> 최종 자연어 답변
```

한 번의 `responses.create()` 호출로 모든 과정이 자동 완료된다고 생각하면 안 됩다.

일반적인 client-executed function은 애플리케이션이 중간 실행을 담당한다

“중간 실행” = 모델이 Tool Call을 만든 뒤, 그 Tool을 네 프로그램이 실제로 실행하는 단계

client-executed function은 모델이 "무엇을 호출할지" 정하고, 네 프로그램이 "진짜로 호출해서 실행"하는 구조



### 먼저 확인할 실행 경계

Tool Calling의 실행 경계

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/09_execution_boundary.png' | relative_url }" alt="09_execution_boundary.png" loading="lazy">

Tool Calling의 핵심은 **모델과 애플리케이션의 책임을 분리하는 것**이다

```
모델
-> 사용할 Tool과 arguments를 제안합니다.

애플리케이션
-> JSON 파싱, Pydantic 검증, 권한·업무 규칙 확인을 수행합니다.

외부 시스템
-> 검증을 통과한 요청만 실제로 실행합니다.
```

모델이 만든 `arguments`는 신뢰할 수 있는 내부 데이터가 아니라 **검증이 필요한 외부 입력**으로 취급한다.

`strict=True`를 사용해도 주문 번호 형식, 사용자 권한, 금액 한도와 같은 업무 규칙까지 자동으로 보장되는 것은 아니다

strict=True는 “형식 검사”에 가깝고, 
<br>
실제 서비스에서 실행해도 되는지는 애플리케이션이 따로 확인해야 한다

## 3. Tool Schema의 구성요소

Responses API에서 function tool은 다음 형태로 정의한다

```python
DELIVERY_TOOL = {
    "type": "function",  # 함수형 Tool 정의임을 선언합니다.
    "name": "get_delivery_status",  # 모델과 라우터가 함께 사용하는 Tool의 고유 이름입니다.
    "description": "주문 ID로 현재 배송 상태와 예상 도착일을 조회합니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
    "parameters": {  # Tool arguments의 JSON Schema를 시작합니다.
        "type": "object",  # arguments가 JSON 객체 형태임을 지정합니다.
        "properties": {  # 허용할 arguments 필드와 각 필드의 규칙을 정의합니다.
            "order_id": {  # 조회 대상을 식별하는 주문 ID를 담습니다.
                "type": "string",  # 이 필드에는 문자열만 허용합니다.
                "description": "ORD-로 시작하는 주문 ID입니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
            },
            "include_history": {  # 배송 이력을 결과에 포함할지 선택합니다.
                "type": "boolean",  # 이 필드에는 true 또는 false만 허용합니다.
                "description": "배송 이력까지 필요하면 true입니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
            },
        },
        "required": ["order_id", "include_history"],  # 호출 시 반드시 포함해야 하는 필드 목록입니다.
        "additionalProperties": False,  # Schema에 없는 임의 필드가 들어오지 못하게 막습니다.
    },
    "strict": True,  # 정의한 Schema를 엄격하게 따르도록 요청합니다.
}
```

### `name`

애플리케이션 router가 실제 함수를 찾을 때 사용하는 식별자다.

모호한 이름보다 동작이 드러나는 이름을 사용한다

Tool Router = 모델이 준 Tool 이름을 실제 허용된 함수에 연결해주는 분배기

```
좋음: get_delivery_status
모호함: process
위험함: execute_anything
```

### `description`

모델이 언제 이 도구를 선택해야 하는지 판단하는 설명이다.

너무 짧으면 다른 Tool과 경계를 구분하기 어렵고, 실제로 하지 않는 기능을 적으면 잘못된 선택을 유도한다

### `parameters`

함수 인자에 대한 JSON Schema다.

### `strict`

strict function calling은 모델이 Schema를 따르는 arguments를 만들도록 강화한다

Tool Schema는 단순 API 문서가 아니라 모델의 도구 선택 설명서이자 애플리케이션의 검증 계약입니다.

## 4. strict Schema 작성 규칙

Strict Tool Schema 체크리스트

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/10_strict_schema_checklist.png' | relative_url }" alt="10_strict_schema_checklist.png" loading="lazy">

Strict mode에서는 객체마다 `additionalProperties: false`가 필요하며,
<br>
`properties`에 선언한 모든 필드를 `required`에 포함해야 한다
<br>
의미상 선택적인 값은 필드를 생략하는 대신 `null`을 허용하는 방식으로 표현한다

strict에서는 필드는 필수, 값은 null로 선택 가능

strict mode에서 기억할 핵심 규칙은 두 가지다

1. 객체의 `additionalProperties`는 `false`로 설정한다
2. `properties`의 모든 필드를 `required`에 포함한다.


모델 arguments와 검증 경계


<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/03_argument_boundary.png' | relative_url }" alt="03_argument_boundary.png" loading="lazy">

### 선택값 표현하기

strict Schema에서 선택적인 의미를 표현하려면 필드를 required에 포함하되 `null`을 허용할 수 있다.

```python
SEARCH_TOOL = {
    "type": "function",  # 함수형 Tool 정의임을 선언합니다.
    "name": "search_orders",  # 모델과 라우터가 함께 사용하는 Tool의 고유 이름입니다.
    "description": "고객 이메일과 선택 기간으로 주문을 검색합니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
    "parameters": {  # Tool arguments의 JSON Schema를 시작합니다.
        "type": "object",  # arguments가 JSON 객체 형태임을 지정합니다.
        "properties": {  # 허용할 arguments 필드와 각 필드의 규칙을 정의합니다.
            "customer_email": {  # 주문 검색에 사용할 고객 이메일입니다.
                "type": "string",  # 이 필드에는 문자열만 허용합니다.
            },
            "start_date": {  # 검색 시작일을 지정하며, 값이 없으면 null을 사용합니다.
                # 값이 없을 수 있지만 필드 자체는 required입니다.
                "type": ["string", "null"],  # 문자열 또는 null만 허용해 선택값을 표현합니다.
                "description": "YYYY-MM-DD 또는 null입니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
            },
        },
        "required": ["customer_email", "start_date"],  # 호출 시 반드시 포함해야 하는 필드 목록입니다.
        "additionalProperties": False,  # Schema에 없는 임의 필드가 들어오지 못하게 막습니다.
    },
    "strict": True,  # 정의한 Schema를 엄격하게 따르도록 요청합니다.
}
```

### Schema와 실제 함수의 불일치

위험한 상황

```
Schema: include_history 필수
실제 함수: include_history 인자 없음
```
또는 반대 상황

```
실제 함수: warehouse_id 필수
Schema: warehouse_id 정의 없음
```
Tool Schema, Pydantic argument model, 실제 Python 함수 signature가 같은 계약을 따라야 한다

모델에게 알려준 Tool 사용법, 검증 코드, 실제 함수의 인자 모양이 서로 똑같아야 한다.

Tool Schema는 사용 설명서, Pydantic은 검사표, 실제 함수는 실행부인데, 셋 다 같은 인자 이름과 타입을 사용해야 한다.


## 5. tool_choice와 parallel tool call


기본적으로 모델은 도구를 호출할지 판단한다

| 설정 | 동작 |
| --- | --- |
| `"auto"` | 도구를 0개, 1개 또는 여러 개 선택할 수 있습니다. |
| `"required"` | 하나 이상의 도구 호출을 요구합니다. |
| 특정 함수 지정 | 지정한 함수를 정확히 한 번 호출하게 합니다. |
| `"none"` | 도구를 사용하지 않게 합니다. |

### 특정 함수 강제하기

```python
forced_choice = {
    "type": "function",
    "name": "get_delivery_status",
}
```

도구가 반드시 필요한 테스트에서는 유용하지만, 
<br>
일반 사용자 요청에 무조건 Tool을 강제하면 불필요한 호출이 생길 수 있다.

### 병렬 호출 막기

parallel tool calls = 병렬 Tool 호출

False = Tools 0~1개 요청
True = Tools 여러 개를 한 번에 요청할 수 있음

```python
# 이 줄에서 실제 API 요청이 발생합니다. 반복 실행 시 token 비용과 rate limit을 확인합니다.
response = client.responses.create(
    model=MODEL,  # 호출에 사용할 모델 ID를 전달합니다.
    input="주문 ORD-2026-0713의 배송 상태를 조회해 주세요.",  # 모델이 처리할 사용자 입력 또는 누적 item을 전달합니다.
    tools=[DELIVERY_TOOL],  # 모델이 선택할 수 있는 Tool Schema 목록을 전달합니다.
    parallel_tool_calls=False,  # 수업 흐름을 단순하게 하기 위해 동시 Tool Call을 제한합니다.
)
```

## 6. function_call item 해석

function_call item 해부하기

<img src="{ '/assets/images/uploads/deep-learning-practice-prompt-engineering/function_call_item.png' | relative_url }" alt="function_call_item.png" loading="lazy">

`function_call` item을 읽을 때는 다음 네 필드를 구분

| 필드 | 의미 | 애플리케이션에서 할 일 |
| --- | --- | --- |
| `type` | item 종류 | `function_call`인지 먼저 확인합니다. |
| `name` | 모델이 선택한 Tool 이름 | allowlist router에 등록된 이름인지 확인합니다. |
| `arguments` | JSON 문자열 | `json.loads()` 후 Pydantic으로 검증합니다. |
| `call_id` | 호출과 결과를 연결하는 ID | `function_call_output.call_id`에 그대로 사용합니다. |

allowlist router는 미리 허용해둔 Tool 이름만 실제 함수로 연결해주는 router

배열의 첫 번째 item이 항상 Tool Call이라고 가정하지 않는다

Reasoning item이나 다른 output item이 함께 들어올 수 있으므로 `response.output`을 순회하며 `item.type`을 확인한다

Reasoning item은 모델이 내부적으로 추론 과정을 관리하기 위해 사용하는 output item 종류

`response.output`에는 text, reasoning, function_call 등 여러 item이 들어갈 수 있다.<br>
특정 배열 위치를 가정하지 말고 type을 확인해 순회한다

대표 function call item에는 다음 값이 있다.

```
item.type       -> "function_call"
item.name       -> 함수 이름
item.arguments  -> JSON 문자열
item.call_id    -> 결과를 연결할 식별자
```

### arguments는 문자열이다

```python
import json

# 모델의 arguments는 JSON 문자열입니다. 파싱 실패 가능성을 예외 처리해야 합니다.
arguments_dict = json.loads(item.arguments)
```

### `call_id`의 역할

1차 모델 응답의 function call과 애플리케이션이 보내는 function result를 연결한다

```json
{
  "type": "function_call_output",
  "call_id": "call_abc123",
  "output": "{...실행 결과...}"
}
```

`call_id`는 주문 ID나 함수 이름이 아니다.<br>
API가 같은 호출 흐름에서 Tool Call과 결과를 연결하는 식별자다.

## 7. 기본 실습: 배송 조회 Tool Schema 만들기

먼저 실제 API 호출 없이 Schema를 Python dict로 만들고 점검g한다

```python
# 배송 상태 조회 Tool의 이름, 설명, 인자 Schema를 정의합니다.

import json

DELIVERY_TOOL = {
    "type": "function",  # 함수형 Tool 정의임을 선언합니다.
    "name": "get_delivery_status",  # 모델과 라우터가 함께 사용하는 Tool의 고유 이름입니다.
    "description": (  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
        "주문 ID로 현재 배송 상태, 예상 도착일, 선택적 배송 이력을 조회합니다. "
        "주문을 변경하거나 취소하지 않습니다."
    ),
    "parameters": {  # Tool arguments의 JSON Schema를 시작합니다.
        "type": "object",  # arguments가 JSON 객체 형태임을 지정합니다.
        "properties": {  # 허용할 arguments 필드와 각 필드의 규칙을 정의합니다.
            "order_id": {  # 조회 대상을 식별하는 주문 ID를 담습니다.
                "type": "string",  # 이 필드에는 문자열만 허용합니다.
                "minLength": 1,  # 빈 문자열을 막기 위한 최소 길이입니다.
                "description": "조회할 주문 ID입니다. 예: ORD-2026-0713",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
            },
            "include_history": {  # 배송 이력을 결과에 포함할지 선택합니다.
                "type": "boolean",  # 이 필드에는 true 또는 false만 허용합니다.
                "description": "상태 변경 이력까지 필요하면 true입니다.",  # 모델이 Tool을 선택하고 인자를 채울 때 참고하는 설명입니다.
            },
        },
        # strict mode에서는 두 필드를 모두 required에 포함합니다.
        "required": ["order_id", "include_history"],  # 호출 시 반드시 포함해야 하는 필드 목록입니다.
        # 예상하지 못한 arguments를 막습니다.
        "additionalProperties": False,  # Schema에 없는 임의 필드가 들어오지 못하게 막습니다.
    },
    "strict": True,  # 정의한 Schema를 엄격하게 따르도록 요청합니다.
}

# Tool 결과나 로그는 JSON 문자열로 직렬화하되 민감 필드는 먼저 제거합니다.
print(json.dumps(DELIVERY_TOOL, ensure_ascii=False, indent=2))

# Tool 정의의 기본 조건을 로컬에서 확인합니다.
assert DELIVERY_TOOL["type"] == "function"
assert DELIVERY_TOOL["strict"] is True
assert DELIVERY_TOOL["parameters"]["additionalProperties"] is False
assert set(DELIVERY_TOOL["parameters"]["required"]) == {
    "order_id",
    "include_history",
}
```

출력

```
{
  "type": "function",
  "name": "get_delivery_status",
  "description": "주문 ID로 현재 배송 상태, 예상 도착일, 선택적 배송 이력을 조회합니다. 주문을 변경하거나 취소하지 않습니다.",
  "parameters": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string",
        "minLength": 1,
        "description": "조회할 주문 ID입니다. 예: ORD-2026-0713"
      },
      "include_history": {
        "type": "boolean",
        "description": "상태 변경 이력까지 필요하면 true입니다."
      }
    },
    "required": [
      "order_id",
      "include_history"
    ],
    "additionalProperties": false
  },
  "strict": true
}

```

### Pydantic argument model도 같이 정의하기

```python
# 모델이 만든 arguments를 다음 강의에서 검증할 Pydantic 모델을 준비합니다.

from pydantic import BaseModel, ConfigDict, Field

# 배송 조회 Tool이 허용하는 arguments 형식과 제약을 정의합니다.
class DeliveryArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")  # 자동 형 변환을 막고 정의하지 않은 필드를 거부합니다.

    order_id: str = Field(  # ORD-로 시작하는 주문 ID만 허용합니다.
        min_length=1,  # 빈 문자열을 막기 위한 최소 길이를 지정합니다.
        pattern=r"^ORD-[A-Za-z0-9-]+$",  # 허용할 문자열 형식을 정규표현식으로 제한합니다.
        description="ORD-로 시작하는 주문 ID입니다.",  # 필드 의미를 Schema와 문서에 함께 남깁니다.
    )
    include_history: bool  # 배송 이력을 포함할지 true/false로 받습니다.
```

Schema와 Pydantic 모델이 같은 필드와 자료형을 사용하는지 확인하세요.

## 8. 연습 문제: 잘못된 Schema 찾기

다음 strict Schema의 문제를 찾으세요.

```python
{
    "type": "function",  # 함수형 Tool 정의임을 선언합니다.
    "name": "get_order",  # 모델과 라우터가 함께 사용하는 Tool의 고유 이름입니다.
    "parameters": {  # Tool arguments의 JSON Schema를 시작합니다.
        "type": "object",  # arguments가 JSON 객체 형태임을 지정합니다.
        "properties": {  # 허용할 arguments 필드와 각 필드의 규칙을 정의합니다.
            "order_id": {"type": "string"},  # 조회 대상을 식별하는 주문 ID를 담습니다.
            # detail은 주문 상세 정보 포함 여부를 나타내는 선택 인자입니다.
            "detail": {"type": "boolean"}  # true이면 상세 정보를 요청하고 false이면 요약만 요청합니다.
        },
        "required": ["order_id"]  # 호출 시 반드시 포함해야 하는 필드 목록입니다.
    },
    "strict": True  # 정의한 Schema를 엄격하게 따르도록 요청합니다.
}
```

- `additionalProperties: false`가 없다.

- strict mode에서는 `properties`의 모든 필드를 required에 포함해야 하므로 `detail`도 required여야 한다.

- detail을 선택 의미로 사용하려면 `boolean` 또는 `null`을 허용하고 required에 포함하는 방식을 고려할 수 있다.

```python
GET_ORDER_TOOL = {
    "type": "function",

    # 모델과 router가 사용하는 Tool 이름
    "name": "get_order",

    # Tool arguments의 JSON Schema
    "parameters": {
        "type": "object",

        "properties": {
            # 주문 ID는 문자열
            "order_id": {
                "type": "string"
            },

            # detail은 값이 있을 수도 있고 없을 수도 있으므로
            # boolean 또는 null 허용
            "detail": {
                "type": ["boolean", "null"]
            }
        },

        # strict mode에서는 properties의 모든 필드를 required에 포함
        "required": [
            "order_id",
            "detail"
        ],

        # Schema에 없는 추가 필드는 허용하지 않음
        "additionalProperties": False
    },

    # Schema를 엄격하게 따르도록 요청
    "strict": True
}
```

### 문제 2

Tool description에 “주문을 조회하고 필요하면 취소합니다”라고 적었지만 실제 함수는 조회만 합니다. 무엇이 문제인가

모델이 취소 기능까지 있다고 잘못 판단할 수 있다.

description은 실제 함수가 수행하는 동작과 부작용을 정확히 표현해야 한다.

조회 Tool과 취소 Tool은 별도로 분리하고 취소에는 확인·권한 정책을 둬야 한다

### 문제 3

`tool_choice="required"`를 모든 요청에 사용하면 어떤 문제가 생길 수 있나요?

단순 인사나 Tool이 필요 없는 질문에서도 불필요한 도구 호출을 만들 수 있다.

실제 서비스에서는 auto를 기본으로 두고, 특정 workflow 단계나 테스트에서만 required/forced를 사용한다

workflow 단계 = 전체 작업 흐름을 쪼갠 각각의 처리 단계

## 9. 추가 실습: 모델이 만든 Tool Call 읽기


다음 코드는 실제 API 호출 비용이 발생할 수 있다. 

아직 로컬 함수를 실행하지 않고 function call만 확인한다

```python
# 모델에 배송 조회 Tool을 제공하고 response.output에서 function_call을 찾습니다.

import json
import os
from openai import OpenAI

# 환경 변수로 모델을 교체할 수 있게 하며, 기본값은 실습 시점의 예시입니다.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
# API key는 코드에 쓰지 않고 OPENAI_API_KEY 환경 변수 또는 Colab Secrets에서 읽습니다.
client = OpenAI()

# 이 줄에서 실제 API 요청이 발생합니다. 반복 실행 시 token 비용과 rate limit을 확인합니다.
response = client.responses.create(
    model=MODEL,  # 호출에 사용할 모델 ID를 전달합니다.
    input=[
        {
            "role": "user",  # 이 메시지를 보낸 주체의 역할을 지정합니다.
            "content": (  # 모델에 전달할 실제 텍스트 또는 멀티모달 항목을 담습니다.
                "주문 ORD-2026-0713의 현재 배송 상태를 알려 주세요. "
                "상세 이력은 필요 없습니다."
            ),
        }
    ],
    tools=[DELIVERY_TOOL],  # 모델이 선택할 수 있는 Tool Schema 목록을 전달합니다.
    tool_choice="auto",  # Tool을 자동 선택할지 특정 Tool로 강제할지 지정합니다.
    # 수업에서는 한 번에 하나의 Tool Call만 처리해 흐름을 단순하게 만듭니다.
    parallel_tool_calls=False,  # 수업 흐름을 단순하게 하기 위해 동시 Tool Call을 제한합니다.
)

function_calls = []

# output 배열 위치를 가정하지 않고 각 item의 type을 확인하며 순회합니다.
for item in response.output:
    # function_call이 아닌 item은 Tool 실행 경로로 보내지 않습니다.
    if item.type != "function_call":
        continue

    # arguments는 JSON 문자열이므로 json.loads()로 dict로 변환합니다.
    # 모델의 arguments는 JSON 문자열입니다. 파싱 실패 가능성을 예외 처리해야 합니다.
    arguments = json.loads(item.arguments)

    function_calls.append(
        {
            # call_id를 바꾸면 모델이 어떤 Tool Call의 결과인지 연결할 수 없습니다.
            "call_id": item.call_id,  # Tool 요청과 Tool 결과를 같은 호출로 연결하는 식별자입니다.
            "name": item.name,  # 모델과 라우터가 함께 사용하는 Tool의 고유 이름입니다.
            "arguments": arguments,  # 모델이 생성한 함수 인자를 JSON 문자열로 담습니다.
        }
    )

print("찾은 function call 수:", len(function_calls))
# 각 항목을 순회하며 현재 조건에 해당하는지 확인합니다.
for call in function_calls:
    # Tool 결과나 로그는 JSON 문자열로 직렬화하되 민감 필드는 먼저 제거합니다.
    print(json.dumps(call, ensure_ascii=False, indent=2))
```
