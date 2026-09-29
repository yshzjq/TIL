---
title: 2장 2강 ChatModel과 Output Parser 연결
date: 2026-09-29
updated: 2026-09-29
description: KANT 강의 '2장 2강 ChatModel과 Output Parser 연결' 정리
---

## 1. 기본 체인의 전체 흐름

Prompt, Model, Parser의 입출력 흐름

<img src="{{ '/assets/images/uploads/langchain/2-2_01_Prompt_Model_Parser_흐름.png' | relative_url }}" alt="2-2_01_Prompt_Model_Parser_흐름.png" loading="lazy">

각 단계는 앞 단계의 출력을 입력으로 받는다

```
입력 딕셔너리
→ Prompt가 역할별 메시지를 만든다
→ ChatModel이 AIMessage를 만든다
→ Parser가 앱에서 사용할 자료형으로 바꾼다
```

### 단계별 자료형

| 단계 | 받는 값 | 내보내는 값 |
| --- | --- | --- |
| `ChatPromptTemplate` | 입력 변수의 값 | 역할별 메시지 목록 |
| `ChatOpenAI` | 역할별 메시지 목록 | `AIMessage` |
| `StrOutputParser` | `AIMessage` | `str` |
| List Parser | `AIMessage` | `list[str]` |
| JSON Parser | `AIMessage` | `dict` 또는 `list` | 

## 2. Prompt와 ChatModel 연결하기

### 2.1 Prompt 만들기
```python
# 역할 기반 Prompt를 만들기 위해 ChatPromptTemplate을 가져옵니다.
from langchain_core.prompts import ChatPromptTemplate

# 고정 답변 원칙과 실행할 때 바뀌는 질문을 분리합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 AI 입문 수업의 튜터입니다."),
        ("human", "{question}"),
    ]
)

# question 변수에 실제 질문을 넣어 메시지 목록을 만듭니다.
messages = prompt.format_messages(
    question="LangChain의 세 가지 기본 구성 요소를 설명해 주세요."
)
```

### 2.2 ChatModel 호출하기

```python
# .env에서 지정한 권장 모델을 읽습니다.
import os

# .env의 환경변수를 현재 Python 프로세스로 불러옵니다.
from dotenv import load_dotenv

# OpenAI ChatModel을 LangChain 방식으로 사용하는 클래스를 가져옵니다.
from langchain_openai import ChatOpenAI

# 프로젝트 루트의 .env를 읽습니다.
load_dotenv()

# ChatOpenAI는 OPENAI_API_KEY 환경변수를 사용해 인증합니다.
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))

# Prompt가 만든 메시지를 모델에 전달합니다.
ai_message = model.invoke(messages)

# 모델 응답 객체의 종류와 본문을 확인합니다.
print(type(ai_message).__name__)
print(ai_message.content)
```

출력 구조

```
AIMessage
모델이 생성한 답변
```

### AIMessage를 그대로 사용하는 경우

`AIMessage`에는 답변 본문 외에도 모델이 제공한 부가 정보가 들어갈 수 있습니다.
<br>
그런 정보가 필요하다면 Parser를 붙이기 전에 `AIMessage`를 유지해야 합니다.



## 3. StrOutputParser로 문자열 만들기

`StrOutputParser`는 `AIMessage`의 텍스트 본문을 일반 문자열로 바꾼다

### 3.1 Parser 연결

```python
# AIMessage의 본문을 문자열로 바꾸는 Parser를 가져옵니다.
from langchain_core.output_parsers import StrOutputParser

# 문자열 변환을 담당할 Parser 객체를 만듭니다.
parser = StrOutputParser()

# 모델이 반환한 AIMessage를 Parser에 전달합니다.
text_result = parser.invoke(ai_message)

# 변환된 값과 문자열 여부를 함께 확인합니다.
print(text_result)
print(isinstance(text_result, str))
```

```
모델이 생성한 답변
True
```

### 3.2 API 호출 없이 Parser만 확인하기

Parser의 동작만 확인할 때는 `AIMessage`를 직접 만들 수 있다.

```python
# 테스트용 AIMessage를 만들기 위해 메시지 클래스를 가져옵니다.
from langchain_core.messages import AIMessage

# 실제 API 호출 대신 고정된 본문을 가진 메시지를 만듭니다.
sample_message = AIMessage(content="Prompt는 모델에 전달할 메시지를 만듭니다.")

# StrOutputParser로 AIMessage의 본문을 꺼냅니다.
sample_result = StrOutputParser().invoke(sample_message)

# 결과와 문자열 여부를 출력합니다.
print(sample_result)
print(isinstance(sample_result, str))
```

이 코드는 실제 모델을 호출하지 않으므로 비용이 발생하지 않고 결과가 항상 같습니다.

## 4. List Parser로 리스트 만들기

여러 키워드를 반복문이나 화면의 태그로 사용하려면 문자열보다 리스트가 편리하다
<br>
`CommaSeparatedListOutputParser`는 쉼표로 구분된 문자열을 `list[str]`로 바꾼다.

Output Parser에 따른 결과 자료형 비교

### 4.1 Parser의 형식 안내 확인

```python
# 쉼표 구분 문자열을 리스트로 바꾸는 Parser를 가져옵니다.
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# List Parser 객체를 만듭니다.
list_parser = CommaSeparatedListOutputParser()

# 모델에게 어떤 형식으로 답해야 하는지 알려 줄 문장을 확인합니다.
format_instructions = list_parser.get_format_instructions()

# Prompt에 넣을 형식 안내를 출력합니다.
print(format_instructions)
```

Parser는 모델이 자동으로 형식을 지키게 만드는 장치가 아니다

`get_format_instructions()`로 원하는 형식을 Prompt에 알려 주고, 실제 응답을 Parser로 변환하는 두 단계가 필요하다

### 4.2 List Parser를 포함한 단계별 실행

```python
# 형식 안내와 사용자 질문이 들어갈 역할 기반 Prompt를 만듭니다.
list_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "질문에 맞는 핵심 키워드만 반환하세요.\n"
            "출력 형식:{format_instructions}",
        ),
        ("human", "{question}"),
    ]
)

# Parser의 형식 안내와 실제 질문을 Prompt에 전달합니다.
list_messages = list_prompt.format_messages(
    format_instructions=list_parser.get_format_instructions(),
    question="LangChain 기본 체인의 구성 요소 세 개를 알려 주세요.",
)

# 모델이 쉼표로 구분된 답변을 생성하도록 메시지를 전달합니다.
list_ai_message = model.invoke(list_messages)

# 모델 응답을 Python 리스트로 변환합니다.
list_result = list_parser.invoke(list_ai_message)

# 변환된 리스트와 자료형을 확인합니다.
print(list_result)
print(type(list_result).__name__)
```

## 5. JSON Parser로 딕셔너리 만들기

JSON은 키와 값을 가진 데이터를 표현할 수 있다. 

`JsonOutputParser`는 모델이 만든 JSON 형식의 문자열을 Python 딕셔너리나 리스트로 변환한다

### 5.1 JSON 형식 안내와 Prompt

```python
# JSON 문자열을 Python 값으로 바꾸는 Parser를 가져옵니다.
from langchain_core.output_parsers import JsonOutputParser

# JSON Parser 객체를 만듭니다.
json_parser = JsonOutputParser()

# 필요한 키와 출력 형식 안내가 들어간 Prompt를 만듭니다.
json_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "질문을 분류하고 category와 reason 키를 가진 JSON 객체로 답하세요.\n"
            "출력 형식:{format_instructions}",
        ),
        ("human", "{question}"),
    ]
)

# 형식 안내와 분류할 질문을 메시지에 넣습니다.
json_messages = json_prompt.format_messages(
    format_instructions=json_parser.get_format_instructions(),
    question="PromptTemplate은 어떤 역할을 하나요?",
)
```

### 5.2 모델 호출과 JSON 변환

```python
# 모델에 JSON 형식의 답변을 요청합니다.
json_ai_message = model.invoke(json_messages)

# 모델 응답을 Python 값으로 변환합니다.
json_result = json_parser.invoke(json_ai_message)

# 이번 Prompt는 JSON 객체를 요구했으므로 dict인지 확인합니다.
if not isinstance(json_result, dict):
    raise TypeError("JSON 객체를 기대했지만 다른 자료형이 반환되었습니다.")

# category와 reason 키가 모두 있는지 확인합니다.
if "category" not in json_result or "reason" not in json_result:
    raise ValueError("JSON 결과에 category 또는 reason 키가 없습니다.")

# 후속 코드가 사용할 category 값을 키로 읽습니다.
print(json_result["category"])

# 전체 결과의 Python 자료형을 확인합니다.
print(type(json_result).__name__)
```

출력

```
교육
dict
```

### 5.3 JSON 변환 결과 확인



<img src="{{ '/assets/images/uploads/langchain/2-2_03_JSON_성공과실패.png' | relative_url }}" alt="2-2_03_JSON_성공과실패.png" loading="lazy">


### 5.4 Parser 오류 처리

```python
# LangChain Parser가 발생시키는 공통 오류 타입을 가져옵니다.
from langchain_core.exceptions import OutputParserException

# JSON 변환에 실패해도 프로그램 전체가 바로 종료되지 않도록 처리합니다.
try:
    # 모델 응답을 JSON 값으로 변환합니다.
    json_result = json_parser.invoke(json_ai_message)
except OutputParserException as error:
    # 실제 응답 전체 대신 오류 종류와 안내 문구를 출력합니다.
    print("모델 응답을 JSON으로 변환하지 못했습니다.")
    print(type(error).__name__)
```

## 6. Parser별 결과 비교하기

| Parser | 모델이 만들도록 요청할 형식 | Python 결과 | 적합한 예 |
| --- | --- | --- | --- |
| `StrOutputParser` | 일반 문장 | `str` | 설명, 요약, 답변 |
| `CommaSeparatedListOutputParser` | 쉼표로 구분한 항목 | `list[str]` | 짧은 키워드, 태그 |
| `JsonOutputParser` | JSON 객체 또는 배열 | `dict` 또는 `list` | 분류 결과, 여러 필드 |

Parser는 모델의 지식이나 답변의 사실성을 검사하지 않는다.

**응답 형식을 해석하고 Python 자료형으로 바꾸는 역할**을 담당한다

### Parser 선택 순서

1. 애플리케이션에서 최종적으로 필요한 Python 자료형을 정한다
2. 그 자료형에 맞는 Parser를 선택한다
3. Parser의 형식 안내를 Prompt에 포함한다
4. 모델 응답을 Parser로 변환한다
5. 변환 결과에 필요한 키와 값이 있는지 확인한다.

## 7. 완성 실습 실행하기

`실습.py`의 상단에는 다음 설정이 있다.

```python
# 한 번 실행할 예제를 text, list, json 중에서 선택합니다.
EXAMPLE = "text"
```

API 호출 횟수를 줄이기 위해 한 번에 예제 하나만 실행한다

| `EXAMPLE` 값 | 실행하는 Parser | 예상 자료형 |
| --- | --- | --- |
| `"text"` | `StrOutputParser` | `str` |
| `"list"` | `CommaSeparatedListOutputParser` | `list` |
| `"json"` | `JsonOutputParser` | `dict` |
