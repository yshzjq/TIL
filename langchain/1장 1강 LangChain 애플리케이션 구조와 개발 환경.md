---
title: 1장 1강 LangChain 애플리케이션 구조와 개발 환경
date: 2026-09-29
updated: 2026-09-29
description: KANT 강의 '1장 1강 LangChain 애플리케이션 구조와 개발 환경' 정리
---

## 1. LangChain

LangChain은 프롬프트, LLM, 출력 처리, 검색(RAG), Tool 같은 기능을 각각 역할별로 나누고, 이 기능들을 연결해서 LLM 애플리케이션을 구성하도록 도와주는 프레임워크다.

## 2. OpenAI SDK로 직접 호출하기

OpenAI SDK로 모델을 직접 호출할 때는 애플리케이션 코드가 메시지 작성, API 요청, 응답 본문 추출을 모두 담당한다

### 2.1 역할별 메시지 만들기

채팅 모델은 보통 역할과 본문을 가진 메시지 목록을 입력으로 받는다

```python
# 모델이 따라야 할 역할과 답변 원칙을 system 메시지에 넣습니다.
system_message = {
    "role": "system",
    "content": "당신은 어려운 AI 용어를 쉽게 설명하는 튜터입니다.",
}

# 실행할 때마다 바뀌는 질문을 user 메시지에 넣습니다.
user_message = {
    "role": "user",
    "content": "프롬프트가 무엇인지 두 문장으로 설명해 주세요.",
}

# 메시지는 모델에 전달할 순서대로 목록에 담습니다.
messages = [system_message, user_message]
```

`system` 메시지는 모델의 역할과 응답 원칙을 정한다, `user` 메시지는 사용자가 실제로 묻는 내용을 전달한다


### 2.2 모델 호출하고 답변 꺼내기

다음 코드는 OpenAI SDK로 메시지를 직접 전달하는 가장 작은 예

```python
# 운영체제 환경변수를 읽기 위해 os를 가져옵니다.
import os

# 공식 OpenAI Python SDK의 클라이언트를 가져옵니다.
from openai import OpenAI

# .env 파일의 값을 운영체제 환경변수로 불러옵니다.
from dotenv import load_dotenv

# OPENAI_API_KEY를 읽기 전에 .env 파일을 먼저 불러옵니다.
load_dotenv()

messages = [
    {"role": "user", "content": "안녕하세요!"}
]

# 코드에 키를 직접 쓰지 않고 환경변수에서 API Key를 읽습니다.
api_key = os.getenv("OPENAI_API_KEY")

# 읽은 API Key로 OpenAI 클라이언트를 만듭니다.
client = OpenAI(api_key=api_key)

# 모델명과 메시지 목록을 직접 지정해 요청을 보냅니다.
response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
    messages=messages,
)

# 첫 번째 응답 후보의 assistant 메시지에서 문자열 본문을 꺼냅니다.
answer = response.choices[0].message.content

# 최종 답변을 화면에 출력합니다.
print(answer)
```

이 방식은 코드의 흐름이 눈에 바로 보인다는 장점이 있다. 

**프롬프트 작성 방식이나 응답 구조가 바뀌면 메시지 생성 코드와 응답 추출 코드를 직접 수정해야 한다**

### 2.3 직접 호출 코드가 맡은 일

위 코드에는 세 가지 책임이 함께 들어 있습니다.

| 책임 | 코드에서 하는 일 |
| --- | --- |
| 입력 준비 | system과 user 메시지를 직접 만듭니다. |
| 모델 실행 | 모델명과 메시지를 API에 전달합니다. |
| 출력 정리 | 응답 객체 안에서 최종 문자열을 직접 꺼냅니다. |

작은 프로그램에서는 이 구성이 가장 이해하기 쉽고 충분히 실용적이다


## 3. LangChain의 세 단계로 나누기

LangChain은 앞에서 한 작업을 `Prompt`, `Model`, `Parser` 세 단계로 나눈다

Prompt, Model, Parser의 기본 흐름

#### - Prompt

질문과지시를 메시지로 만든다

#### - Model

메시지를 읽고 답변을 만든다

#### - Parser

답변을 앱이 쓰기 쉬운 형태로 정리한다.

### 3.1 Prompt

Prompt는 입력값을 모델이 받을 메시지로 바꾼다.

```python
# 역할별 메시지를 만들 수 있는 ChatPromptTemplate을 가져옵니다.
from langchain_core.prompts import ChatPromptTemplate

# system 메시지는 고정하고, human 메시지의 질문만 실행할 때 바뀌게 만듭니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 어려운 AI 용어를 쉽게 설명하는 튜터입니다."),
        ("human", "{question}"),
    ]
)

# question 변수에 실제 질문을 넣어 역할별 메시지 목록을 만듭니다.
messages = prompt.format_messages(
    question="프롬프트가 무엇인지 두 문장으로 설명해 주세요."
)
```

### 3.2 Model

Model은 Prompt가 만든 메시지를 받아 실제 LLM을 호출한다

```python
# .env에서 지정한 권장 모델을 읽습니다.
import os
from dotenv import load_dotenv

# OpenAI 모델을 LangChain 방식으로 연결하는 클래스를 가져옵니다.
from langchain_openai import ChatOpenAI

# 사용할 모델을 지정해 ChatModel 객체를 만듭니다.
load_dotenv()

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"))

# Prompt가 만든 메시지를 모델에 전달해 AIMessage 응답을 받습니다.
ai_message = model.invoke(messages)
```

`model.invoke()`의 결과는 일반 문자열이 아니라 `AIMessage`입니다. `AIMessage`에는 답변 본문과 함께 모델 호출에 관한 부가 정보가 들어갈 수 있다.

### 3.3 Parser

Parser는 모델 응답을 애플리케이션에서 사용하기 쉬운 형태로 바꾼다.

```python
# AIMessage의 본문을 문자열로 바꾸는 Parser를 가져옵니다.
from langchain_core.output_parsers import StrOutputParser

# 문자열 변환을 담당할 Parser 객체를 만듭니다.
parser = StrOutputParser()

# AIMessage를 Parser에 전달해 최종 문자열을 얻습니다.
answer = parser.invoke(ai_message)

# 화면에 표시할 문자열을 출력합니다.
print(answer)
```

### 3.4 세 단계의 입출력

| 단계 | 받는 값 | 내보내는 값 | 한 문장 역할 |
| --- | --- | --- | --- |
| Prompt | 질문이 담긴 입력값 | 역할별 메시지 | 모델에게 보낼 내용을 만듭니다. |
| Model | 역할별 메시지 | `AIMessage` | LLM을 호출해 응답을 만듭니다. |
| Parser | `AIMessage` | 문자열 | 앱에서 쓰기 쉬운 값으로 바꿉니다. |


## 4. 두 방식의 코드와 결과 비교

### 4.1 공통점

두 방식 모두 같은 OpenAI 모델 API를 호출합니다. API Key가 필요하고, 모델 사용량에 따라 비용이 발생하며, 답변 내용은 실행할 때마다 조금 달라질 수 있다.


### 4.2 차이점

| 비교 항목 | OpenAI SDK 직접 호출 | LangChain 방식 |
| --- | --- | --- |
| 메시지 구성 | 딕셔너리 목록을 직접 만듭니다. | Prompt 객체가 메시지를 만듭니다. |
| 모델 호출 | SDK 클라이언트를 직접 호출합니다. | ChatModel의 `invoke()`를 호출합니다. |
| 결과 추출 | 응답 객체에서 본문을 직접 찾습니다. | Parser가 원하는 자료형으로 바꿉니다. |
| 코드 규모 | 작은 기능에서 짧고 단순합니다. | 단계가 많아질수록 구조가 선명합니다. |
| 변경 범위 | 관련 코드를 직접 찾아 수정합니다. | 바뀌는 단계만 교체하기 쉽습니다. |


