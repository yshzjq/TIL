---
title: 3장 1강 LCEL pipe 연산자와 Runnable 인터페이스
date: 2026-09-30
updated: 2026-09-30
description: KANT 강의 '3장 1강 LCEL pipe 연산자와 Runnable 인터페이스' 정리
---

# 3-1강: Prompt, Model, Parser를 LCEL pipe로 연결하기

## 1. 학습 목표

이번 강의에서는 Prompt, Model, Parser를 하나의 흐름으로 연결하는 방법을 배웁니다.

- Prompt, Model, Parser를 하나의 체인으로 연결할 수 있습니다.
- LCEL의 `|` 연산자가 앞 단계의 출력을 다음 단계의 입력으로 전달한다는 것을 이해합니다.
- `Runnable`의 기본 실행 방식을 이해합니다.
- 완성된 체인에 입력을 전달하고 `invoke()`로 결과를 받을 수 있습니다.

이번 강의에서는 가장 기본적인 형태인 다음 구조를 다룹니다.

```python
prompt | model | parser
```

`batch()`와 `stream()`은 다음 강의에서 다룹니다.

---

## 2. 단계별 실행 방식

기존에는 Prompt, Model, Parser를 각각 따로 실행했습니다.

```python
prompt_value = prompt.invoke(
    {"question": "LCEL은 무엇인가요?"}
)

ai_message = model.invoke(prompt_value)

raw_text = parser.invoke(ai_message)

final_text = str(raw_text)

print(final_text)
```

흐름은 다음과 같습니다.

```text
입력 dict
   ↓
Prompt
   ↓
Model
   ↓
Parser
   ↓
최종 문자열
```

이 방식은 각 단계의 결과를 직접 확인하기 쉽지만, 같은 구조를 반복해서 사용할 때는 코드가 길어질 수 있습니다.

---

# 3. LCEL

LCEL은 **LangChain Expression Language**의 약자입니다.

LangChain의 여러 구성 요소를 연결해 실행 흐름을 표현하는 방법입니다.

가장 기본적인 형태는 다음과 같습니다.

```python
prompt | model | parser
```

읽는 순서는 왼쪽에서 오른쪽입니다.

```text
Prompt의 출력
    ↓
Model의 입력

Model의 출력
    ↓
Parser의 입력
```

`|`는 LangChain의 Runnable 사이에서 앞 단계와 뒤 단계를 연결하는 **pipe 연산자**로 사용됩니다.

### 자료형 흐름

| 단계 | 구성 요소 | 입력 | 출력 |
|---|---|---|---|
| 시작 | 입력 | - | `dict` |
| 1단계 | `ChatPromptTemplate` | `dict` | 역할별 메시지 |
| 2단계 | `ChatOpenAI` | 역할별 메시지 | `AIMessage` |
| 3단계 | `StrOutputParser` | `AIMessage` | `str` |

중요한 점은 `|`가 자동으로 모든 자료형을 바꾸는 것은 아니라는 것입니다.

앞 단계의 출력을 다음 단계가 받을 수 있어야 합니다.

---

# 4. Runnable

`Runnable`은 LangChain에서 **입력을 받아 작업한 뒤 출력을 반환하는 기본 실행 방식**입니다.

이번 강의에서 다음 객체들은 모두 Runnable 방식으로 실행할 수 있습니다.

- `ChatPromptTemplate`
- `ChatOpenAI`
- `StrOutputParser`
- 세 객체를 연결한 전체 `chain`

그래서 모두 `invoke()`를 사용할 수 있습니다.

```python
prompt_result = prompt.invoke(
    {"question": "Runnable은 무엇인가요?"}
)

model_result = model.invoke(prompt_result)

raw_parser_result = parser.invoke(model_result)

parser_result = str(raw_parser_result)
```

## Runnable의 장점

- 각 단계를 같은 방식으로 실행할 수 있습니다.
- 여러 단계를 `|`로 연결할 수 있습니다.
- 연결한 전체 체인도 하나의 Runnable처럼 실행할 수 있습니다.
- 체인을 만든 뒤 입력만 바꾸어 재사용할 수 있습니다.

## 실행 메서드

| 메서드 | 의미 |
|---|---|
| `invoke()` | 입력 한 건을 실행하고 결과 한 개를 받음 |
| `batch()` | 여러 입력을 한꺼번에 실행하고 결과 목록을 받음 |
| `stream()` | 결과를 나누어 받음 |

이번 강의에서는 `invoke()`만 사용합니다.

---

# 5. Prompt, Model, Parser 준비하기

## 필요한 모듈

```python
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
```

---

## Prompt 만들기

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 LangChain 입문 수업의 튜터입니다. "
            "전문 용어를 먼저 풀어 쓰세요.",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)
```

Prompt에는 `{question}`이라는 입력 변수가 있습니다.

따라서 실행할 때도 같은 이름의 키를 가진 딕셔너리를 전달해야 합니다.

```python
{
    "question": "LCEL은 무엇인가요?"
}
```

---

## Model 만들기

```python
import os

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
)
```

모델 객체를 만드는 것만으로 API가 호출되지는 않습니다.

실제 호출은 `invoke()`를 실행할 때 발생합니다.

---

## Parser 만들기

```python
parser = StrOutputParser()
```

`StrOutputParser`는 모델의 `AIMessage`에서 답변 본문을 문자열 형태로 꺼냅니다.

---

# 6. LCEL로 체인 연결하기

Prompt, Model, Parser를 다음과 같이 연결합니다.

```python
chain = prompt | model | parser
```

이 코드는 세 단계를 **연결만 하는 코드**입니다.

아직 모델이 실행된 것은 아닙니다.

연결된 체인의 클래스 이름을 확인할 수도 있습니다.

```python
print(type(chain).__name__)
```

일반적으로 다음과 같은 결과를 볼 수 있습니다.

```text
RunnableSequence
```

`RunnableSequence`는 여러 Runnable이 순서대로 연결된 객체입니다.

---

# 7. 체인 실행하기

## 입력 준비

```python
input_data = {
    "question": "LCEL의 pipe 연산자는 어떤 역할을 하나요?"
}
```

`question`이라는 키 이름은 Prompt의 `{question}`과 같아야 합니다.

## invoke 실행

```python
raw_result = chain.invoke(input_data)

result = str(raw_result)

print(result)
print(type(result).__name__)
```

내부에서는 다음 순서로 실행됩니다.

```text
1. input_data의 question 값을 Prompt에 넣음
2. Prompt가 system, human 메시지를 만듦
3. Model이 메시지를 읽고 AIMessage를 만듦
4. Parser가 AIMessage의 본문을 문자열로 바꿈
5. invoke()가 최종 결과를 반환
```

---

# 8. 체인 생성과 실행의 차이

### 체인 생성

```python
chain = prompt | model | parser
```

이 단계에서는 구조만 만듭니다.

API 호출은 발생하지 않습니다.

### 체인 실행

```python
result = chain.invoke(input_data)
```

`invoke()`가 실행되어야 실제로 Prompt → Model → Parser 순서로 동작합니다.

즉,

```text
chain 생성
= 실행 순서 연결

invoke()
= 실제 실행
```

---

# 9. 단계별 실행과 체인 실행 비교

## 단계별 실행

```python
prompt_value = prompt.invoke(input_data)

ai_message = model.invoke(prompt_value)

raw_step_result = parser.invoke(ai_message)

step_result = str(raw_step_result)
```

## 체인 실행

```python
raw_chain_result = chain.invoke(input_data)

chain_result = str(raw_chain_result)
```

두 방식 모두 같은 구성 요소를 같은 순서로 실행합니다.

| 방식 | 특징 |
|---|---|
| 단계별 실행 | 각 단계의 결과를 확인하기 쉬움 |
| 체인 실행 | 코드가 간결하고 재사용하기 쉬움 |

처음 구조를 배우거나 오류를 확인할 때는 단계별 실행이 편하고, 구조를 이해한 뒤에는 체인 실행이 더 간단합니다.

---

# 10. 전체 실행 흐름

```text
.env 읽기
   ↓
ChatOpenAI 생성
   ↓
Prompt 생성
   ↓
Parser 생성
   ↓
prompt | model | parser
   ↓
입력 dict 전달
   ↓
chain.invoke()
   ↓
최종 문자열
```

실행은 다음과 같이 합니다.

```powershell
uv run python 실습.py
```

실행 결과에서 확인할 내용은 다음과 같습니다.

```text
체인 종류 → RunnableSequence
입력 → question 키를 가진 dict
최종 결과 → str
```

---

# 11. 자주 발생하는 오류

## 1) `OPENAI_API_KEY가 없습니다`

프로젝트의 `.env` 파일을 확인합니다.

```text
OPENAI_API_KEY=발급받은_API_Key
```

API Key를 코드에 직접 작성하거나 화면에 출력하지 않습니다.

---

## 2) Prompt 입력 변수 오류

Prompt가 다음과 같다면:

```python
"{question}"
```

입력도 다음과 같이 `question` 키를 사용해야 합니다.

```python
{
    "question": "LCEL은 무엇인가요?"
}
```

다음처럼 다른 키를 사용하면 오류가 발생합니다.

```python
{
    "query": "LCEL은 무엇인가요?"
}
```

---

## 3) Parser를 빼면 `AIMessage`가 반환됨

```python
message_chain = prompt | model
```

이 경우 결과는 문자열이 아니라 `AIMessage`입니다.

일반 문자열이 필요하면 마지막에 다음을 연결합니다.

```python
StrOutputParser()
```

---

## 4) 체인을 만들었는데 결과가 나오지 않음

```python
chain = prompt | model | parser
```

이 코드는 구조만 만듭니다.

실제 실행을 위해서는 다음과 같이 `invoke()`를 호출해야 합니다.

```python
result = chain.invoke(input_data)
```

그리고 화면에서 확인하려면 `print()`를 사용합니다.

---

## 5) `unsupported operand type` 오류

`|`로 연결하는 값 중 Runnable로 사용할 수 없는 값이 들어 있을 수 있습니다.

각 객체의 자료형을 확인합니다.

```python
print(type(prompt))
print(type(model))
print(type(parser))
```

---

# 12. RunnableLambda

일반 Python 함수도 `RunnableLambda`로 감싸면 Runnable처럼 사용할 수 있습니다.

```python
from langchain_core.runnables import RunnableLambda

def add_label(text: str) -> str:
    return f"주제:{text}"

def to_upper(text: str) -> str:
    return text.upper()

label_step = RunnableLambda(add_label)
upper_step = RunnableLambda(to_upper)

text_chain = label_step | upper_step

result = text_chain.invoke("langchain")

print(result)
```

결과:

```text
주제: LANGCHAIN
```

이 예제는 모델 API 없이도 `|`가 앞 단계의 출력을 다음 단계의 입력으로 전달한다는 것을 보여줍니다.

---

# 핵심 정리

```python
chain = prompt | model | parser
```

이 코드는 다음 실행 순서를 연결합니다.

```text
입력 dict
   ↓
Prompt
   ↓
Model
   ↓
Parser
   ↓
최종 결과
```

- LCEL은 LangChain 구성 요소의 실행 흐름을 표현하는 방법입니다.
- `|`는 앞 단계의 출력을 다음 단계의 입력으로 연결합니다.
- Prompt, Model, Parser는 Runnable 방식으로 실행할 수 있습니다.
- 연결된 전체 체인도 하나의 Runnable입니다.
- `chain = prompt | model | parser`는 연결만 합니다.
- 실제 실행은 `chain.invoke(input_data)`에서 시작됩니다.
- 이번 체인의 시작 입력은 `dict`, 최종 결과는 `str`입니다.