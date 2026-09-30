---
title: 3장 3강 중간 출력 디버깅과 chain 재사용
date: 2026-09-30
updated: 2026-09-30
description: KANT 강의 '3장 3강 중간 출력 디버깅과 chain 재사용' 정리
---

# 3장 3강: 중간 출력 디버깅과 chain 재사용

## 1. 학습 목표

이번 강의에서는 다음 내용을 배웁니다.

- 체인 실행이 실패했을 때 입력과 각 단계의 자료형을 순서대로 확인합니다.
- `str`을 잘못 전달한 오류를 올바른 `dict` 입력으로 수정합니다.
- Prompt, Model, Parser를 함수로 묶어 체인을 재사용합니다.

이번 강의에서는 특히 **`dict`가 필요한 자리에 `str`을 전달한 오류**를 중심으로 확인합니다.

---

# 2. 체인 오류 확인 순서

LCEL 체인은 다음과 같이 실행됩니다.

```text
입력 dict
   ↓
Prompt
   ↓
Model
   ↓
Parser
   ↓
최종 str
```

코드에서는 한 줄로 연결되어 있어도 실제 실행은 왼쪽부터 순서대로 진행됩니다.

```python
chain = prompt | model | parser
```

오류가 발생하면 다음 순서로 확인합니다.

```text
1. 처음 입력의 자료형과 키 확인
2. Prompt 결과 확인
3. Model 결과 확인
4. Parser 결과 확인
```

특히 가장 먼저 확인할 것은:

```text
입력값의 자료형
입력값의 key
```

입니다.

---

# 3. 두 개의 입력 변수를 가진 Prompt

이번 실습에서는 `topic`, `example` 두 개의 입력 변수를 사용합니다.

```python
from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 LangChain 입문 수업의 튜터입니다.",
        ),
        (
            "human",
            "주제:{topic}\n"
            "간단한 예:{example}\n"
            "주제를 두 문장으로 설명해 주세요.",
        ),
    ]
)
```

이 Prompt가 요구하는 변수는:

```text
topic
example
```

입니다.

확인하려면:

```python
print(prompt.input_variables)
```

결과:

```text
['example', 'topic']
```

순서는 정렬되어 표시될 수 있지만, 중요한 점은 **두 키가 모두 필요하다**는 것입니다.

---

# 4. 올바른 입력 형태

Prompt가 `topic`, `example` 두 값을 요구하므로 딕셔너리로 전달합니다.

```python
input_data = {
    "topic": "LCEL",
    "example": "prompt | model | parser",
}
```

자료형과 key를 확인할 수 있습니다.

```python
print(type(input_data).__name__)
print(list(input_data.keys()))
```

결과:

```text
dict
['topic', 'example']
```

딕셔너리를 사용하면 여러 값을 key 이름으로 구분할 수 있습니다.

---

# 5. 잘못된 `str` 입력

다음처럼 문자열 하나만 전달하면 문제가 발생합니다.

```python
wrong_input = "LCEL"

print(type(wrong_input).__name__)

prompt.invoke(wrong_input)
```

자료형은:

```text
str
```

입니다.

하지만 Prompt에서는:

```text
topic
example
```

두 값을 구분해서 찾아야 합니다.

문자열 `"LCEL"`에는 `topic`, `example`이라는 key가 없기 때문에 값을 구분할 수 없습니다.

즉:

```text
잘못된 입력

"LCEL"
   ↓
str
   ↓
topic과 example을 구분할 수 없음
```

반면 올바른 형태는:

```python
{
    "topic": "LCEL",
    "example": "prompt | model | parser",
}
```

입니다.

---

# 6. 입력 자료형 먼저 확인하기

오류가 발생했을 때 먼저 다음처럼 자료형을 확인합니다.

```python
print(f"입력 자료형:{type(wrong_input).__name__}")
```

결과:

```text
입력 자료형:str
```

긴 오류 메시지를 모두 읽기 전에 실제 입력이:

```text
str인지
dict인지
```

먼저 확인하면 원인을 찾기 쉽습니다.

---

# 7. 올바른 `dict`로 수정하기

잘못된 문자열 입력을 다음과 같이 수정합니다.

```python
correct_input = {
    "topic": "LCEL",
    "example": "prompt | model | parser",
}
```

Prompt에 전달합니다.

```python
print(type(correct_input).__name__)

prompt_value = prompt.invoke(correct_input)

print(type(prompt_value).__name__)
```

일반적인 결과:

```text
dict
ChatPromptValue
```

즉:

```text
dict
 ↓
Prompt
 ↓
ChatPromptValue
```

가 됩니다.

---

# 8. Prompt의 중간 결과 확인하기

Prompt 결과 안에 만들어진 메시지를 직접 확인할 수도 있습니다.

```python
messages = prompt_value.to_messages()

for message in messages:
    print(message.type, message.content)
```

결과 구조:

```text
system 당신은 LangChain 입문 수업의 튜터입니다.

human 주제: LCEL
간단한 예: prompt | model | parser
주제를 두 문장으로 설명해 주세요.
```

이 단계에서는 아직 Model을 실행하지 않았기 때문에 API가 호출되지 않습니다.

Prompt에 값이 제대로 들어갔는지 확인할 때 사용할 수 있습니다.

---

# 9. 중간 결과를 단계별로 확인하기

완성된 체인에서 오류가 발생하면 잠시 각 단계를 나누어 실행합니다.

## 구성 요소 준비

```python
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


load_dotenv()


model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
)

parser = StrOutputParser()
```

## 각 단계 실행

```python
prompt_value = prompt.invoke(correct_input)

ai_message = model.invoke(prompt_value)

raw_text = parser.invoke(ai_message)

final_text = str(raw_text)
```

전체 흐름:

```text
correct_input
     ↓
Prompt
     ↓
prompt_value
     ↓
Model
     ↓
ai_message
     ↓
Parser
     ↓
final_text
```

---

# 10. 중간 자료형 확인

각 단계의 내용을 전부 출력하기 전에 자료형부터 확인합니다.

```python
print(f"입력:{type(correct_input).__name__}")

print(f"Prompt 결과:{type(prompt_value).__name__}")

print(f"Model 결과:{type(ai_message).__name__}")

print(f"Parser 결과:{type(final_text).__name__}")
```

일반적인 결과:

```text
입력: dict
Prompt 결과: ChatPromptValue
Model 결과: AIMessage
Parser 결과: str
```

즉 자료형 흐름은:

```text
dict
 ↓
ChatPromptValue
 ↓
AIMessage
 ↓
str
```

입니다.

오류를 찾을 때 긴 중간값 전체를 출력하기보다 먼저 자료형 흐름이 맞는지 확인합니다.

---

# 11. 체인 생성 함수 만들기

같은 구조를 여러 번 사용할 경우 Prompt, Model, Parser를 만드는 코드를 함수로 묶을 수 있습니다.

```python
def build_explain_chain(model, audience: str):

    system_message = (
        f"당신은{audience} 대상 LangChain 튜터입니다. "
        "전문 용어를 먼저 풀어 쓰세요."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                "주제:{topic}\n"
                "간단한 예:{example}\n"
                "주제를 두 문장으로 설명해 주세요.",
            ),
        ]
    )

    parser = StrOutputParser()

    return prompt | model | parser
```

이 함수는:

```text
Prompt 생성
+
Model 연결
+
Parser 생성
+
체인 반환
```

역할을 합니다.

중요한 부분은:

```python
return prompt | model | parser
```

입니다.

---

# 12. 체인 생성과 실행은 다름

다음 코드는 체인을 만듭니다.

```python
beginner_chain = build_explain_chain(
    model=model,
    audience="파이썬 입문자",
)
```

이 시점에서는 API가 호출되지 않습니다.

자료형을 확인하면:

```python
print(type(beginner_chain).__name__)
```

일반적으로:

```text
RunnableSequence
```

가 나옵니다.

실제 API 호출은 다음과 같이 `invoke()`할 때 발생합니다.

```python
beginner_chain.invoke(input_data)
```

즉:

```text
build_explain_chain()
→ 체인 생성

invoke()
→ 체인 실행
```

입니다.

---

# 13. 만든 체인 재사용하기

체인은 한 번 만든 뒤 입력값만 바꿔 다시 사용할 수 있습니다.

## 첫 번째 입력

```python
first_input = {
    "topic": "LCEL",
    "example": "prompt | model | parser",
}

raw_first_result = beginner_chain.invoke(first_input)

first_result = str(raw_first_result)

print(first_result)
```

## 두 번째 입력

```python
second_input = {
    "topic": "Runnable",
    "example": "chain.invoke(input_data)",
}

raw_second_result = beginner_chain.invoke(second_input)

second_result = str(raw_second_result)

print(second_result)
```

같은 체인을 다시 만들지 않고 입력만 변경합니다.

---

# 14. 재사용되는 값과 바뀌는 값

## 재사용되는 부분

- system 메시지의 튜터 역할
- human 메시지의 문장 구조
- 같은 ChatModel
- 같은 `StrOutputParser`
- Prompt → Model → Parser 순서

## 바뀌는 부분

```text
topic
example
```

즉:

```text
같은 체인
   ↓
입력만 변경
   ↓
다른 질문에 사용
```

할 수 있습니다.

---

# 15. 전체 실습 흐름

실습에서는 다음 흐름으로 진행됩니다.

```text
체인 생성
   ↓
잘못된 str 입력 확인
   ↓
입력 오류 확인
   ↓
올바른 dict로 수정
   ↓
같은 체인을 invoke
   ↓
최종 str 결과
```

실행:

```powershell
uv run python 실습.py
```

결과 구조:

```text
[체인 종류] RunnableSequence

[잘못된 입력 자료형] str

[입력 오류 확인]
체인 입력은 topic과 example 키를 가진 dict여야 합니다.

[수정된 입력 자료형] dict

[최종 답변]
모델 답변

[최종 자료형] str
```

잘못된 문자열 입력은 Model에 전달되기 전에 중단되므로 그 단계에서는 API가 호출되지 않습니다.

올바른 딕셔너리를 `invoke()`할 때 API가 호출됩니다.

---

# 16. 자주 발생하는 오류

## 1) 문자열을 `dict` 자리에 전달

잘못된 예:

```python
wrong_input = "LCEL"
```

올바른 형태:

```python
{
    "topic": "LCEL",
    "example": "prompt | model | parser",
}
```

---

## 2) 입력 key 오타

Prompt:

```python
"{topic}"
```

인데 입력을:

```python
{
    "topics": "LCEL"
}
```

처럼 작성하면 변수를 찾을 수 없습니다.

확인하려면:

```python
print(prompt.input_variables)
```

와 딕셔너리 key를 비교합니다.

---

## 3) 체인 생성 함수에서 `return`을 빼먹음

다음 부분이 없으면:

```python
return prompt | model | parser
```

함수의 결과가 `None`이 됩니다.

자료형을 확인할 수 있습니다.

```python
print(type(beginner_chain).__name__)
```

---

## 4) 생성 함수 안에서 `invoke()`까지 실행

체인 생성 함수는 체인을 만들어 반환하는 역할에 집중합니다.

함수 안에서 질문까지 고정하고 실행하면 입력을 바꾸어 재사용하기 어렵습니다.

즉:

```text
체인 만드는 함수
→ 체인 반환

실제 질문 실행
→ 함수 밖에서 invoke()
```

로 나눕니다.

---

## 5) 같은 입력이면 답변도 완전히 같다고 생각함

같은 체인과 같은 입력을 사용해도 모델이 생성한 문장은 달라질 수 있습니다.

따라서 답변 문장을 글자 단위로 비교하기보다:

```text
최종 자료형
Prompt에서 요구한 형식
```

을 중심으로 확인합니다.

---

# 핵심 정리

## 오류가 발생하면

```text
입력
 ↓
Prompt
 ↓
Model
 ↓
Parser
```

순서대로 확인합니다.

가장 먼저:

```python
type(input_data)
```

로 입력 자료형을 확인합니다.

---

## 여러 Prompt 변수가 있다면

예를 들어:

```text
topic
example
```

두 변수가 있다면:

```python
{
    "topic": "...",
    "example": "...",
}
```

처럼 `dict`로 값을 구분합니다.

---

## 중간 자료형 흐름

```text
dict
 ↓
ChatPromptValue
 ↓
AIMessage
 ↓
str
```

---

## 체인 재사용

체인을 만드는 함수를 작성할 수 있습니다.

```python
def build_explain_chain(model, audience):
    ...
    return prompt | model | parser
```

그리고 한 번 만든 체인에 입력만 바꿔 재사용합니다.

```python
chain.invoke(first_input)

chain.invoke(second_input)
```

가장 간단하게 기억하면:

```text
디버깅
= 입력부터 중간 자료형을 순서대로 확인

체인 재사용
= 구조는 그대로 두고 입력만 변경
```