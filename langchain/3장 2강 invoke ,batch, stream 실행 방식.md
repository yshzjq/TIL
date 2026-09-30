---
title: 3장 2강 invoke/batch/stream 실행 방식
date: 2026-09-30
updated: 2026-09-30
description: KANT 강의 '3장 2강 invoke/batch/stream 실행 방식' 정리
---

# 3-2강: `invoke()`, `batch()`, `stream()` 실행 방식

## 1. 학습 목표

같은 LCEL 체인을 상황에 따라 여러 방식으로 실행할 수 있습니다.

이번 강의에서는 다음 세 가지를 구분합니다.

```text
입력 한 건, 결과 한 개
→ invoke()

입력 여러 건, 결과 목록
→ batch()

입력 한 건, 결과를 조각씩 받음
→ stream()
```

핵심은 **체인의 구조는 그대로 두고 실행 방식만 바꿀 수 있다**는 것입니다.

---

# 2. 공통으로 사용하는 체인

세 방식 모두 같은 체인을 사용할 수 있습니다.

```python
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 LangChain 입문 수업의 튜터입니다. "
            "질문에 한 문장으로 답하세요.",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
)


parser = StrOutputParser()


chain = prompt | model | parser
```

실행 흐름은 세 방식 모두 같습니다.

```text
입력
 ↓
Prompt
 ↓
Model
 ↓
Parser
 ↓
결과
```

달라지는 것은 **입력 개수와 결과를 받는 방식**입니다.

---

# 3. `invoke()`

`invoke()`는 **입력 한 건을 실행하고 최종 결과 한 개를 받을 때** 사용합니다.

## 기본 사용법

```python
input_data = {
    "question": "invoke는 언제 사용하나요?"
}

raw_result = chain.invoke(input_data)

result = str(raw_result)

print(result)
print(type(result).__name__)
```

흐름은 다음과 같습니다.

```text
dict 한 개
 ↓
Prompt
 ↓
Model
 ↓
Parser
 ↓
str 한 개
```

예를 들어 사용자가 질문 하나를 입력하고 답변 하나를 받는 경우 사용할 수 있습니다.

---

# 4. `batch()`

`batch()`는 **여러 입력을 한 번에 전달하고 여러 결과를 받을 때** 사용합니다.

## 입력 준비

여러 개의 딕셔너리를 리스트로 만듭니다.

```python
inputs = [
    {
        "question": "invoke는 언제 사용하나요?"
    },
    {
        "question": "batch는 어떤 값을 반환하나요?"
    },
]
```

각 입력에는 Prompt가 요구하는 `question` 키가 있어야 합니다.

## 실행

```python
raw_results = chain.batch(inputs)

results = [str(result) for result in raw_results]

print(type(results).__name__)

for number, result in enumerate(results, start=1):
    print(f"{number}.{result}")
```

결과 구조는 다음과 같습니다.

```text
입력
[
    dict,
    dict
]

      ↓

batch()

      ↓

결과
[
    str,
    str
]
```

즉:

```text
입력 → 리스트
결과 → 리스트
```

입니다.

결과 목록의 위치는 전달한 입력의 위치와 대응합니다.

`Runnable.batch()`는 LangChain 체인에 여러 입력을 전달하는 기능이며, 별도의 공급자 Batch API와는 다른 기능입니다.

---

# 5. `stream()`

`stream()`은 **입력 한 건을 전달하고 결과를 만들어지는 순서대로 나누어 받을 때** 사용합니다.

## 기본 사용법

```python
input_data = {
    "question": "stream의 장점을 한 문장으로 설명해 주세요."
}

print("[스트리밍 답변]")

for raw_chunk in chain.stream(input_data):
    chunk = str(raw_chunk)

    print(chunk, end="", flush=True)

print()
```

`stream()`은 최종 결과 한 개를 바로 주는 것이 아니라 여러 **조각(chunk)**을 순서대로 받을 수 있습니다.

```text
입력 한 건
   ↓
stream()
   ↓
조각 1
조각 2
조각 3
...
```

따라서 반복문을 사용합니다.

```python
for raw_chunk in chain.stream(input_data):
    ...
```

---

# 6. stream에서 반복문이 필요한 이유

`invoke()`는 최종 결과 하나를 반환합니다.

```python
result = chain.invoke(input_data)
```

반면 `stream()`은 여러 결과 조각을 차례대로 받을 수 있는 값을 반환합니다.

```python
chunks = chain.stream(input_data)

for raw_chunk in chunks:
    chunk = str(raw_chunk)

    print(repr(chunk), type(chunk).__name__)
```

조각이 나뉘는 위치와 개수는 실행할 때마다 달라질 수 있습니다.

따라서 특정 글자나 단어마다 항상 똑같이 나뉜다고 생각하면 안 됩니다.

---

# 7. stream 결과를 하나로 합치기

조각을 화면에 바로 보여주면서 전체 답변도 사용하고 싶다면 리스트에 저장한 뒤 합칠 수 있습니다.

```python
collected_chunks = []

for raw_chunk in chain.stream(input_data):
    chunk = str(raw_chunk)

    collected_chunks.append(chunk)

    print(chunk, end="", flush=True)


full_text = "".join(collected_chunks)

print(f"\n[전체 자료형]{type(full_text).__name__}")
```

흐름은 다음과 같습니다.

```text
chunk 1 ┐
chunk 2 ├─ 리스트에 저장
chunk 3 ┘
    ↓
"".join()
    ↓
전체 문자열
```

---

# 8. `invoke`, `batch`, `stream` 비교

| 구분 | `invoke()` | `batch()` | `stream()` |
|---|---|---|---|
| 입력 | 한 건 | 여러 건 | 한 건 |
| 결과 | 결과 한 개 | 결과 리스트 | 여러 조각 |
| 결과 처리 | 변수에 저장 | 반복문으로 결과 확인 | 반복문으로 조각 확인 |
| 사용 예 | 질문 하나에 답하기 | 여러 입력 각각 처리하기 | 생성 중인 답변을 바로 보여주기 |

## 공통점

세 방식 모두:

- 같은 `chain`을 사용할 수 있습니다.
- Prompt의 입력 키는 같습니다.
- `Prompt → Model → Parser` 순서는 같습니다.
- 실행할 때 모델 API 요청이 발생합니다.

## 차이점

다른 것은 다음 세 가지입니다.

```text
입력 개수
결과 형태
결과를 보여주는 시점
```

---

# 9. 어떤 방식을 선택할까?

## 입력이 한 건일 때

완성된 결과만 필요하다면:

```python
invoke()
```

생성되는 내용을 바로 보여주고 싶다면:

```python
stream()
```

## 입력이 여러 건일 때

여러 독립 입력을 처리하려면:

```python
batch()
```

예시는 다음과 같습니다.

| 상황 | 사용 |
|---|---|
| 사용자 질문 하나에 답변 | `invoke()` |
| 여러 상품 설명을 각각 요약 | `batch()` |
| 긴 답변을 생성하면서 화면에 표시 | `stream()` |

처음 구현할 때는 `invoke()`로 정상 동작을 확인한 뒤 필요하면 `batch()`나 `stream()`으로 바꾸는 방식이 이해하기 쉽습니다.

---

# 10. 실행 방식 선택하기

실습 코드에서는 다음 값으로 실행 방식을 선택합니다.

```python
EXECUTION_MODE = "invoke"
```

사용할 수 있는 값은:

```text
"invoke"
"batch"
"stream"
```

입니다.

실행은 다음과 같이 합니다.

```powershell
uv run python 실습.py
```

---

# 11. 자주 발생하는 오류

## 1) `batch()`에 딕셔너리 하나만 전달

잘못된 형태:

```python
single_input = {
    "question": "batch는 무엇인가요?"
}

chain.batch(single_input)
```

`batch()`는 여러 입력이 담긴 리스트를 사용합니다.

```python
batch_inputs = [
    {
        "question": "batch는 무엇인가요?"
    }
]

results = chain.batch(batch_inputs)
```

---

## 2) batch 결과를 문자열처럼 사용

`batch()`의 결과는 리스트입니다.

따라서 다음처럼 사용할 수 없습니다.

```python
results.upper()
```

각 결과를 반복문으로 꺼내야 합니다.

```python
for result in results:
    print(result)
```

---

## 3) stream 결과 자체를 바로 출력

다음처럼 사용하면:

```python
print(chain.stream(input_data))
```

답변 내용이 아니라 객체 표시가 나올 수 있습니다.

반복문으로 조각을 꺼내야 합니다.

```python
for chunk in chain.stream(input_data):
    print(chunk)
```

---

## 4) stream에서 계속 줄바꿈됨

다음 코드는 조각마다 줄이 바뀝니다.

```python
print(chunk)
```

조각을 이어서 출력하려면:

```python
print(chunk, end="", flush=True)
```

를 사용합니다.

### `end=""`

출력 후 줄바꿈을 하지 않습니다.

### `flush=True`

출력을 바로 화면에 표시합니다.

---

## 5) stream 조각 개수를 고정된 값으로 생각함

스트리밍 조각의 개수와 나뉘는 위치는 실행에 따라 달라질 수 있습니다.

따라서 조각 개수보다는 모든 조각을 합친 전체 문자열을 확인합니다.

---

## 6) 실행 방식 이름 오류

다음은 서로 다른 문자열입니다.

```text
"Invoke"
"invoke"
```

실습에서는 정확히 다음 중 하나를 사용합니다.

```text
invoke
batch
stream
```

---

# 12. `RunnableLambda`로 간단히 확인하기

모델 API 없이도 세 실행 방식을 연습할 수 있습니다.

## `invoke()`

```python
from langchain_core.runnables import RunnableLambda


def add_ten(number: int) -> int:
    return number + 10


add_chain = RunnableLambda(add_ten)

result = add_chain.invoke(5)

print(result)
print(type(result).__name__)
```

결과:

```text
15
int
```

---

## `batch()`

```python
from langchain_core.runnables import RunnableLambda


def add_ten(number: int) -> int:
    return number + 10


add_chain = RunnableLambda(add_ten)

inputs = [1, 2, 3]

results = add_chain.batch(inputs)

print(results)
print(len(results))
```

결과:

```text
[11, 12, 13]
3
```

즉:

```text
invoke(5)
→ 15

batch([1, 2, 3])
→ [11, 12, 13]
```

---

# 핵심 정리

### `invoke()`

```python
chain.invoke(input_data)
```

```text
입력 한 건
→ 결과 한 개
```

---

### `batch()`

```python
chain.batch(inputs)
```

```text
입력 여러 건
→ 결과 여러 건의 리스트
```

---

### `stream()`

```python
for chunk in chain.stream(input_data):
    print(chunk, end="", flush=True)
```

```text
입력 한 건
→ 결과를 여러 조각으로 받음
```

가장 간단하게 기억하면:

```text
invoke
= 하나 넣고 하나 받기

batch
= 여러 개 넣고 여러 개 받기

stream
= 하나 넣고 결과를 조금씩 받기
```