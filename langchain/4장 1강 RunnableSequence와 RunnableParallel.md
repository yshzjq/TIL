---
title: 4장 1강 RunnableSequence와 RunnableParallel
date: 2026-09-30
updated: 2026-09-30
description: KANT 강의 '4장 1강 RunnableSequence와 RunnableParallel' 정리
---

# 4장 1강: RunnableSequence와 RunnableParallel

## 1. 학습 목표

이번 강의에서는 다음 내용을 배웁니다.

- 여러 Runnable을 순서대로 실행하는 `RunnableSequence`
- 같은 입력을 여러 작업에 전달하는 `RunnableParallel`
- 전처리를 먼저 한 뒤 여러 branch로 나누는 구조
- 병렬 실행 결과를 딕셔너리로 확인하는 방법

이번 실습에서는 하나의 글을 먼저 정리한 뒤:

```text
요약
키워드 추출
```

두 작업에 같은 입력을 전달합니다.

---

# 2. 순차 실행과 병렬 실행

## 순차 실행

앞 단계의 결과가 다음 단계의 입력으로 사용되는 구조입니다.

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

즉:

```text
두 번째 단계가 첫 번째 단계의 결과를 사용함
→ 순차 실행
```

---

## 병렬 실행

같은 입력을 여러 작업이 각각 사용합니다.

```text
              ┌→ 요약
같은 입력 ────┤
              └→ 키워드 추출
```

두 작업은 같은 원문을 사용하지만 서로의 결과를 입력으로 사용하지 않습니다.

### 구분 방법

| 상황 | 실행 방식 |
|---|---|
| 앞 단계 결과를 다음 단계가 사용 | 순차 실행 |
| 여러 작업이 같은 입력을 각각 사용 | 병렬 실행 |
| 입력을 정리한 뒤 여러 작업으로 나눔 | 순차 → 병렬 |

이번 강의에서는 마지막 구조를 사용합니다.

```text
원문 문자열
 ↓
전처리
 ↓
{"text": "..."}
 ↓
 ├─ 요약
 └─ 키워드 추출
 ↓
결과 dict
```

---

# 3. `RunnableSequence`

`RunnableSequence`는 여러 Runnable을 **정해진 순서대로 실행**합니다.

## 간단한 예시

```python
from langchain_core.runnables import RunnableLambda, RunnableSequence


def clean_text(text: str) -> str:
    return text.strip()


def add_label(text: str) -> str:
    return f"학습 주제: {text}"


clean_step = RunnableLambda(clean_text)
label_step = RunnableLambda(add_label)


sequence = RunnableSequence(
    clean_step,
    label_step,
)


result = sequence.invoke("  LangChain  ")

print(result)
```

결과:

```text
학습 주제: LangChain
```

실행 순서는:

```text
"  LangChain  "
 ↓
clean_step
 ↓
"LangChain"
 ↓
label_step
 ↓
"학습 주제: LangChain"
```

입니다.

`RunnableSequence`에서는 **단계의 순서가 중요합니다.**

뒤 단계는 바로 앞 단계의 출력을 입력으로 받습니다.

---

# 4. `|`와 `RunnableSequence`

다음 두 코드는 같은 순서의 체인을 만듭니다.

```python
sequence_by_class = RunnableSequence(
    clean_step,
    label_step,
)
```

```python
sequence_by_pipe = clean_step | label_step
```

둘 다 결과적으로:

```text
RunnableSequence
```

입니다.

즉:

```python
first | second
```

도 순차 실행 체인입니다.

---

# 5. `RunnableParallel`

`RunnableParallel`은 **같은 입력을 여러 branch에 전달**합니다.

그리고 각 branch의 결과를 하나의 딕셔너리로 모읍니다.

## 간단한 예시

```python
from langchain_core.runnables import RunnableLambda, RunnableParallel


def make_upper(text: str) -> str:
    return text.upper()


def count_length(text: str) -> int:
    return len(text)


parallel = RunnableParallel(
    upper=RunnableLambda(make_upper),
    length=RunnableLambda(count_length),
)


result = parallel.invoke("LangChain")


print(result["upper"])
print(result["length"])
print(type(result).__name__)
```

결과:

```text
LANGCHAIN
9
dict
```

같은 `"LangChain"`이 두 branch에 전달됩니다.

```text
                ┌→ make_upper()
"LangChain" ────┤
                └→ count_length()
```

결과는:

```python
{
    "upper": "LANGCHAIN",
    "length": 9
}
```

형태가 됩니다.

---

# 6. branch 이름과 결과 key

다음 코드에서:

```python
parallel = RunnableParallel(
    upper=...,
    length=...,
)
```

`upper`, `length`가 최종 결과 딕셔너리의 key가 됩니다.

따라서:

```python
result["upper"]
result["length"]
```

로 값을 가져옵니다.

이 이름들은 LangChain이 정한 이름이 아니라 **코드를 작성한 사람이 정한 이름**입니다.

---

# 7. 요약 branch 만들기

이번 실습에서는 첫 번째 branch로 글을 요약합니다.

두 branch 모두 `{text}`를 사용합니다.

```python
input_data = {
    "text": "LCEL은 여러 단계를 연결하는 표현 방식입니다."
}
```

요약 Prompt:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "주어진 글을 입문자가 이해하기 쉬운 한 문장으로 요약하세요.",
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
```

요약 체인:

```python
summary_chain = (
    summary_prompt
    | model
    | StrOutputParser()
)
```

이 branch는 최종적으로 요약 문자열을 반환합니다.

---

# 8. 키워드 branch 만들기

두 번째 branch는 핵심 키워드 세 개를 추출합니다.

```python
keyword_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "주어진 글에서 핵심 키워드 세 개를 쉼표로 구분해 답하세요.",
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
```

체인:

```python
keyword_chain = (
    keyword_prompt
    | model
    | StrOutputParser()
)
```

---

# 9. 두 branch 병렬로 연결하기

요약과 키워드 체인을 `RunnableParallel`로 묶습니다.

```python
parallel_analysis = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)
```

구조:

```text
              ┌→ summary_chain
같은 입력 ────┤
              └→ keyword_chain
```

아직 `invoke()`를 하지 않았기 때문에 이 시점에서는 구조만 만들어진 상태입니다.

---

# 10. 입력 전처리하기

완성 체인에서는 원문 문자열을 바로 Prompt에 전달하지 않습니다.

먼저 입력을 확인하고 정리합니다.

```python
def prepare_input(text: str) -> dict[str, str]:

    if not isinstance(text, str):
        raise TypeError("text는 문자열이어야 합니다.")

    clean_text = text.strip()

    if not clean_text:
        raise ValueError("text는 비어 있을 수 없습니다.")

    return {"text": clean_text}
```

예를 들어:

```python
prepared = prepare_input(
    "  LCEL은 단계를 연결합니다.  "
)

print(prepared)
```

결과:

```python
{
    "text": "LCEL은 단계를 연결합니다."
}
```

즉:

```text
"  LCEL은 단계를 연결합니다.  "
 ↓
공백 제거
 ↓
"LCEL은 단계를 연결합니다."
 ↓
dict로 변환
 ↓
{"text": "LCEL은 단계를 연결합니다."}
```

입니다.

---

# 11. 전처리도 Runnable로 만들기

일반 Python 함수인 `prepare_input()`을 `RunnableLambda`로 감쌉니다.

```python
prepare_step = RunnableLambda(prepare_input)
```

그러면 다른 Runnable과 연결할 수 있습니다.

---

# 12. 순차 실행 + 병렬 실행 연결

전처리를 먼저 실행하고 그 결과를 병렬 분석에 전달합니다.

```python
analysis_chain = RunnableSequence(
    prepare_step,
    parallel_analysis,
)
```

전체 구조:

```text
원문 str
   ↓
prepare_step
   ↓
{"text": "..."}
   ↓
 ┌─────────────────────┐
 ↓                     ↓
summary_chain      keyword_chain
 ↓                     ↓
요약 str             키워드 str
 └─────────┬───────────┘
           ↓
      결과 dict
```

자료형 흐름은 다음과 같습니다.

| 단계 | 입력 | 출력 |
|---|---|---|
| `prepare_step` | `str` | `{"text": str}` |
| `summary_chain` | `dict` | 요약 `str` |
| `keyword_chain` | `dict` | 키워드 `str` |
| `RunnableParallel` | 두 결과 | `dict` |

전처리는 한 번만 실행되고 그 결과가 두 branch에 전달됩니다.

---

# 13. 병렬 결과 확인하기

전체 체인을 실행합니다.

```python
result = analysis_chain.invoke(
    "LCEL은 Prompt, Model, Parser를 연결합니다."
)
```

결과 형태는:

```python
{
    "summary": "요약 결과",
    "keywords": "키워드 결과"
}
```

입니다.

각 결과는 다음처럼 가져옵니다.

```python
summary_text = result["summary"]

keyword_text = result["keywords"]
```

출력:

```python
print(f"[요약]\n{summary_text}")
print(f"[키워드]\n{keyword_text}")
```

---

# 14. branch 이름과 결과 key 관계

다음처럼 만들었다면:

```python
parallel_analysis = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)
```

결과는:

```python
result["summary"]
result["keywords"]
```

로 읽습니다.

| branch 이름 | 결과 접근 |
|---|---|
| `summary` | `result["summary"]` |
| `keywords` | `result["keywords"]` |

branch 이름과 결과를 읽을 때 사용하는 key가 같아야 합니다.

---

# 15. API 호출 횟수

이번 구조에는 Model을 사용하는 branch가 두 개 있습니다.

```text
summary branch
→ API 1회

keywords branch
→ API 1회
```

따라서:

```text
총 API 호출
→ 2회
```

입니다.

같은 `model` 객체를 사용해도 각각 별도의 요청입니다.

```python
parallel_analysis.invoke(...)
```

를 한 번 실행했다고 해서 API 요청도 한 번만 발생하는 것은 아닙니다.

---

# 16. 전체 실행 흐름

완성 실습은 다음 순서로 실행됩니다.

```text
.env 읽기
 ↓
ChatOpenAI 생성
 ↓
원문 문자열 입력
 ↓
전처리
 ↓
{"text": "..."}
 ↓
 ├─ 요약 branch
 └─ 키워드 branch
 ↓
결과 dict
 ↓
summary / keywords 출력
```

실행:

```powershell
uv run python 실습.py
```

출력 구조:

```text
[전체 체인 타입]
RunnableSequence

[요약]
모델이 만든 한 문장 요약

[키워드]
키워드1, 키워드2, 키워드3
```

---

# 17. 자주 발생하는 오류

## 1) API Key 없음

`.env` 파일을 확인합니다.

```text
OPENAI_API_KEY=발급받은_API_Key
OPENAI_MODEL=gpt-5.6-luna
```

실제 Model을 호출하려면 `OPENAI_API_KEY`가 필요합니다.

---

## 2) 문자열이 아닌 값을 전달

잘못된 입력:

```python
wrong_input = 100
```

`prepare_input()`은 문자열을 기대하므로 오류가 발생합니다.

올바르게:

```python
analysis_chain.invoke(
    "LCEL에 대한 설명"
)
```

처럼 문자열을 전달합니다.

---

## 3) 빈 문자열 전달

```python
empty_text = "   "
```

공백을 제거하면 내용이 없기 때문에 전처리 단계에서 중단됩니다.

실제 내용이 있는 문자열을 사용합니다.

---

## 4) Prompt 변수와 dict key가 다름

Prompt가:

```python
"{text}"
```

를 요구하는데:

```python
{
    "content": "LCEL 설명"
}
```

처럼 반환하면 `text`를 찾을 수 없습니다.

다음처럼 일치해야 합니다.

```python
{
    "text": "LCEL 설명"
}
```

---

## 5) 결과 key 이름이 다름

branch 이름이:

```python
keywords=keyword_chain
```

이면:

```python
result["keywords"]
```

로 읽어야 합니다.

다음은 잘못된 예입니다.

```python
result["keyword"]
```

---

## 6) API 호출을 한 번이라고 생각함

요약과 키워드 branch에 Model이 각각 있으므로:

```text
요약 → 1회
키워드 → 1회
총 2회
```

호출됩니다.

---

# 18. 핵심 비교

## `RunnableSequence`

```text
앞 단계 결과
 ↓
다음 단계 입력
```

즉:

```text
A → B → C
```

처럼 순서대로 실행합니다.

---

## `RunnableParallel`

```text
        ┌→ A
입력 ───┤
        └→ B
```

같은 입력을 여러 branch에 전달하고 결과를 딕셔너리로 모읍니다.

---

# 핵심 정리

### 순차 실행

```python
RunnableSequence(
    first,
    second,
)
```

또는:

```python
first | second
```

```text
앞 단계의 출력
→ 다음 단계의 입력
```

---

### 병렬 실행

```python
RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)
```

```text
같은 입력
 ├─ summary
 └─ keywords

→ dict 결과
```

---

### 이번 강의의 전체 구조

```text
원문 str
 ↓
전처리
 ↓
{"text": "..."}
 ↓
 ├─ 요약
 └─ 키워드
 ↓
{
    "summary": "...",
    "keywords": "..."
}
```

가장 간단하게 기억하면:

```text
RunnableSequence
= 앞에서 뒤로 순서대로

RunnableParallel
= 같은 입력을 여러 갈래로

Sequence + Parallel
= 먼저 정리하고 여러 작업으로 나누기
```