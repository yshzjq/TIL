---
title: 4장 2강 RunnablePassthrough/RunnableLambda/assign
date: 2026-09-30
updated: 2026-09-30
description: KANT 강의 '4장 2강 RunnablePassthrough/RunnableLambda/assign' 정리
---

# 4장 2강: RunnablePassthrough, RunnableLambda, assign

## 1. 학습 목표

이번 강의에서는 다음 세 가지를 배웁니다.

- `RunnablePassthrough`로 기존 입력을 유지하기
- `RunnableLambda`로 일반 Python 함수를 Runnable로 만들기
- `RunnablePassthrough.assign()`으로 기존 딕셔너리에 새로운 필드 추가하기

이번 실습에서는 다음 입력을 사용합니다.

```python
input_data = {
    "question": "  RunnablePassthrough는 어떤 역할을 하나요?  ",
    "student_level": "입문",
}
```

최종적으로 다음과 같이 필드가 추가됩니다.

```python
{
    "question": "  RunnablePassthrough는 어떤 역할을 하나요?  ",
    "student_level": "입문",
    "clean_question": "RunnablePassthrough는 어떤 역할을 하나요?",
    "question_length": 32,
}
```

중요한 점은 **기존 값을 지우지 않고 새로운 값을 추가한다는 것**입니다.

---

# 2. 세 가지 도구의 역할

| 도구 | 역할 |
|---|---|
| `RunnablePassthrough()` | 받은 입력을 그대로 전달 |
| `RunnableLambda(function)` | 일반 Python 함수를 Runnable로 변환 |
| `RunnablePassthrough.assign()` | 기존 딕셔너리를 유지하면서 새 필드 추가 |

간단히 보면:

```text
RunnablePassthrough
= 그대로 통과

RunnableLambda
= Python 함수 실행

assign
= 기존 값 + 새 값
```

---

# 3. `RunnablePassthrough`

`RunnablePassthrough`는 입력을 바꾸지 않고 그대로 다음 단계로 전달합니다.

## 기본 사용법

```python
from langchain_core.runnables import RunnablePassthrough

input_data = {
    "question": "LCEL은 무엇인가요?",
    "student_level": "입문",
}

result = RunnablePassthrough().invoke(input_data)

print(result)
print(result == input_data)
```

결과:

```text
{'question': 'LCEL은 무엇인가요?', 'student_level': '입문'}
True
```

입력 내용을 바꾸지 않았기 때문에 원래 딕셔너리와 결과의 내용이 같습니다.

---

# 4. 왜 Passthrough를 사용할까?

체인을 만들다 보면 원본 값을 유지하면서 새로운 값도 추가해야 할 때가 있습니다.

예를 들어:

```text
question
→ 그대로 유지

student_level
→ 그대로 유지

clean_question
→ 새로 추가
```

이번 강의에서는 `assign()`을 사용할 때 기존 딕셔너리를 유지하기 위해 `RunnablePassthrough`를 사용합니다.

---

# 5. `RunnableLambda`

`RunnableLambda`는 일반 Python 함수를 Runnable로 감싸는 역할을 합니다.

Runnable이 되면:

```text
invoke()
|
```

를 사용할 수 있습니다.

## 일반 함수와 비교

```python
from langchain_core.runnables import RunnableLambda


def double_number(number: int) -> int:
    return number * 2


direct_result = double_number(5)

double_step = RunnableLambda(double_number)

runnable_result = double_step.invoke(5)

print(direct_result)
print(runnable_result)
```

결과:

```text
10
10
```

함수의 계산은 그대로이고, `RunnableLambda`를 사용하면 LangChain 체인의 한 단계로 사용할 수 있습니다.

---

# 6. 질문 정리 함수를 Runnable로 만들기

다음 함수는 딕셔너리에서 `question`을 꺼내 앞뒤 공백을 제거합니다.

```python
def extract_clean_question(data: dict) -> str:
    clean_question = data["question"].strip()

    return clean_question
```

이를 Runnable로 감쌉니다.

```python
clean_question_step = RunnableLambda(
    extract_clean_question
)
```

실행:

```python
clean_question = clean_question_step.invoke(
    {
        "question": "  assign은 무엇을 하나요?  ",
        "student_level": "입문",
    }
)

print(clean_question)
```

결과:

```text
assign은 무엇을 하나요?
```

`RunnableLambda`의 출력은 **감싼 함수가 반환한 값**입니다.

---

# 7. `assign()`으로 새 필드 추가하기

`RunnablePassthrough.assign()`은 기존 딕셔너리의 값을 유지하면서 새 필드를 추가합니다.

예를 들어:

```python
def extract_clean_question(data: dict) -> str:
    return data["question"].strip()


add_clean_question = RunnablePassthrough.assign(
    clean_question=RunnableLambda(
        extract_clean_question
    )
)
```

입력:

```python
input_data = {
    "question": "  Runnable이란 무엇인가요?  ",
    "student_level": "입문",
}
```

실행:

```python
result = add_clean_question.invoke(input_data)
```

결과:

```python
{
    "question": "  Runnable이란 무엇인가요?  ",
    "student_level": "입문",
    "clean_question": "Runnable이란 무엇인가요?"
}
```

기존 `question`과 `student_level`은 그대로 남아 있습니다.

---

# 8. `assign()` 문법

다음 코드를 보면:

```python
add_clean_question = RunnablePassthrough.assign(
    clean_question=RunnableLambda(
        extract_clean_question
    )
)
```

각 부분의 의미는 다음과 같습니다.

| 코드 | 의미 |
|---|---|
| `RunnablePassthrough` | 기존 필드 유지 |
| `assign(...)` | 새 필드 추가 |
| `clean_question=` | 새로 추가할 필드 이름 |
| `RunnableLambda(...)` | 새 필드의 값을 계산 |

즉:

```text
기존 딕셔너리
+
clean_question 계산
↓
새 딕셔너리
```

입니다.

`assign()`의 입력은 딕셔너리여야 합니다.

---

# 9. 두 번의 `assign()` 사용하기

이번 실습에서는 새 필드를 순서대로 두 개 추가합니다.

```text
1. clean_question 추가
2. clean_question을 이용해 question_length 추가
```

---

# 10. 질문 길이 계산 함수

```python
def count_clean_question(data: dict) -> int:
    clean_question = data["clean_question"]

    return len(clean_question)
```

이 함수는 이미 다음 값이 있다고 가정합니다.

```python
data["clean_question"]
```

따라서 `clean_question`을 먼저 만들어야 합니다.

---

# 11. 두 번째 `assign()`

```python
add_question_length = RunnablePassthrough.assign(
    question_length=RunnableLambda(
        count_clean_question
    )
)
```

이 단계에서는 기존 딕셔너리를 유지하면서:

```text
question_length
```

필드를 새로 추가합니다.

---

# 12. 두 `assign()` 연결하기

```python
transform_chain = (
    add_clean_question
    | add_question_length
)
```

입력:

```python
input_data = {
    "question": "  LCEL  ",
    "student_level": "입문",
}
```

실행:

```python
result = transform_chain.invoke(input_data)

print(result)
```

결과:

```python
{
    "question": "  LCEL  ",
    "student_level": "입문",
    "clean_question": "LCEL",
    "question_length": 4
}
```

흐름:

```text
원본 dict
 ↓
clean_question 추가
 ↓
question_length 추가
 ↓
최종 dict
```

---

# 13. 실행 순서가 중요한 이유

두 번째 함수에서는:

```python
data["clean_question"]
```

을 사용합니다.

따라서 반드시:

```text
clean_question 생성
↓
question_length 계산
```

순서여야 합니다.

올바른 순서:

```python
add_clean_question | add_question_length
```

잘못된 순서:

```python
add_question_length | add_clean_question
```

잘못된 순서에서는 아직 `clean_question`이 없기 때문에 `KeyError`가 발생합니다.

---

# 14. 입력값 확인하기

전체 체인을 실행하기 전에 입력이 올바른지 확인합니다.

```python
from typing import Any


def require_input_dict(
    data: Any
) -> dict[str, Any]:

    if not isinstance(data, dict):
        raise TypeError(
            "입력은 딕셔너리여야 합니다."
        )

    if "question" not in data:
        raise ValueError(
            "입력에 question 키가 필요합니다."
        )

    if not isinstance(
        data["question"], str
    ):
        raise TypeError(
            "question 값은 문자열이어야 합니다."
        )

    return data
```

확인하는 내용은 세 가지입니다.

```text
입력이 dict인가?

question 키가 있는가?

question 값이 str인가?
```

---

# 15. 빈 질문 확인하기

질문의 앞뒤 공백을 제거한 뒤 내용이 남아 있는지도 확인합니다.

```python
def extract_clean_question(
    data: dict
) -> str:

    clean_question = (
        data["question"].strip()
    )

    if not clean_question:
        raise ValueError(
            "question은 비어 있을 수 없습니다."
        )

    return clean_question
```

예를 들어:

```python
{
    "question": "   "
}
```

은 `strip()` 후 빈 문자열이 되기 때문에 오류가 발생합니다.

---

# 16. 전체 체인 만들기

입력 확인 → 질문 정리 → 길이 계산 순서로 연결합니다.

```python
validate_step = RunnableLambda(
    require_input_dict
)


add_clean_question = (
    RunnablePassthrough.assign(
        clean_question=RunnableLambda(
            extract_clean_question
        )
    )
)


add_question_length = (
    RunnablePassthrough.assign(
        question_length=RunnableLambda(
            count_clean_question
        )
    )
)


chain = (
    validate_step
    | add_clean_question
    | add_question_length
)
```

전체 흐름:

```text
입력 dict
 ↓
validate_step
 ↓
입력 확인
 ↓
clean_question 추가
 ↓
question_length 추가
 ↓
최종 dict
```

---

# 17. 단계별 변화

| 단계 | 하는 일 |
|---|---|
| `validate_step` | 입력이 올바른지 확인 |
| `add_clean_question` | 정리된 질문 추가 |
| `add_question_length` | 정리된 질문 길이 추가 |

각 단계의 결과 딕셔너리가 다음 단계의 입력이 됩니다.

---

# 18. 최종 결과

입력:

```python
{
    "question": "  RunnablePassthrough는 어떤 역할을 하나요?  ",
    "student_level": "입문",
}
```

최종 결과:

```python
{
    "question": "  RunnablePassthrough는 어떤 역할을 하나요?  ",
    "student_level": "입문",
    "clean_question": "RunnablePassthrough는 어떤 역할을 하나요?",
    "question_length": 32,
}
```

원래 필드는 그대로 유지되고:

```text
question
student_level
```

새 필드가 추가됩니다.

```text
clean_question
question_length
```

---

# 19. 전체 실행 과정

```text
입력 딕셔너리
 ↓
RunnablePassthrough
→ 기존 값 유지

RunnableLambda
→ 질문 정리

assign()
→ clean_question 추가

assign()
→ question_length 추가

최종 dict
```

이번 실습은 모델을 사용하지 않으므로 API Key가 필요하지 않고 API 호출도 발생하지 않습니다.

---

# 20. 자주 발생하는 오류

## 1) 입력이 딕셔너리가 아님

잘못된 입력:

```python
wrong_input = (
    "RunnablePassthrough는 무엇인가요?"
)
```

올바른 입력:

```python
correct_input = {
    "question":
        "RunnablePassthrough는 무엇인가요?"
}
```

---

## 2) `question` 키가 없음

잘못된 예:

```python
wrong_input = {
    "query": "assign은 무엇인가요?"
}
```

체인에서는 `question` 키를 사용하므로:

```python
{
    "question": "assign은 무엇인가요?"
}
```

형태로 전달해야 합니다.

---

## 3) `question`이 문자열이 아님

잘못된 예:

```python
wrong_input = {
    "question": 100
}
```

`strip()`을 사용해야 하기 때문에 질문은 문자열이어야 합니다.

---

## 4) 질문이 비어 있음

```python
wrong_input = {
    "question": "   "
}
```

공백 제거 후 내용이 없으므로 오류가 발생합니다.

---

## 5) `clean_question` KeyError

잘못된 순서:

```python
wrong_chain = (
    add_question_length
    | add_clean_question
)
```

올바른 순서:

```python
chain = (
    add_clean_question
    | add_question_length
)
```

먼저 `clean_question`을 만든 뒤 그 길이를 계산해야 합니다.

---

## 6) 원본 `question`이 바뀐다고 생각함

이번 실습에서는 원본 `question`을 수정하지 않습니다.

```text
question
→ 원본 그대로 유지

clean_question
→ 공백을 제거한 새 값
```

따라서 두 필드의 값이 다른 것이 정상입니다.

---

# 핵심 정리

## `RunnablePassthrough`

```python
RunnablePassthrough()
```

```text
입력
→ 그대로 전달
```

---

## `RunnableLambda`

```python
RunnableLambda(function)
```

```text
일반 Python 함수
→ Runnable로 변환
```

이후:

```python
.invoke()
```

또는:

```python
|
```

로 연결할 수 있습니다.

---

## `assign()`

```python
RunnablePassthrough.assign(
    clean_question=...
)
```

```text
기존 dict
+
새 필드
↓
필드가 추가된 dict
```

---

## 이번 강의의 전체 흐름

```text
{
    question,
    student_level
}

      ↓

clean_question 추가

      ↓

{
    question,
    student_level,
    clean_question
}

      ↓

question_length 추가

      ↓

{
    question,
    student_level,
    clean_question,
    question_length
}
```

가장 간단하게 기억하면:

```text
Passthrough
= 기존 값 유지

Lambda
= 일반 함수 실행

assign
= 기존 dict에 새 값 추가
```

그리고 여러 `assign()`을 연결할 때는 **뒤 단계에서 필요한 필드를 앞 단계에서 먼저 만들어야 합니다.**