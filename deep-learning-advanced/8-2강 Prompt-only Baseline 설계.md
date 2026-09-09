---
title: 8-2강 Prompt-only Baseline 설계
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '8-2강 Prompt-only Baseline 설계' 정리
---

## 1. Prompt-only baseline이 필요한 이유

Prompt-only = 모델은 그대로 두고, 질문이나 지시문(Prompt)만 잘 만들어서 문제를 해결하는 방식

Fine-tuning을 하기 전에 Prompt-only baseline을 만드는 이유는 두 가지

첫째, 학습 없이도 어느 정도 해결되는 문제인지 확인할 수 있다. 

둘째, PEFT나 Fine-tuning 결과가 실제로 baseline보다 좋아졌는지 비교할 수 있다.

Baseline이 없으면 Fine-tuning 결과가 좋아 보여도 그 개선이 학습 때문인지, prompt를 바꿨기 때문인지, 평가셋이 쉬웠기 때문인지 알기 어렵다.



<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_baseline_loop.png' | relative_url }}" alt="01_baseline_loop.png" loading="lazy">


Prompt-only Baseline 설계 루프

## 2. Baseline 설계 4요소

Prompt-only baseline에는 다음 네 가지가 필요하다

| 요소 | 설명 | 예시 |
| --- | --- | --- |
| 평가셋 | 모델이 풀어야 할 고정된 샘플 | 고객 문의 100개 |
| Prompt template | 모든 샘플에 적용할 입력 형식 | 역할, 라벨 목록, 출력 규칙 |
| Output schema | 모델이 따라야 하는 출력 형식 | label only 또는 JSON |
| Metric | 성능을 판단하는 기준 | Accuracy, Macro-F1, format success |

<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_prompt_template_parts.png' | relative_url }}" alt="02_prompt_template_parts.png" loading="lazy">

Prompt Template 구성요소

Prompt template을 바꾸면 실험 조건이 바뀐 것입니다. 결과표에는 반드시 prompt 버전을 남겨야 한다

## 3. Prompt template 만들기

Prompt template = 모델에게 문제를 내는 고정된 양식

프롬프트 조건 자체는 동일하게 유지해서 성능을 공정하게 비교

고객 문의 intent 분류 baseline을 위한 prompt template을 만듭니다.

실제 모델 호출은 하지 않고, **모델에 보낼 입력 문자열을 일관되게 생성하는 함수**를 먼저 만든다

```python
# ============================================================
# Prompt-only baseline: prompt template 함수 만들기
# ============================================================
# 이 함수는 입력 text를 받아 LLM에게 보낼 prompt 문자열을 반환합니다.
# 모델 호출 코드는 환경마다 다를 수 있으므로 이 강에서는 prompt 생성과 결과 저장 구조를 먼저 다룹니다.

from textwrap import dedent

# 가능한 라벨을 고정합니다.
# 라벨 목록이 바뀌면 평가 기준도 바뀌므로 실험 결과와 함께 저장해야 합니다.
LABELS = ["refund", "delivery", "account", "other"]

# prompt 버전은 실험 추적을 위한 이름입니다.
# v1, v2처럼 단순 번호를 붙이되, 변경 이유를 별도 메모로 남기는 것이 좋습니다.
PROMPT_VERSION = "prompt_v1_label_only"

def build_prompt(text: str) -> str:
    """고객 문의 문장을 intent label로 분류하기 위한 prompt를 만듭니다.

    Args:
        text: 고객이 작성한 원문 메시지입니다.

    Returns:
        LLM에 전달할 하나의 prompt 문자열입니다.
    """
    prompt = dedent(f"""
    You are a customer support intent classifier.

    Task:
    Classify the user message into exactly one label.

    Allowed labels:
{", ".join(LABELS)}

    Output rule:
    Return only one label from the allowed labels.
    Do not include explanation.

    User message:
{text}
    """).strip()

    return prompt

sample_text = "My package has not arrived for two weeks."
print(build_prompt(sample_text))
```
출력
```
You are a customer support intent classifier.

    Task:
    Classify the user message into exactly one label.

    Allowed labels:
refund, delivery, account, other

    Output rule:
    Return only one label from the allowed labels.
    Do not include explanation.

    User message:
My package has not arrived for two weeks.
```

`build_prompt()`는 입력 문장마다 같은 구조의 prompt를 만들기 위한 함수입니다.

이처럼 template을 함수로 관리하면 평가셋 전체에 같은 조건을 적용하기 쉽다. 

또한 prompt version을 결과표에 저장하면 나중에 `prompt_v1`과 `prompt_v2` 결과를 비교할 수 있다.


## 4. 출력 schema와 parsing

Prompt-only baseline에서 가장 흔한 문제는 모델이 형식을 어기는 것이다.

예를 들어 “label만 출력하세요”라고 했는데 `The answer is refund.`처럼 설명을 붙이는 경우가 있다. 

따라서 출력 schema를 정하고, 모델 출력이 그 schema를 지켰는지 검사해야 한다



<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_output_schema.png' | relative_url }}" alt="03_output_schema.png" loading="lazy">


출력 형식을 고정하는 이유

```python
# ============================================================
# Label-only 출력 검증 함수
# ============================================================
# 실제 LLM 출력은 항상 우리가 원하는 형식을 지키지 않을 수 있습니다.
# 아래 함수는 모델 출력 문자열을 받아 허용된 라벨인지 검사합니다.

from typing import Optional

def parse_label_only(raw_output: str, allowed_labels: list[str]) -> Optional[str]:
    """모델 출력에서 label-only 형식을 검증합니다.

    Args:
        raw_output: LLM이 반환한 원문 출력입니다.
        allowed_labels: 허용된 label 문자열 목록입니다.

    Returns:
        형식이 올바르면 label 문자열을 반환합니다.
        형식이 틀리거나 허용되지 않은 label이면 None을 반환합니다.
    """
    # 모델이 앞뒤 공백이나 줄바꿈을 붙일 수 있으므로 strip으로 제거합니다.
    normalized = raw_output.strip()

    # 소문자 기준으로 비교하면 Refund, refund처럼 대소문자 차이를 완화할 수 있습니다.
    normalized_lower = normalized.lower()
    allowed_lower = [label.lower() for label in allowed_labels]

    if normalized_lower in allowed_lower:
        # 원래 allowed_labels에 있는 표준 라벨 형태로 되돌립니다.
        return allowed_labels[allowed_lower.index(normalized_lower)]

    # label이 아닌 설명문, 잘못된 라벨, 여러 라벨을 반환한 경우 None으로 처리합니다.
    return None

examples = ["refund", "Refund", "The label is refund.", "payment"]
for output in examples:
    parsed = parse_label_only(output, LABELS)
    print(f"raw={output!r} -> parsed={parsed}")
```

출력

```
raw='refund' -> parsed=refund
raw='Refund' -> parsed=refund
raw='The label is refund.' -> parsed=None
raw='payment' -> parsed=None
```

### 코드 해설

`parse_label_only()`는 모델 성능뿐 아니라 **형식 준수율**을 측정하기 위한 함수다.

예측 내용은 맞아도 형식이 깨지면 운영 시스템에서 후처리 오류가 날 수 있다.

따라서 baseline 단계부터 parsing 실패를 따로 기록해야 합니다.

Parsing = 모델의 출력 문자열을 프로그램이 쓸 수 있는 데이터로 해석하는 것

## 선택 · 5. 결과표 저장과 오류 유형화

Prompt-only 결과는 나중에 PEFT/Fine-tuning과 비교해야 하므로 표 형태로 저장했다



<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_baseline_report.png' | relative_url }}" alt="04_baseline_report.png" loading="lazy">

label = 클래스 이름 자체

gold = 그 샘플의 정답으로 지정된 label

```python
# ============================================================
# Prompt-only 결과표 예시 만들기
# ============================================================
# 실제 모델 호출 대신, 예시 raw_output을 사용해 결과표 구조를 확인합니다.
# 실제 LLM을 연결할 때는 raw_output 자리에 모델 응답을 넣습니다.

import pandas as pd

# 평가셋 예시입니다.
# text는 모델 입력, gold는 사람이 정한 정답 라벨입니다.
eval_rows = [
    {"id": 1, "text": "I want my money back.", "gold": "refund"},
    {"id": 2, "text": "Where is my order?", "gold": "delivery"},
    {"id": 3, "text": "I cannot reset my password.", "gold": "account"},
]

# 실제로는 LLM 호출 결과가 들어갑니다.
# 여기서는 parsing 성공/실패 예시를 섞어 둡니다.
raw_outputs = ["refund", "The label is delivery.", "account"]

records = []
for row, raw_output in zip(eval_rows, raw_outputs):
    parsed_label = parse_label_only(raw_output, LABELS)
    records.append({
        "id": row["id"],
        "prompt_version": PROMPT_VERSION,
        "text": row["text"],
        "gold": row["gold"],
        "raw_output": raw_output,
        "prediction": parsed_label,
        "format_success": parsed_label is not None,
        "is_correct": parsed_label == row["gold"],
    })

result_df = pd.DataFrame(records)
display(result_df)

format_success_rate = result_df["format_success"].mean()
accuracy_on_parsed = result_df.loc[result_df["format_success"], "is_correct"].mean()

print("format_success_rate:", format_success_rate)
print("accuracy_on_parsed:", accuracy_on_parsed)
```
출력

| id | prompt_version | text | gold | raw_output | prediction | format_success | is_correct |
|---:|---|---|---|---|---|---|---|
| 1 | prompt_v1_label_only | I want my money back. | refund | refund | refund | True | True |
| 2 | prompt_v1_label_only | Where is my order? | delivery | The label is delivery. | None | False | False |
| 3 | prompt_v1_label_only | I cannot reset my password. | account | account | account | True | True |

```
format_success_rate: 0.6666666666666666
accuracy_on_parsed: 1.0
```



