---
title: 8-1강 Prompt Engineering과 Prompt-tuning, PEFT 구분
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '8-1강 Prompt Engineering과 Prompt-tuning, PEFT 구분' 정리
---

## 1. 왜 방법을 구분해야 할까?

LLM을 특정 업무에 맞게 쓰고 싶을 때 선택지는 하나가 아니다

같은 문제라도 prompt만 잘 쓰면 충분한 경우, 검색 문서를 붙여야 하는 경우, 모델의 행동 자체를 학습해야 하는 경우가 다르다.

이 구분을 하지 않으면 비용이 큰 Fine-tuning을 너무 빨리 선택하거나,
<br>
반대로 학습이 필요한 문제를 prompt만 고치며 오래 붙잡게 된다.



## 2. Prompt Engineering과 Prompt-only


Prompt Engineering은 모델에게 일을 시키기 위한 **입력 설계**다.

모델이 같은 질문에도 다른 방식으로 답하는 이유는 입력에 들어간 역할, 지시문, 예시, 출력 형식이 다르기 때문이다

Prompt-only 접근에서는 모델의 가중치가 전혀 바뀌지 않는다.


바뀌는 것 입력요소들

| 요소 | 설명 | 예시 |
| --- | --- | --- |
| 역할 | 모델이 어떤 관점으로 답할지 지정합니다. | “당신은 고객 문의 분류기입니다.” |
| 지시문 | 모델이 수행할 일을 명확히 적습니다. | “아래 문장을 하나의 intent로 분류하세요.” |
| 라벨 목록 | 가능한 출력 범위를 제한합니다. | `refund`, `delivery`, `account`, `other` |
| 출력 형식 | 후처리를 쉽게 하기 위해 형식을 고정합니다. | JSON 또는 label만 출력 |
| 예시 | few-shot으로 판단 기준을 보여줍니다. | 입력/정답 예시 2~3개 |


<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_prompt_vs_peft.png' | relative_url }}" alt="02_prompt_vs_peft.png" loading="lazy">

Prompt Engineering과 PEFT의 차이

### 예시 코드: Prompt-only 입력 만들기

```python
# ============================================================
# Prompt-only baseline용 prompt 템플릿 예시
# ============================================================
# 이 코드는 실제 LLM API를 호출하지 않습니다.
# 목적은 모델에 보내기 전 prompt 문자열을 어떻게 일관되게 만들지 확인하는 것입니다.
# prompt도 실험 자산이므로 버전, 라벨 목록, 출력 형식을 함께 기록해야 합니다.

from textwrap import dedent

# 가능한 라벨 목록입니다.
# 모델이 자유롭게 답하지 않도록 label space를 명시합니다.
LABELS = ["refund", "delivery", "account", "other"]

# prompt 버전은 나중에 실험 결과와 함께 저장합니다.
# v1과 v2의 성능이 다를 수 있으므로 prompt 변경 내역을 추적해야 합니다.
PROMPT_VERSION = "prompt_v1_label_only"

# 테스트할 고객 문의 문장입니다.
# 실제 프로젝트에서는 evaluation set의 각 문장을 반복해서 넣습니다.
text = "I paid twice and want my money back."

prompt = dedent(f"""
You are a customer support intent classifier.

Task:
Classify the user's message into exactly one label.

Allowed labels:
{", ".join(LABELS)}

Output rule:
Return only the label. Do not add explanation.

User message:
{text}
""").strip()

print("prompt version:", PROMPT_VERSION)
print(prompt)
```

```
prompt version: prompt_v1_label_only
You are a customer support intent classifier.

Task:
Classify the user's message into exactly one label.

Allowed labels:
refund, delivery, account, other

Output rule:
Return only the label. Do not add explanation.

User message:
I paid twice and want my money back.
```


### 코드 해설

이 코드는 모델 호출보다 앞 단계인 **입력 표준화**를 다룬다

Prompt-only 실험이 재현 가능하려면 같은 문장은 항상 같은 템플릿에 들어가야 한다
<br>
또한 출력 형식을 `label only`로 제한하면 나중에 정확도나 Macro-F1을 계산하기 쉽다.


## 3. Prompt-tuning은 왜 prompt engineering과 다른가

Prompt-tuning은 이름에 prompt가 들어가지만, prompt engineering과 다르다
<br>
Prompt engineering은 사람이 읽는 자연어 prompt를 고치는 작업이다
<br>
반면 Prompt-tuning은 모델 입력 앞쪽에 **학습 가능한 soft prompt 벡터**를 붙이고, 이 벡터만 학습하는 방법이다

Soft prompt는 실제 단어가 아니라, 모델 입력 앞에 추가되는 학습 가능한 가짜 토큰 embedding이다.

즉 prompt engineering은 사람이 문장을 편집하는 것이고, <br>
prompt-tuning은 모델이 학습할 수 있는 작은 파라미터를 추가하는 것이다

| 구분 | Prompt Engineering | Prompt-tuning |
| --- | --- | --- |
| 학습 여부 | 학습 없음 | 학습 있음 |
| 바뀌는 대상 | 사람이 읽는 prompt 문장 | 학습 가능한 soft prompt 파라미터 |
| 모델 본체 | 고정 | 보통 고정 |
| 데이터 필요 | 없어도 가능 | 학습 데이터 필요 |
| 결과물 | prompt template | soft prompt checkpoint |


Soft prompt는 사람이 읽을 수 있는 단어가 아니다.<br>
모델 내부 embedding 공간에 추가되는 학습 가능한 벡터로 이해하면 된다


## 4. PEFT와 Full Fine-tuning

PEFT는 큰 모델 전체를 학습하지 않고, 작은 추가 파라미터만 학습하는 접근이다. 

대표적으로 LoRA, Prompt Tuning, Prefix Tuning, Adapter 계열 방법이 있다. 

Full Fine-tuning은 모델의 전체 파라미터를 업데이트 한다. 

데이터가 충분하고 자원도 넉넉하며 모델을 완전히 특정 업무에 맞추고 싶을 때 사용할 수 있다.

하지만 저장 비용, GPU 메모리, 과적합 위험, 배포 관리 부담이 커질 수 있다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_adapter_concept.png' | relative_url }}" alt="04_adapter_concept.png" loading="lazy">

Adapter 관점으로 보는 PEFT

| 방법 | 장점 | 단점 | 먼저 고려할 상황 |
| --- | --- | --- | --- |
| Prompt-only | 빠르고 비용이 낮습니다. | 복잡한 도메인 기준을 안정적으로 학습하지 못할 수 있습니다. | 기준 성능 측정 |
| Prompt-tuning | 모델 본체를 고정하면서 학습 가능합니다. | 개념이 추상적이고 task에 따라 효과가 다릅니다. | 가벼운 학습 실험 |
| LoRA/PEFT | 메모리와 저장 비용을 줄입니다. | adapter 관리와 병합 전략이 필요합니다. | 실무형 Fine-tuning 시작점 |
| Full Fine-tuning | 모델 전체를 강하게 적응시킬 수 있습니다. | 비용과 위험이 큽니다. | 충분한 데이터·자원·운영 계획이 있을 때 |

## 5. 방법 선택 흐름

새로운 태스크를 만났다면 다음 순서를 권장한다

여기서 태스크(task) 는 쉽게 말하면 모델에게 새로 맡기려는 문제나 업무를 뜻

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_decision_flow.png' | relative_url }}" alt="03_decision_flow.png" loading="lazy">

방법 선택 흐름

1. Prompt-only baseline을 만든다.

2. 고정된 평가셋에서 성능과 오류를 확인한다.

3. 실패 원인이 지식 부족인지, 형식 문제인지, 모델 행동 문제인지 나눈다

4. 지식 부족이면 RAG를 검토한다

5. 반복되는 행동 오류가 있고 학습 데이터가 있으면 PEFT/Fine-tuning을 검토한다

6. 비용과 운영 제약을 보고 PEFT와 Full Fine-tuning 중 선택한다

Fine-tuning은 Prompt-only를 무조건 대체하는 방법이 아니다
 
데이터와 평가 체계가 준비되었을 때 선택하는 모델 적응 방법입니다.


