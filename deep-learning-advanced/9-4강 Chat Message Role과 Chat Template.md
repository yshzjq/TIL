---
title: 9-4강 Chat Message Role과 Chat Template
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '9-4강 Chat Message Role과 Chat Template' 정리
---

## 1. Chat model 입력은 왜 messages인가

일반 Causal LM은 문자열 prompt를 이어받아 다음 텍스트를 생성한다

하지만 chat model은 대화 데이터로 학습되었기 때문에, 누가 말했는지에 대한 역할 정보가 중요하다

같은 문장이라도 system 지침인지, 사용자 질문인지, 모델의 이전 답변인지에 따라 의미가 달라진다


<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_chat_roles.png' | relative_url }}" alt="01_chat_roles.png" loading="lazy">


## 2. role의 의미

role은 메시지를 누가 어떤 목적으로 작성했는지 나타내는 역할표다.


- system: 모델이 따라야 할 전체 지침·규칙

- user: 사용자의 질문이나 요청

- assistant: 모델이 작성한 답변


| role | 의미 | 예시 |
| --- | --- | --- |
| `system` | 모델의 전반적 행동 지침 | “친절하고 간결하게 답하세요.” |
| `user` | 사용자의 요청 | “Transformer를 쉽게 설명해줘.” |
| `assistant` | 모델의 이전 답변 | “Transformer는 Attention 기반 구조다.” |

role 이름은 모델과 템플릿이 기대하는 값과 맞아야 한다.

`usr`, `bot`처럼 임의로 쓰면 template 적용이 실패하거나 예상과 다르게 동작할 수 있다.

Chat Template = system, user, assistant 대화를 해당 모델이 학습한 입력 형식으로 변환하는 규칙


## 3. Chat template이 필요한 이유

모델마다 대화 형식이 다르다.

어떤 모델은 `<|user|>` 같은 특수 토큰을 쓰고, 어떤 모델은 `[INST] ... [/INST]` 형식을 쓴다.

이 차이를 사용자가 직접 문자열로 만들면 실수하기 쉽다.

## 3. Chat template이 필요한 이유

모델마다 대화 형식이 다르다.

어떤 모델은 `<|user|>` 같은 특수 토큰을 쓰고, 어떤 모델은 `[INST] ... [/INST]` 형식을 쓴다. 

이 차이를 사용자가 직접 문자열로 만들면 실수하기 쉽다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_template_conversion.png' | relative_url }}" alt="02_template_conversion.png" loading="lazy">

Chat Template 변환

Chat template은 tokenizer 안에 들어 있는 규칙이다.

`messages` 리스트를 모델이 기대하는 하나의 문자열 또는 token ids로 바꾼다.


<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_model_specific_format.png' | relative_url }}" alt="03_model_specific_format.png" loading="lazy">

모델별 포맷이 다른 이유

## 선택 · 4. 대화 이력 구성

멀티턴 대화에서는 이전 user와 assistant 메시지를 순서대로 넣는다

모델은 이 이력을 보고 다음 assistant 답변을 생성한다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_message_history.png' | relative_url }}" alt="04_message_history.png" loading="lazy">

대화 이력 구성

대화가 길어지면 token budget을 초과할 수 있다. 

Token budget = 모델이 한 번의 처리에서 사용할 수 있는 토큰의 총 허용량

따라서 실제 서비스에서는 오래된 대화를 요약하거나, 필요한 일부 이력만 유지하는 전략이 필요하다





## 5. apply_chat_template 기본 사용

```python
# ============================================================
# apply_chat_template 기본 사용
# ============================================================
# 이 코드는 chat model tokenizer가 messages를 어떤 문자열로 바꾸는지 확인합니다.
# 모델 전체를 다운로드하지 않고 tokenizer만 불러와도 template 확인은 가능합니다.

from transformers import AutoTokenizer

# chat_template이 있는 instruct/chat 모델 tokenizer를 사용합니다.
# 모델 파일이 크더라도 tokenizer만 불러오는 것은 상대적으로 가볍습니다.
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# messages는 role/content 딕셔너리의 리스트입니다.
# 순서가 대화 순서이므로, 메시지를 임의로 섞으면 의미가 달라집니다.
messages = [
    {
        "role": "system",
        "content": "You are a helpful tutor who explains deep learning clearly.",
    },
    {
        "role": "user",
        "content": "Explain what a chat template is in one paragraph.",
    },
]

# tokenize=False로 두면 token id가 아니라 사람이 읽을 수 있는 문자열을 반환합니다.
# 디버깅할 때는 먼저 문자열을 확인하는 것이 좋습니다.
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(formatted_text)
```

출력

```

<|system|>
You are a helpful tutor who explains deep learning clearly.</s>
<|user|>
Explain what a chat template is in one paragraph.</s>
<|assistant|> # <-- add_generation_prompt=True

```

### 코드 해설

`apply_chat_template()`은 messages를 모델별 대화 문자열로 바꾼다.

`add_generation_prompt=True`는 다음 assistant 답변이 시작될 위치를 표시한다  생성에 사용할 입력을 만들 때는 보통 이 값을 킨다

add_generation_prompt=True

chat template 끝에 “이제 assistant 답변을 생성할 차례”라는 시작 표시를 추가하는 옵션
