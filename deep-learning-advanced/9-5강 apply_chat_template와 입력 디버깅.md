---
title: 9-5강 apply_chat_template와 입력 디버깅(선택 정리 필요)
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '9-5강 apply_chat_template와 입력 디버깅' 정리
---

## 1. 디버깅은 messages에서 시작한다

Chat template 오류는 대부분 messages 구조에서 시작한다.

role 이름이 틀렸거나 content가 비어 있거나, assistant 답변을 user 메시지로 넣는 경우가 있다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_debugging_flow.png' | relative_url }}" alt="01_debugging_flow.png" loading="lazy">

apply_chat_template 디버깅 흐름

## 2. formatted text 확인

먼저 `tokenize=False`로 template 결과 문자열을 확인한다
 
이 단계에서는 모델을 실행하지 않아도 된다



```python
# ============================================================
# formatted text 확인하기
# ============================================================
# 이 코드는 chat template이 messages를 어떤 문자열로 바꾸는지 확인합니다.
# 생성 품질을 보기 전에 입력 포맷을 눈으로 확인하는 것이 목적입니다.

from transformers import AutoTokenizer

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Give me two tips for learning Transformers."},
]

formatted_without_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

formatted_with_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("[without generation prompt]")
print(formatted_without_prompt)

print("\n[with generation prompt]")
print(formatted_with_prompt)
```

출력

```
[without generation prompt]
<|system|>
You are a concise assistant.</s>
<|user|>
Give me two tips for learning Transformers.</s>


[with generation prompt]
<|system|>
You are a concise assistant.</s>
<|user|>
Give me two tips for learning Transformers.</s>
<|assistant|>
```

### 코드 해설

 `add_generation_prompt=True`는 모델이 다음 assistant 답변을 생성해야 할 위치를 알려주는 형식을 추가한다
 
모델별 template 구현에 따라 실제 문자열은 다르게 보일 수 있다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_generation_prompt.png' | relative_url }}" alt="02_generation_prompt.png" loading="lazy">

## 3. tokenized chat 확인

문자열 확인이 끝나면 `tokenize=True` 또는 `return_tensors="pt"`로 실제 모델 입력 Tensor를 확인한다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_tokenized_chat.png' | relative_url }}" alt="03_tokenized_chat.png" loading="lazy">

Tokenized Chat 확인

```python
# ============================================================
# tokenized chat 확인하기
# ============================================================
# return_tensors="pt"와 return_dict=False를 함께 사용해 PyTorch Tensor를 받습니다.
# 생성 모델에 넣기 전 input_ids shape과 token 길이를 확인합니다.

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=False,
)

# encoded는 보통 shape [B, L]의 Tensor입니다.
# B는 batch size, L은 tokenized chat의 길이입니다.
print("encoded type:", type(encoded))
print("encoded shape:", encoded.shape)
print("token count:", encoded.shape[-1])

# decode로 다시 문자열을 확인하면 실제 모델이 읽는 토큰 흐름을 점검할 수 있습니다.
# skip_special_tokens=False로 두면 특수 토큰까지 확인할 수 있습니다.
decoded = tokenizer.decode(encoded[0], skip_special_tokens=False)
print(decoded)
```

출력

```
encoded type: <class 'torch.Tensor'>
encoded shape: torch.Size([1, 43])
token count: 43
<|system|>
You are a concise assistant.</s> 
<|user|>
Give me two tips for learning Transformers.</s> 
<|assistant|>
```

### 코드 해설

`encoded.shape[-1]`은 입력 토큰 길이다.

이 길이가 너무 길면 모델의 context length를 넘을 수 있다. 

실제 생성에서는 여기에 `max_new_tokens`로 생성할 토큰까지 더해 전체 token budget을 고려해야 한다

## 4. add_generation_prompt 확인

`add_generation_prompt=True`는 생성할 답변의 시작 위치를 만들어 준다.

Chat model에게 “이제 assistant가 답변할 차례”라는 포맷을 제공하는 역할로 이해하면 된다

```python
# ============================================================
# add_generation_prompt 차이 확인 함수
# ============================================================
# 같은 messages에 대해 옵션만 바꿔 결과 길이와 마지막 부분을 비교합니다.

def inspect_template(add_generation_prompt: bool) -> None:
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=False,
    )

    print("add_generation_prompt:", add_generation_prompt)
    print("token length:", encoded.shape[-1])
    print("last 300 chars:")
    print(formatted[-300:])
    print("-" * 80)

inspect_template(False)
inspect_template(True)
```

출력

```

add_generation_prompt: False
token length: 36
last 300 chars:
<|system|>
You are a concise assistant.</s>
<|user|>
Give me two tips for learning Transformers.</s>

--------------------------------------------------
add_generation_prompt: True
token length: 43
last 300 chars:
<|system|>
You are a concise assistant.</s>
<|user|>
Give me two tips for learning Transformers.</s>
<|assistant|>

--------------------------------------------------

```

### 코드 해설

add_generation_prompt: True False 일때를 비교한 코드

실습에서는 전체 문자열보다 마지막 부분을 보는 것이 유용합니다.

대부분의 generation prompt는 끝부분에 assistant 답변 시작 표시를 추가하기 때문입니다.


## 5. 자주 나는 오류

04_common_errors.png

자주 나는 오류

| 오류 | 원인 | 해결 |
| --- | --- | --- |
| `chat_template`이 없습니다 | tokenizer에 template이 정의되지 않았습니다. | 다른 chat model tokenizer 사용 또는 template 정의 확인 |
| role 오류 | `user`, `assistant`, `system` 외 임의 role 사용 | 모델 문서의 role 규칙 확인 |
| 출력이 이상합니다 | generation prompt 누락 가능성 | `add_generation_prompt=True` 확인 |
| token 길이가 너무 깁니다 | 대화 이력이 과도하게 길어졌습니다. | 요약, truncation, history pruning |
| 특수 토큰이 중복됩니다 | 직접 포맷한 문자열에 template을 또 적용했습니다. | messages 원본에만 template 적용 |


## 8. Decoder-only Batch의 Left Padding과 Context Budget (선택 학습)

