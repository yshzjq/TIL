---
title: 9-1강 generate()와 max_new_tokens
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '9-1강 generate()와 max_new_tokens' 정리
---

## 1. Forward pass와 generate()

Forward pass는 주어진 입력에 대해 모델 출력을 한 번 계산하는 과정

Causal LM에서 forward pass를 하면 각 위치마다 vocabulary 전체에 대한 logits가 나온다

반면 generate()는 이 logits를 이용해 다음 토큰을 선택하고, 선택한 토큰을 입력 뒤에 붙인 뒤 다시 forward pass를 반복한다

Forward pass와 generate의 차이

| 구분 | Forward pass | generate() |
| --- | --- | --- |
| 목적 | 주어진 입력의 logits 계산 | 새 텍스트 생성 |
| 반복 여부 | 보통 한 번 | 여러 번 반복 |
| 출력 | logits, hidden states 등 | 생성된 token ids 또는 text |
| 사용 예 | loss 계산, 다음 토큰 점수 확인 | 챗봇 답변, 문장 생성 |



## 2. Autoregressive 생성 루프

Autoregressive = 이전 결과를 참고해서 다음 결과를 순서대로 하나씩 생성하는 방식

GPT 계열 모델은 보통 autoregressive 방식으로 생성한다.

즉 이미 있는 토큰을 보고 다음 토큰을 예측하고, 그 토큰을 다시 입력에 붙입니다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_generation_loop.png' | relative_url }}" alt="03_generation_loop.png" loading="lazy">

생성 루프

1. prompt를 tokenizer로 `input_ids`로 바꾼다

2. 모델이 다음 토큰 후보 logits를 계산한다

3. decoding strategy에 따라 토큰 하나를 선택한다

4. 선택한 토큰을 `input_ids` 뒤에 붙인다

5. 종료 조건이 만족될 때까지 반복한다


## 3. max_length와 max_new_tokens

`max_length`는 입력과 출력 전체 길이를 제한한다.

`max_new_tokens`는 새로 생성할 토큰 수만 제한합니다.

max_new_tokens의 의미

| 설정 | 기준 | 예시 |
| --- | --- | --- |
| `max_length=50` | 입력+출력 전체 길이 | prompt가 30토큰이면 새 토큰은 최대 20개 정도 |
| `max_new_tokens=50` | 새로 생성할 토큰 수 | prompt 길이와 별개로 새 토큰 최대 50개 |

실습에서는 `max_new_tokens`를 사용하는 것이 이해하기 쉽다. 

입력 prompt 길이가 매번 달라도 출력량을 비슷하게 제어할 수 있기 때문이다

## 4. 생성 종료 조건

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_stop_tokens.png' | relative_url }}" alt="04_stop_tokens.png" loading="lazy">

생성 종료 조건

- `max_new_tokens`에 도달한다

- EOS token이 생성된다

- 사용자가 지정한 stopping criteria를 만족한다

- 서비스에서 timeout이나 stop string 기준으로 중단한다


종료 조건을 너무 크게 잡으면 불필요하게 긴 답변이나 반복 출력이 나올 수 있다

## 5. generate() 코드 읽기

다음 코드는 작은 GPT 계열 모델로 `generate()`를 실행하는 예시



```python
# ============================================================
# generate() 기본 사용 예시
# ============================================================
# 이 코드는 Hugging Face Hub에서 작은 Causal LM 모델과 tokenizer를 내려받습니다.
# Colab에서 실행할 때는 인터넷 연결과 런타임 메모리를 확인하세요.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# 재현 가능한 sampling 실험을 위해 seed를 고정합니다.
# Greedy decoding은 결정적이지만, sampling을 사용할 때는 seed가 결과에 영향을 줄 수 있습니다.
set_seed(42)

# 교육용으로 작은 모델을 사용합니다.
# 실제 품질 평가용 모델이 아니라 generate() 흐름을 관찰하기 위한 모델입니다.
MODEL_ID = "sshleifer/tiny-gpt2"

# tokenizer는 prompt 문자열을 input_ids로 바꿉니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Causal LM head가 붙은 모델을 불러옵니다.
# AutoModelForCausalLM은 다음 토큰 예측 logits를 반환할 수 있는 모델 클래스입니다.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# GPU가 있으면 GPU를 사용하고, 없으면 CPU를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

prompt = "Deep learning helps machines"

# return_tensors="pt"는 PyTorch Tensor 형식으로 반환하라는 의미입니다.
# input_ids shape은 [B, L]입니다. 여기서는 문장 1개이므로 B=1입니다.
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

print("input_ids shape:", inputs["input_ids"].shape)
print("prompt token count:", inputs["input_ids"].shape[-1])

# generate()는 다음 토큰 선택을 반복합니다.
# max_new_tokens=20은 prompt 뒤에 새 토큰을 최대 20개 생성하겠다는 뜻입니다.
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,  # False이면 greedy decoding을 사용합니다.
    )

print("generated_ids shape:", generated_ids.shape)
print("total token count:", generated_ids.shape[-1])

# decode는 token id를 사람이 읽는 문자열로 되돌립니다.
# skip_special_tokens=True는 EOS 같은 특수 토큰을 출력 문자열에서 제거합니다.
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
```

출력

```
input_ids shape: torch.Size([1, 4])
prompt token count: 4
generated_ids shape: torch.Size([1, 24])
total token count: 24
Deep learning helps machines factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors
```

### 코드 해설

`model.generate()`의 입력은 tokenizer가 만든 `input_ids`와 `attention_mask`입니다. 출력 `generated_ids`는 prompt 토큰과 새로 생성된 토큰을 포함한 전체 토큰 ID입니다. 따라서 새로 생성된 부분만 보고 싶다면 입력 길이를 기준으로 잘라낼 수 있습니다.

```python
# ============================================================
# 새로 생성된 토큰만 분리하기
# ============================================================
# generated_ids에는 prompt 토큰과 생성 토큰이 모두 들어 있습니다.
# input_length를 기준으로 뒤쪽만 자르면 새로 생성된 토큰만 확인할 수 있습니다.

input_length = inputs["input_ids"].shape[-1]
new_token_ids = generated_ids[0, input_length:]
new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

print("new token count:", new_token_ids.shape[-1])
print("new text only:", new_text)
```

출력

```
new token count: 20
new text only:  factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors
```


## 8. Prefill·Decode와 Cache-aware Generation (선택 학습)
