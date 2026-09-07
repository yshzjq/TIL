---
title: 5-4강 GPT Decoder-only와 Causal LM
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '5-4강 GPT Decoder-only와 Causal LM .md' 정리
---

## 1. GPT란 무엇인가?

GPT는 **Generative Pre-trained Transformer**의 약자다

| 단어 | 의미 |
| --- | --- |
| Generative | 이전 문맥을 바탕으로 새 토큰을 생성합니다. |
| Pre-trained | 대규모 텍스트에서 먼저 언어 패턴을 학습합니다. |
| Transformer | Attention 기반 Transformer Block을 사용합니다. |

**GPT 계열은 Transformer Decoder-only Block을 여러 층 쌓고,<br> 이전 토큰을 바탕으로 다음 토큰을 예측하도록 사전학습된 생성형 언어모델 계열이다.**



### GPT와 원래 Transformer Decoder의 차이

원래 Encoder-Decoder Transformer의 Decoder는 두 정보를 사용한다

1. 이전 Target 토큰
2. Encoder가 만든 memory

반면 GPT형 Decoder-only 모델에는 별도 Encoder가 없다.

```
GPT Decoder-only Block
= Causal Self-Attention
+ FFN
+ Residual
+ LayerNorm
```

따라서 GPT에는 원래 Encoder-Decoder Decoder의 Cross-Attention이 일반적으로 없다.


## 2. Decoder-only 구조


GPT의 대표 흐름은 다음과 같다.

```
Prompt text
-> Tokenizer
-> input_ids [B,L]
-> Token Embedding + Position
-> Decoder-only Transformer Stack
-> hidden state [B,L,D]
-> LM Head
-> logits [B,L,V]
```

각 위치의 logits는 해당 위치까지의 문맥을 바탕으로 **다음 토큰 후보 전체**에 대한 점수를 나타낸다

예를 들어 입력이 다음과 같다고 가정

```
AI can help
```

마지막 위치 logits는 다음 토큰 후보를 평가한다

```
people, students, companies, us, ...
```

## 3. Causal Self-Attention

GPT는 각 위치에서 오른쪽 미래 토큰을 볼 수 없다.

```
위치 1: 1번 위치만 참고
위치 2: 1~2번 위치 참고
위치 3: 1~3번 위치 참고
위치 4: 1~4번 위치 참고
```

### 왜 미래 토큰을 차단할까요?

다음 토큰 예측 문제에서 정답이 입력에 미리 보이면 학습이 성립하지 않는다


```
입력: I learn
정답: AI
```

`AI` 위치를 예측할 때 `AI`를 이미 참고할 수 있다면 모델은 정답을 복사할 수 있다.<br>
Causal Mask는 이런 정보 누출을 막는다


> Causal은 “원인과 결과를 과학적으로 추론한다”는 뜻이 아니라, **시간·토큰 순서상 미래 위치를 차단한다**는 의미로 사용된다

## 4. Causal Language Modeling Objective

Causal LM은 토큰 시퀀스에서 각 위치의 다음 토큰을 예측한다



예시:

```
전체 문장: <BOS> I learn AI <EOS>

입력 위치: <BOS> | I     | learn | AI
정답 위치: I     | learn | AI    | <EOS>
```

### 4-1. 한 칸 Shift의 의미

입력과 정답은 같은 길이처럼 보이지만 학습 의미가 한 칸 어긋난다

```
<BOS>를 보고 I 예측
<BOS> I를 보고 learn 예측
<BOS> I learn을 보고 AI 예측
<BOS> I learn AI를 보고 <EOS> 예측
```

### 4-2. 학습 시 여러 위치를 한 번에 계산할 수 있다

Causal Mask를 적용하면 미래 정보는 차단되지만,<br>
학습 데이터에는 전체 정답 문장이 이미 있다. 

따라서 Transformer는 여러 위치의 logits를 큰 행렬 연산으로 한 번에 계산할 수 있다.

학습할 때는 문장 전체를 한 번 모델에 넣고, 각 위치의 “다음 토큰 예측 점수”를 동시에 계산할 수 있다 라는 뜻이다

```
학습: 여러 위치의 next-token loss를 병렬 계산
생성: 새 토큰을 하나씩 순차 생성
```

## 5. 학습과 생성의 차이

| 구분 | 학습 | 생성 |
| --- | --- | --- |
| 정답 토큰 | 전체 문장에 존재 | 미리 알 수 없음 |
| 계산 | 여러 위치 logits를 함께 계산 | 마지막 위치에서 다음 토큰을 선택 |
| 파라미터 업데이트 | 있음 | 없음 |
| 반복 방식 | Batch 단위 forward/backward | 토큰 선택 후 입력에 추가 |

### Teacher Forcing 직관

Causal LM 학습은 문장의 각 위치에서 다음 토큰을 예측하는 학습

Causal LM 학습에서는 각 위치의 이전 문맥으로 **실제 정답 토큰 시퀀스**를 사용한다.<br>

이를 넓은 의미에서 Teacher Forcing 관점으로 이해할 수 있다. <br>
Causal LM 학습 방식이 Teacher Forcing과 비슷한 구조를 가진다

생성 단계에서는 실제 다음 토큰이 없기 때문에 모델이 방금 생성한 토큰을 다음 입력으로 사용한다.

생성할 때는 정답 문장이 준비되어 있지 않기 때문에, 모델이 자기가 방금 만든 토큰을 이어서 다음 예측에 사용한다

생성 단계에서는 “정답을 참고해서 이어가는 것”이 아니라, 자기 출력값을 다시 입력으로 사용하면서 문장을 이어간다.

## 6. Autoregressive Generation

Autoregressive 생성 반복

생성 과정

```
1. Prompt를 입력합니다.
2. 마지막 위치의 next-token logits를 계산합니다.
3. 규칙에 따라 토큰 하나를 선택합니다.
4. 선택한 토큰을 입력 뒤에 추가합니다.
5. 종료 조건까지 반복합니다.
```

### 종료 조건 예시

- EOS Token을 생성
- `max_new_tokens`에 도달했습니다.
- 사용자가 지정한 Stop Sequence를 만났습니다.
- 외부 애플리케이션이 생성을 중단했습니다.

### 토큰 선택 규칙

- Greedy: 가장 높은 점수의 토큰을 선택합니다.
- Sampling: 확률 분포에서 토큰을 샘플링합니다.
- Beam Search: 여러 후보 경로를 동시에 유지합니다.

## 7. GPT 입력과 출력

### 7-1. 입력

```
input_ids:      [B,L]
attention_mask: [B,L]
```

GPT-2 Tokenizer는 기본적으로 BERT의 `[CLS]`, `[SEP]`, `[MASK]`를 사용하지 않습니다.

### 7-2. hidden state

```
[B,L,D]
```
각 위치가 왼쪽 문맥을 반영한 내부 표현


### 7-3. LM logits

```
[B,L,V]
```

생성할 다음 토큰을 고를 때는 일반적으로 **현재 입력의 마지막 실제 토큰 위치** logits를 사용한다

```python
next_token_logits = outputs.logits[:, -1, :]

첫 번째 :  → 모든 Batch
두 번째 -1 → 마지막 Token 위치
세 번째 :  → Vocabulary 전체

# Shape: [B,V]
```

> Padding이 포함된 Batch에서 단순히 `[:, -1, :]`를 사용하면 마지막 위치가 `[PAD]`일 수 있다.<br>
>실제 Batch 생성 코드는 padding side와 attention mask를 고려해야 합니다. 이 강의에서는 단일 Prompt를 사용해 흐름을 단순화한다


## 선택 · 8. 연습 문제: GPT Tokenizer와 모델 불러오기

### 8-1. 라이브러리 설치

```python
# Google Colab/Jupyter 셀에서 실행합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0"
```

### 8-2. 모델과 Tokenizer 준비

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face에서 제공하는 원래 GPT-2 checkpoint ID입니다.
MODEL_ID = "openai-community/gpt2"

# GPT-2용 Tokenizer를 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Causal LM Head가 붙은 모델을 불러옵니다.
# AutoModel이 아니라 AutoModelForCausalLM을 사용해야 vocabulary logits를 얻을 수 있습니다.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# 추론 모드로 전환합니다.
model.eval()

# GPU가 있으면 CUDA, 없으면 CPU를 사용합니다.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = model.to(device)

# GPT-2는 기본 pad_token이 없을 수 있습니다.
# 단일 Prompt에서는 padding이 필요 없지만 generate() 경고와 Batch 확장을 고려해
# eos_token을 pad_token으로 재사용하는 예시를 사용합니다.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("device:", device)
print("eos token:", tokenizer.eos_token)
print("eos token id:", tokenizer.eos_token_id)
print("pad token:", tokenizer.pad_token)
print("vocab size:", model.config.vocab_size)
```

### 8-3. Prompt Tokenization

```python
prompt = "Artificial intelligence can"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

# 모델과 입력 Tensor는 같은 device에 있어야 합니다.
inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

print("input keys:", list(inputs.keys()))
print("input_ids shape:", tuple(inputs["input_ids"].shape))
print("attention_mask shape:", tuple(inputs["attention_mask"].shape))

# Token ID를 사람이 읽는 token 조각으로 확인합니다.
tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)
print("tokens:", tokens)
```

## 선택 · 9. 연습 문제: 마지막 위치의 next-token 후보 확인

```python
# 추론에서는 gradient가 필요하지 않습니다.
with torch.inference_mode():
    outputs = model(**inputs)

# 전체 Causal LM logits입니다.
# Shape: [B,L,V]
logits = outputs.logits
print("full logits shape:", tuple(logits.shape))

# 단일 Prompt이며 padding이 없으므로 마지막 위치를 선택합니다.
# Shape: [B,V]
next_token_logits = logits[:, -1, :]
print("next-token logits shape:", tuple(next_token_logits.shape))

# Batch의 첫 번째 샘플에서 vocabulary 전체 확률을 계산합니다.
next_token_probabilities = torch.softmax(
    next_token_logits[0],
    dim=-1,
)

# 확률이 높은 상위 10개 다음 토큰 후보를 확인합니다.
top_probs, top_ids = torch.topk(
    next_token_probabilities,
    k=10,
)

for rank, (token_id, probability) in enumerate(
    zip(top_ids.tolist(), top_probs.tolist()),
    start=1,
):
    # GPT-2 token 문자열에는 단어 앞 공백이 포함될 수 있습니다.
    token_text = tokenizer.decode([token_id])
    print(
        f"{rank:02d}위 | "
        f"token={token_text!r} | "
        f"probability={probability:.4f}"
    )
```

### 코드 해석

| 코드 | 의미 |
| --- | --- |
| `outputs.logits` | 모든 입력 위치의 vocabulary logits `[B,L,V]` |
| `logits[:, -1, :]` | 마지막 위치의 다음 토큰 점수 `[B,V]` |
| `softmax(..., dim=-1)` | vocabulary 축 점수를 확률 분포로 변환 |
| `topk(..., k=10)` | 상위 10개 후보 선택 |
| `tokenizer.decode([id])` | Token ID를 문자열 조각으로 변환 |


## 선택 · 10. 연습 문제: generate()로 텍스트 이어쓰기

```python
# generate()는 다음 토큰 선택과 입력 갱신을 반복하는 고수준 API입니다.
with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        # Prompt 뒤에 새로 생성할 최대 token 수입니다.
        max_new_tokens=30,
        # 가장 점수가 높은 token을 고르는 Greedy Decoding입니다.
        do_sample=False,
        # GPT-2에는 pad token이 기본으로 없으므로 EOS ID를 사용합니다.
        pad_token_id=tokenizer.eos_token_id,
    )

print("input length:", inputs["input_ids"].shape[1])
print("generated total length:", generated_ids.shape[1])

# skip_special_tokens=True로 special token 문자열을 제거합니다.
generated_text = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

print("generated text:")
print(generated_text)
```

### 새로 생성된 부분만 분리하기

```python
# 원래 Prompt token 길이입니다.
prompt_length = inputs["input_ids"].shape[1]

# Prompt 뒤에 추가된 token만 선택합니다.
new_token_ids = generated_ids[0, prompt_length:]

continuation = tokenizer.decode(
    new_token_ids,
    skip_special_tokens=True,
)

print("prompt:", prompt)
print("continuation:", continuation)
```

## 참고 · 13. 이해도 점검

1. GPT의 세 단어는 각각 무엇을 의미하나
2. GPT형 Decoder-only 모델이 원래 Encoder-Decoder Decoder와 다른 점은 무엇인가
3. Causal Mask가 필요한 이유는 무엇인가
4. Causal LM의 입력과 정답은 어떻게 정렬되나
5. 학습과 생성에서 병렬성 차이는 무엇인가
6. next-token 후보는 logits의 어느 위치에서 꺼내지는지
7. `max_new_tokens`는 무엇을 제한하나

### 정답 확인

1. Generative, Pre-trained, Transformer입니다.
2. 별도 Encoder와 Encoder memory를 참고하는 Cross-Attention이 일반적으로 없다.
3. 미래 정답 토큰을 미리 보는 정보 누출을 막기 위해서입니다.
4. 각 입력 위치가 한 칸 앞의 다음 토큰을 정답으로 갖습니다.
5. 학습은 여러 위치의 loss를 병렬로 계산할 수 있지만, 생성은 방금 만든 토큰이 필요하므로 순차적입니다.
6. 현재 입력의 마지막 실제 토큰 위치 logits `[B,V]`를 사용합니다.
7. Prompt를 제외하고 새로 생성할 최대 토큰 수를 제한한다

## 15. 이번 강의 요약

- GPT는 Decoder-only Transformer를 사용해 다음 토큰을 예측하는 생성형 사전학습 모델 계열이다.
- Causal Self-Attention은 각 위치가 미래 토큰을 보지 못하게 한다.
- Causal LM은 입력 토큰과 정답 토큰을 한 칸 이동해 next-token loss를 계산한다
- 학습에서는 여러 위치를 병렬 계산할 수 있지만 생성은 토큰별로 순차 진행된다
- Causal LM logits는 `[B,L,V]`이며 마지막 위치의 `[B,V]` 점수로 다음 토큰 후보를 확인한다
- `generate()`는 토큰 선택과 입력 갱신을 반복한다.
- 다음 강의에서는 BERT와 GPT의 LM Head, logits shape, 입력·출력 의미를 나란히 비교한다.


