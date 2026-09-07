---
title: 5-2강 BERT_Encoder-only와 Masked LM
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '5-2강 BERT_Encoder-only와 Masked LM' 정리
---

## 1. BERT란 무엇인가

BERT는 **Bidirectional Encoder Representations from Transformers**의 약자다.

| 단어 | 의미 |
| --- | --- |
| Bidirectional | 왼쪽과 오른쪽 문맥을 함께 사용합니다. |
| Encoder | Transformer Encoder-only 구조를 사용합니다. |
| Representations | 입력을 문맥이 반영된 벡터 표현으로 바꿉니다. |
| Transformers | Self-Attention 기반 Transformer Block을 사용합니다. |

**BERT는 Transformer Encoder를 여러 층 쌓아, 입력의 각 토큰을 문맥이 반영된 표현으로 만드는 사전학습 모델이다.**


### BERT가 직접 반환하는 대표 표현

Base model 기준 대표 출력은 다음과 같다.

```
last_hidden_state: [B, L, D]
```

- `B`: Batch size
- `L`: Sequence length
- `D`: Hidden dimension

이 표현 위에 문제별 Head를 붙이면 분류, 토큰 분류, 질의응답, Masked LM 등으로 확장할 수 있다.

BERT 본체는 문장을 이해해서 [B,L,D] 형태의 문맥 벡터를 만든다. 그 뒤에 분류, 토큰 분류, 질의응답, MLM 등 각 문제에 맞는 출력층(Task Head)을 추가해서 실제 문제를 풀 수 있게 만든다.

Head를 붙인다는 것은 BERT의 문맥 벡터를 우리가 원하는 정답 형태로 바꿔주는 출력층을 추가한다는 뜻입니다.

## 2. Encoder-only 구조와 Bidirectional Context

### 2-1. Bidirectional은 “문장을 거꾸로도 읽는다”는 뜻이 아니다

Bidirectional Context는 각 토큰의 표현을 만들 때 **왼쪽 토큰과 오른쪽 토큰을 함께 참고할 수 있다**는 뜻이다.

```
Paris is the [MASK] of France.
```

`[MASK]`를 예측할 때 BERT는 다음 정보를 함께 봅니다.

- 왼쪽: `Paris is the`
- 오른쪽: `of France`

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_bidirectional_context.png
' | relative_url }}" alt="02_bidirectional_context.png
" loading="lazy">

양쪽 문맥을 함께 사용하는 BERT

BERT가 `capital`을 높은 후보로 평가하는 데에는 `Paris`와 `France`라는 양쪽 단서가 모두 도움이 된다


### 2-2. 왜 일반적인 다음 토큰 예측을 그대로 사용하지 않았을까

일반 Causal LM은 각 위치에서 미래 토큰을 볼 수 없다.<br> 하지만 BERT는 입력 전체를 이해하는 표현을 만들고자 했다.

문제는 정답 토큰을 입력에 그대로 보여 주면 모델이 정답을 복사할 수 있다

```
입력에 capital이 그대로 있음
-> capital 위치에서 capital을 맞히는 문제는 너무 쉽다.
```

그래서 일부 위치를 가려 원래 토큰을 맞히도록 설계한 것이 Masked Language Modeling다.

## 3. Masked Language Modeling

Masked Language Modeling은 입력 토큰 일부를 변경하고 **선택된 위치의 원래 토큰**을 예측하는 문제다

```
원문: The movie was surprisingly good.
입력: The movie was surprisingly [MASK].
정답: good
```

### 3-1. 학습 단계

```
원문 선택
-> 예측할 위치 선택
-> 입력 일부 변경
-> BERT Encoder 통과
-> 선택 위치의 vocabulary logits 계산
-> 원래 token ID와 loss 계산
```

### 3-2. Masked LM의 출력

Masked LM Head는 각 토큰 위치마다 vocabulary 전체에 대한 점수를 만든다

```
hidden state: [B, L, D]
LM Head
logits:       [B, L, V]
```

그러나 일반적으로 MLM loss를 계산할 때는 **선택된 위치만 정답 계산에 사용**하다.

> BERT가 `[MASK]` 문자열을 이해하도록 만드는 것이 최종 목적이 아닙니다. `[MASK]` 위치를 맞히는 과정에서 문장 전체의 문맥 관계를 학습하도록 만드는 것이 목적이다.


BERT는 모든 토큰 위치에 대해 [B,L,V] 점수를 출력하지만, MLM 학습에서는 미리 선택한 일부 토큰 위치의 예측만 정답과 비교해서 loss를 계산한다. 

## 4. 원래 BERT의 Masking 방식

원래 BERT 논문에서는 전체 토큰 중 약 15%를 학습 대상 위치로 선택하고, 선택된 위치를 다음처럼 처리한다


<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_mlm_training_recipe.png.png
' | relative_url }}" alt="03_mlm_training_recipe.png.png
" loading="lazy">

원래 BERT MLM 데이터 만들기

| 선택된 위치 처리 | 비율 | 이유 |
| --- | --- | --- |
| `[MASK]`로 교체 | 80% | 가려진 토큰을 주변 문맥으로 예측하게 합니다. |
| 임의의 토큰으로 교체 | 10% | 항상 `[MASK]`만 찾는 전략을 줄입니다. |
| 원래 토큰 유지 | 10% | 실제 입력에는 `[MASK]`가 없다는 차이를 완화합니다. |

예시:

```
원래 토큰: good
80% 경우: [MASK]
10% 경우: random token
10% 경우: good 그대로 유지
정답은 세 경우 모두 good
```

> 위 비율은 원래 BERT 논문의 방식입니다. RoBERTa, ModernBERT 등 후속 모델은 Masking 방식이나 추가 Objective가 다를 수 있으므로 Model card와 논문을 확인해야 한다

### 참고 · 4-1. Next Sentence Prediction은 무엇인가

원래 BERT는 MLM(Masked Language Modeling) 외에 Next Sentence Prediction(NSP)도 함께 사용했습니다. 두 문장이 실제로 이어지는지 구분하는 문제이다

이 장에서는 BERT와 GPT의 핵심 비교를 위해 MLM에 집중한다<br>후속 모델 중에는 NSP(Next Sentence Prediction)를 제거하거나 다른 문장 관계 Objective로 대체한 경우도 있습니다.

## 5. BERT Special Token

### `[CLS]`

- 입력 맨 앞에 추가된다.
- BERT의 첫 위치 문맥 표현이다.
- 문장 분류에서 대표 문장 표현으로 자주 활용된다.
- `[CLS]` 자체가 자동으로 좋은 분류 결과를 만드는 것은 아니며, 보통 Fine-tuning과 Classification Head가 필요하다.

### `[SEP]`

- 문장 끝을 표시한다.
- 문장 쌍 입력에서는 두 문장의 경계를 표시한다.

```
[CLS] sentence A [SEP] sentence B [SEP]
```

### `[MASK]`

- Masked LM에서 예측할 위치를 표시한다.
- 일반 BERT 추론 입력에 항상 넣는 토큰은 아니다.

### `[PAD]`

- Batch 안에서 서로 다른 문장 길이를 맞추기 위해 추가된다.
- `attention_mask`로 실제 토큰과 구분한다.


> Special Token 문자열을 직접 외우기보다 `tokenizer.cls_token`, `tokenizer.sep_token`, `tokenizer.mask_token`, `tokenizer.pad_token`을 확인해야한다.
<br>
모델 계열마다 토큰 문자열과 ID가 다를 수 있습니다.

## 6. BERT 입력 표현의 세 구성요소

원래 BERT 입력 표현은 대표적으로 다음 세 정보를 합쳐 만든다

```
Token Embedding
+ Position Embedding
+ Token Type Embedding
```

### 6-1. Token Embedding

각 Token ID를 벡터로 바꾼다.



### 6-2. Position Embedding

토큰의 순서 위치를 알려 준다

### 6-3. Token Type Embedding

문장 쌍을 넣을 때 첫 번째 문장과 두 번째 문장을 구분한다

```
문장 A, token_type_id = 0
문장 B, token_type_id = 1
```

단일 문장 입력에서는 보통 모든 위치가 0이다

> 모든 Encoder-only 모델이 `token_type_ids`를 사용하지는 않는다.<br>
>예를 들어 RoBERTa 계열은 일반적으로 BERT와 같은 Segment Embedding을 사용하지 않는다.<br>
>AutoTokenizer가 반환하는 key와 모델 forward signature를 확인해야한다.

## 7. Masked LM Head와 출력

BERT Encoder 본체는 `[B,L,D]` hidden state를 만든다<br>
빈칸 채우기를 하려면 각 위치를 vocabulary 점수로 바꾸는 Masked LM Head가 필요하다

```
BERT hidden state [B,L,D]
-> Masked LM Head
-> logits [B,L,V]
```

- `D`: BERT hidden dimension
- `V`: Tokenizer vocabulary size



```
logits[b, l, v]

b번째 샘플의
l번째 토큰 위치에서
v번째 vocabulary token에 대한 정규화 전 점수
```

빈칸 채우기에서는 `[MASK]` 위치의 `V`개 점수만 꺼내 top-k 후보를 확인한다

## 선택 · 8. 연습 문제: BERT 입력 형식 확인

### 8-1. 라이브러리 설치

```python
# Google Colab/Jupyter 셀에서 실행하는 셸 명령입니다.
# 아래 명령은 이 교안에서 검증한 라이브러리 버전을 설치합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0"
```

### 8-2. Tokenizer와 모델 불러오기

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 원래 BERT 계열의 영어 uncased checkpoint입니다.
# 이 모델은 영어 문장을 대상으로 학습되었으므로 한국어 입력 평가에는 적합하지 않습니다.
MODEL_ID = "google-bert/bert-base-uncased"

# Tokenizer는 문자열을 BERT 입력 형식으로 바꿉니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Masked LM Head가 포함된 BERT 모델을 불러옵니다.
# Base AutoModel이 아니라 AutoModelForMaskedLM을 사용해야 vocab logits를 얻을 수 있습니다.
model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

# 추론에서는 Dropout을 평가 모드로 전환합니다.
model.eval()

print("mask token:", tokenizer.mask_token)
print("mask token id:", tokenizer.mask_token_id)
print("cls token:", tokenizer.cls_token)
print("sep token:", tokenizer.sep_token)
print("pad token:", tokenizer.pad_token)
```

### 8-3. 단일 문장 Tokenization

```python
# mask token을 문자열로 직접 하드코딩하기보다 tokenizer에서 읽습니다.
text = f"Paris is the {tokenizer.mask_token} of France."

# return_tensors="pt"를 지정하면 Python list가 아니라 PyTorch Tensor를 반환합니다.
inputs = tokenizer(
    text,
    return_tensors="pt",
)

# BatchEncoding에 어떤 입력 key가 들어 있는지 확인합니다.
print("input keys:", list(inputs.keys()))

# BERT Tokenizer의 대표 출력은 input_ids, token_type_ids, attention_mask입니다.
for name, tensor in inputs.items():
    print(
        name,
        "shape=", tuple(tensor.shape),
        "dtype=", tensor.dtype,
    )

# 정수 ID를 사람이 읽는 token 문자열로 변환합니다.
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("tokens:", tokens)
```

예상 shape:

```
input_ids:      [B,L]
token_type_ids: [B,L]
attention_mask: [B,L]
```


### 8-4. 문장 쌍 입력

```python
sentence_a = "Transformers process token sequences."
sentence_b = "BERT uses encoder blocks."

pair_inputs = tokenizer(
    sentence_a,
    sentence_b,
    return_tensors="pt",
)

pair_tokens = tokenizer.convert_ids_to_tokens(
    pair_inputs["input_ids"][0]
)

print("tokens:", pair_tokens)
print("token_type_ids:", pair_inputs["token_type_ids"][0].tolist())

# 일반적으로 첫 문장과 첫 [SEP]까지 0,
# 두 번째 문장과 마지막 [SEP]에는 1이 들어갑니다.
```

출력

```
tokens: ['[CLS]', 'transformers', 'process', 'token', 'sequences', '.', '[SEP]', 'bert', 'uses', 'en', '##code', '##r', 'blocks', '.', '[SEP]']
token_type_ids: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
```


## 선택 · 9. 연습 문제: Mask 위치의 logits 확인

```python
# model과 inputs는 앞 셀에서 만든 객체를 사용합니다.

# 추론에서는 gradient graph가 필요하지 않으므로 inference_mode를 사용합니다.
with torch.inference_mode():
    outputs = model(**inputs)

# Masked LM 출력의 대표 logits shape은 [B,L,V]입니다.
logits = outputs.logits
print("logits shape:", tuple(logits.shape))
print("vocab size:", model.config.vocab_size)

# input_ids에서 mask_token_id와 같은 위치를 찾습니다.
# 결과는 [찾은 개수, 2] 형태이며 각 행은 [batch_index, token_index]입니다.
mask_positions = (
    inputs["input_ids"] == tokenizer.mask_token_id
).nonzero(as_tuple=False)

print("mask positions:", mask_positions.tolist())

# 이 예제에는 mask가 정확히 하나 있다고 가정합니다.
assert mask_positions.shape[0] == 1

batch_index, token_index = mask_positions[0].tolist()

# mask 위치에서 vocabulary 전체 점수 [V]를 꺼냅니다.
mask_logits = logits[batch_index, token_index]
print("mask logits shape:", tuple(mask_logits.shape))

# 아직 확률이 아니라 logits이므로 softmax를 적용합니다.
mask_probabilities = torch.softmax(mask_logits, dim=-1)

# 확률이 높은 상위 5개 token ID와 확률을 꺼냅니다.
top_probs, top_ids = torch.topk(mask_probabilities, k=5)

for rank, (token_id, probability) in enumerate(
    zip(top_ids.tolist(), top_probs.tolist()),
    start=1,
):
    token_string = tokenizer.decode([token_id]).strip()
    print(
        f"{rank}위 | token={token_string!r} | "
        f"probability={probability:.4f}"
    )
```

- `outputs.logits` 전체 shape은 `[B,L,V]`입니다.
- 빈칸 후보는 `[MASK]` 위치의 `[V]` 벡터에서 찾는다
- `logits`는 확률이 아니므로 해석할 때 softmax가 필요하다.
- 후보 점수는 모델 내부 분포일 뿐 사실성 보장이 아니다



## 선택 · 10. BERT가 이해 태스크에 자연스러운 이유

BERT는 각 토큰 표현을 만들 때 입력 전체를 참고한다. 그래서 다음과 같은 문제에 자연스럽게 연결된다

### 문장 분류

```
BERT hidden [B,L,D]
-> [CLS] 또는 pooling
-> Classification Head
-> logits [B,C]
```

### 토큰 분류

```
BERT hidden [B,L,D]
-> 각 토큰에 Linear Head
-> logits [B,L,C]
```

### 검색·유사도

```
문장 표현 벡터
-> 질의와 문서 벡터 비교
```

## 참고 · 12. 이해도 점검

1. BERT의 B는 어떤 문맥 사용 방식을 의미하나
2. Masked LM에서 정답은 무엇인가
3. `[MASK]`를 사용하는 목적은 무엇인가
4. `[CLS]`, `[SEP]`, `[PAD]`의 역할을 각각 설명
5. `token_type_ids`는 어떤 상황에서 사용되나
6. Masked LM logits `[B,L,V]`에서 fill-mask 후보는 어느 위치를 확인해야 하나


### 정답 확인

1. 각 토큰이 왼쪽과 오른쪽 문맥을 함께 참고할 수 있는 Bidirectional Context를 의미
2. 학습 대상으로 선택된 위치의 원래 token ID입니다.
3. 정답 토큰을 그대로 보여 주지 않고 주변 문맥을 통해 원래 토큰을 예측하게 하기 위해서다.
4. `[CLS]`는 입력 시작 및 대표 위치, `[SEP]`는 문장 끝·문장 쌍 경계, `[PAD]`는 배치 길이 맞춤입니다.
5. BERT에서 문장 쌍의 첫 번째 문장과 두 번째 문장을 구분할 때 사용된다
6. `input_ids`가 `mask_token_id`인 위치의 vocabulary logits를 확인한다