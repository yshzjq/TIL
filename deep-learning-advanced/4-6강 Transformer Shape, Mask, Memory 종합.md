---
title: 4-6강 Transformer Shape, Mask, Memory 종합(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-6강 Transformer Shape, Mask, Memory 종합' 정리
---

## 1. End-to-End Shape 지도

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_end_to_end_shapes.png
' | relative_url }}" alt="01_end_to_end_shapes.png
" loading="lazy">

End-to-End Shape

| 단계 | 대표 값 | Shape |
| --- | --- | --- |
| 원본 입력 | 문자열 | Python string/list |
| Tokenizer | `input_ids` | `[B,L]` |
| Tokenizer | `attention_mask` | `[B,L]` |
| Embedding | hidden | `[B,L,D]` |
| Transformer Block | hidden | `[B,L,D]` |
| 문장 분류 Head | logits | `[B,C]` |
| 토큰 분류 Head | logits | `[B,L,C]` |
| Causal LM Head | logits | `[B,L,V]` |

### Shape 조건

1. Batch 축 `B`는 유지되는지
2. Sequence 축 `L`은 유지되나요, pooling으로 사라지는지
3. 마지막 축이 hidden `D`인가요, class `C`인가요, vocabulary `V`인지
4. Head 축 `H`가 중간 Tensor에만 존재하는지

## 2. Hidden state와 logits 다시 구분


```
Base Transformer 출력: [B,L,D]
Task Head 출력:         문제에 따라 [B,C], [B,L,C], [B,L,V]
```

Base Model을 불러왔는데 클래스 점수가 보이지 않는다면 오류가 아닐 수 있다.  

Base Transformer
= 문맥이 반영된 벡터를 만든다

Task Head
= 그 벡터를 실제 문제의 점수로 바꾼다


`AutoModel`은 보통 task-specific head 없이 hidden representation을 반환한다

model = AutoModel.from_pretrained(...)
<br>
이라고 하면 보통 Transformer 본체만 가져온다
```
입력
↓
AutoModel
↓
Hidden State [B, L, D]
```
출력

```
긍정: 3.2
부정: -1.1
```
같은 클래스 점수가 안 보여도 이상한 게 아니다<br>
왜냐하면 아직 분류하는 부분이 안 붙어 있기 때문에 오류가 아닐 수 있다.


`AutoModelForSequenceClassification`, `AutoModelForCausalLM` 같은 클래스에는 특정 task head가 붙는다

AutoModelForSequenceClassification은 Transformer 본체 뒤에 분류용 Head가 붙어 있다

```
입력
↓
Transformer
↓
Hidden State
↓
Classification Head
↓
Logits [B, C]
```

예를 들어 긍정/부정 2개 클래스라면
```
C = 2

출력

[3.2, -1.1]

```
처럼 나올 수 있다

이 숫자는 클래스 점수(logits)

AutoModel 은 "문장을 벡터로 잘 표현해줄게" 이고

AutoModelForSequenceClassification 은 문장을 벡터로 표현하고, 그걸 이용해서 클래스 점수까지 만들어준다

AutoModelForCausalLM은 GPT처럼 다음 토큰을 예측하는 모델

AutoModel → hidden state까지만
AutoModelFor... → hidden state를 task에 맞는 logits까지 변환

## 3. Padding Mask와 Causal Mask 종합

Mask 종류

| 구분 | Padding Mask | Causal Mask |
| --- | --- | --- |
| 목적 | `[PAD]` 위치 무시 | 미래 토큰 차단 |
| 대표 shape | `[B,L]` | `[L,L]` 또는 broadcast 형태 |
| 샘플마다 다름 | 보통 예 | 보통 같은 길이에서 공통 |
| 주로 사용 | Encoder/Decoder 모두 가능 | 생성용 Decoder Self-Attention |

> Hugging Face의 `attention_mask`는 흔히 1=사용, 0=무시다. <br>PyTorch bool mask는 흔히 True=차단입니다. API별 의미를 반드시 확인해야한다

## 4. Sequence Length와 L×L 증가

한 head의 Self-Attention은 모든 Query와 모든 Key 조합을 계산한다
```
Query 수 L × Key 수 L = L² score cell
```

| L | L² |
| --- | --- |
| 128 | 16,384 |
| 256 | 65,536 |
| 512 | 262,144 |
| 1,024 | 1,048,576 |

L이 2배가 되면 score cell은 4배가 됩니다. Batch와 head를 포함한 대표 Score Tensor의 원소 수는 다음과 같다.

```
B × H × L × L
```
 긴 문맥에서 Standard Self-Attention의 메모리와 계산 비용이 빠르게 증가하는 핵심 이유

### 4-1. 학습은 병렬화할 수 있는데 생성은 왜 순차적인가?

Decoder-only Causal LM의 학습에서는 정답 문장 전체를 이미 알고 있다. 

정답을 한 칸 이동한 입력과 Causal Mask를 함께 사용하면, 각 위치가 미래 토큰을 보지 못하게 하면서도 여러 위치의 logits를 한 번의 Forward Pass에서 계산할 수 있다.

```
학습 입력: <BOS> 나는 AI를 배운다
학습 정답: 나는  AI를 배운다 <EOS>
```

위 예시에서 각 위치의 정답은 준비되어 있으므로 GPU는 여러 위치의 다음 토큰 예측을 병렬로 계산할 수 있다. 

이를 흔히 Teacher Forcing 기반 학습 흐름으로 설명한다

반면 생성 시점에는 다음 토큰이 아직 존재하지 않는다

```
Prompt
-> 첫 번째 새 토큰 생성
-> 그 토큰을 입력에 추가
-> 두 번째 새 토큰 생성
-> 반복
```

두 번째 새 토큰을 만들려면 첫 번째 새 토큰이 먼저 결정되어야 한다. 

따라서 Transformer가 RNN보다 학습 병렬화에 유리하더라도 **Autoregressive 생성의 토큰 간 의존성 자체는 순차적**이다.

### 4-2. Prefill과 Decode

LLM 추론은 크게 Prefill과 Decode 두 단계로 나눌 수 있다.

| 단계 | 입력 | 주요 동작 | 특징 |
| --- | --- | --- | --- |
| Prefill | 전체 Prompt | Prompt 토큰들의 hidden state와 각 Layer의 K·V 계산 | 여러 Prompt 위치를 병렬 처리할 수 있음 |
| Decode | 직전에 생성한 새 토큰 | 다음 토큰 logits 계산 후 토큰 하나 선택 | 한 Token Step씩 순차 반복 |

Prefill에서는 길이 `L_prompt`인 입력 전체를 처리한다

Standard Self-Attention이라면 Prompt 내부의 Query-Key 조합 때문에 길이가 길수록 Attention 계산량이 빠르게 증가한다

Decode에서는 매 Step마다 새 토큰 하나의 Query를 계산하고, 지금까지의 Prompt와 생성 토큰 전체를 참고해 다음 토큰을 만든다

### 4-3. KV Cache는 무엇을 저장하나

KV Cache는 이전 Token Step에서 각 Transformer Layer가 계산한 **Key와 Value Tensor**를 저장한다


```
저장하는 값: 과거 토큰들의 K, V
저장하지 않는 값: 과거 전체 Attention Score 행렬, 최종 확률 전체, 생성된 모든 중간 연산
```


Cache가 없다면 새 토큰을 하나 생성할 때마다 과거의 모든 토큰을 다시 모델에 넣고 각 Layer의 K와 V를 반복 계산해야 한다

Cache가 있으면 과거 K·V는 재사용하고, 새 토큰의 Q·K·V만 추가로 계산할 수 있다.

```
과거 Cache
K_cache: [B,H_kv,L_past,D_head]
V_cache: [B,H_kv,L_past,D_head]

새 토큰의 K, V: [B,H_kv,1,D_head]
-> Sequence 축에 이어 붙여 Cache 갱신
```

새 토큰의 Query는 갱신된 K Cache 전체와 비교한다

따라서 KV Cache는 과거 Projection의 반복 계산을 줄이지만, 새 Query가 긴 과거 Context를 참고하는 Attention 연산까지 없애는 것은 아니다. 
<br>
또한 다음 토큰이 이전 토큰에 의존한다는 순차성도 제거하지 않는다

### 4-4. KV Cache 메모리는 무엇에 비례하나

```
KV Cache bytes
≈ 2
× num_layers
× batch_size
× cached_length
× num_kv_heads
× D_head
× bytes_per_element
```

- 앞의 `2`는 Key와 Value 두 Tensor를 뜻한다
- `cached_length`는 Prompt 토큰과 지금까지 생성한 토큰 수에 따라 증가한다
- FP16/BF16은 일반적으로 원소당 2 byte, FP32는 4 byte를 사용한다
- 일반 Multi-Head Attention에서는 `num_kv_heads`가 Attention Head 수와 같을 수 있다.
- GQA나 MQA는 여러 Query Head가 더 적은 수의 K·V Head를 공유하여 KV Cache 크기를 줄일 수 있다.

실제 메모리 사용량은 모델 구현, Tensor Layout, Padding, Beam Search, Batch 구성, Cache dtype과 Serving Engine에 따라 달라질 수 있으므로 위 식은 구조를 이해하기 위한 근사식이다



### 4-5. Context Length가 길어질 때 달라지는 것

| 항목 | Context가 길어질 때의 변화 |
| --- | --- |
| Prefill Attention | Standard Attention의 Query-Key 조합이 많아져 계산량이 크게 증가한다. |
| Attention Score Tensor | 명시적으로 저장하는 구현에서는 대표적으로 `L × L` 크기가 됩니다. |
| KV Cache | 저장할 과거 K·V가 늘어 Context 길이에 대체로 선형으로 증가합니다. |
| Decode 한 Step | 새 Query가 더 긴 Cache를 참고하므로 일반적으로 처리해야 할 과거 위치가 늘어난다 |

FlashAttention 같은 최적화는 중간 Attention Matrix를 메모리에 전부 저장하지 않는 방식으로 메모리 접근을 줄일 수 있지만, <br>
Standard Full Attention이 토큰 쌍의 관계를 계산한다는 구조적 특성까지 사라지는 것은 아니다

## 5. 연습 문제: 사전학습 모델 Shape 추적


### 5-1. 라이브러리 설치

```python
# Google Colab 또는 Jupyter Notebook 셀에서 실행합니다.
# 아래 명령은 이 교안에서 검증한 라이브러리 버전을 설치합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0"
```
### 5-2. Tokenizer와 Base Model 불러오기

```python
import torch
from transformers import AutoModel, AutoTokenizer

# 한국어를 포함한 여러 언어를 처리하는 DistilBERT checkpoint입니다.
MODEL_ID = "distilbert/distilbert-base-multilingual-cased"

# 모델과 같은 checkpoint 계열의 Tokenizer를 불러옵니다.
# Tokenizer의 vocabulary와 special token 규칙이 모델과 일치해야 합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# AutoModel은 분류 Head가 없는 Base Transformer를 불러옵니다.
# 따라서 대표 출력은 class logits가 아니라 last_hidden_state입니다.
model = AutoModel.from_pretrained(MODEL_ID)

# 추론 관찰이므로 Dropout을 평가 모드로 전환합니다.
model.eval()
```


### 5-3. 서로 다른 길이의 문장을 Batch로 만들기

```python
texts = [
    "Transformer는 토큰 관계를 문맥화합니다.",
    "짧은 문장입니다.",
]

# 길이가 다른 문장을 같은 Batch Tensor로 묶습니다.
inputs = tokenizer(
    texts,
    padding=True,        # 짧은 문장을 batch 최장 길이에 맞춰 padding합니다.
    truncation=True,     # 모델 최대 길이를 넘으면 자릅니다.
    return_tensors="pt", # Python list가 아니라 PyTorch Tensor를 반환합니다.
)

print("keys:", inputs.keys())
print("input_ids:", inputs["input_ids"].shape)
print("attention_mask:", inputs["attention_mask"].shape)
print("attention_mask values:\n", inputs["attention_mask"])
```

### 5-4. Base Model 출력 확인

```python
# 추론에서는 gradient가 필요 없으므로 autograd 기록을 끕니다.
with torch.no_grad():
    # **inputs는 input_ids와 attention_mask를 keyword argument로 펼칩니다.
    outputs = model(**inputs)

# Base Model의 각 토큰 위치별 문맥 표현입니다.
# Shape: [B,L,D]
last_hidden_state = outputs.last_hidden_state

print("last_hidden_state:", last_hidden_state.shape)
print("hidden size from config:", model.config.dim)

# Batch와 Sequence 축은 Tokenizer 출력과 일치해야 합니다.
assert last_hidden_state.shape[0] == inputs["input_ids"].shape[0]
assert last_hidden_state.shape[1] == inputs["input_ids"].shape[1]
```

### 5-5. Masked Mean Pooling으로 문장 표현 만들기

```python
# Hugging Face attention_mask는 일반적으로 1=실제 토큰, 0=padding입니다.
# Hidden state와 곱할 수 있도록 마지막 축을 추가하고 dtype을 맞춥니다.
valid_mask = inputs["attention_mask"].unsqueeze(-1).to(
    last_hidden_state.dtype
)

# Padding 위치의 hidden state를 0으로 만듭니다.
masked_hidden = last_hidden_state * valid_mask

# 문장별 실제 토큰 수를 계산합니다.
valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)

# 실제 토큰 위치만 평균내 [B,D] 문장 표현을 만듭니다.
sentence_embeddings = masked_hidden.sum(dim=1) / valid_counts

print("sentence_embeddings:", sentence_embeddings.shape)
assert sentence_embeddings.ndim == 2
```

## 6. 연습 문제: Attention Score Memory 근사

Memory 손잡이

아래 함수는 Attention Score Tensor 하나의 이론적 원소 수와 bytes만 계산한다

```python

def estimate_attention_score_memory(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    bytes_per_value: int,
) -> tuple[int, float]:
    """[B,H,L,L] Score Tensor 하나의 원소 수와 MiB를 계산합니다."""

    # 모든 Batch, Head, Query, Key 조합의 원소 수입니다.
    num_elements = (
        batch_size
        * num_heads
        * sequence_length
        * sequence_length
    )

    # 원소 수에 dtype당 bytes를 곱해 전체 bytes를 구합니다.
    total_bytes = num_elements * bytes_per_value

    # 1 MiB = 1024 * 1024 bytes입니다.
    total_mib = total_bytes / (1024 ** 2)

    return num_elements, total_mib

# 예: B=2, H=8, L=512, float32=4 bytes
num_elements, memory_mib = estimate_attention_score_memory(
    batch_size=2,
    num_heads=8,
    sequence_length=512,
    bytes_per_value=4,
)

print("원소 수:", f"{num_elements:,}")
print("Score Tensor 하나의 근사 크기(MiB):", f"{memory_mib:.2f}")
```

### 예상 결과

```
원소 수: 4,194,304
Score Tensor 하나의 근사 크기(MiB): 16.00
```

### Length를 2배로 바꿔보세요

`L=1024`로 바꾸면 원소 수와 메모리가 4배가 되는지 확인한다

## 참고 · 7. 실제 GPU Memory가 더 큰 이유

앞의 계산은 Attention Score Tensor 하나만 계산했습니다. 실제 학습 GPU 메모리에는 다음이 포함된다


- 모델 파라미터
- Q/K/V와 FFN activation
- 여러 Layer의 중간 Tensor
- Gradient
- Optimizer state
- Temporary workspace
- KV cache(생성/서빙 상황)


따라서 16 MiB라고 계산되었다고 해서 전체 모델이 16 MiB만 사용한다는 뜻은 아니다



## 참고 · 9. 이해도 점검

1. `input_ids`와 hidden state의 대표 shape은 각각 무엇인가
2. `[B,L,D]`와 `[B,L,V]`의 마지막 축은 어떻게 다른가
3. Padding Mask와 Causal Mask의 목적 차이는 무엇인가
4. Sequence length가 2배가 되면 L² score cell 수는 몇 배가 되나
5. Attention Score Tensor 근사값이 전체 GPU memory와 같지 않은 이유는 무엇인가

### 정답 확인

1. `input_ids`는 `[B,L]`, hidden state는 `[B,L,D]`다.
2. D는 모델 내부 hidden dimension, V는 vocabulary 전체 후보 수다.
3. Padding Mask는 PAD 위치를 무시하고, Causal Mask는 미래 토큰을 차단한다
4. 4배가 됩니다.
5. 실제 메모리에는 파라미터, activation, gradient, optimizer state 등 많은 항목이 추가되기 때문이다.


## 10. KV Cache Shape과 추론 지표 (선택 학습)

## 요약

- 선택 학습에서는 Prefill·Decode와 KV Cache를 구분하고 TTFT·TPOT·throughput 및 Cache 메모리를 함께 살펴본다.
- Transformer Shape 추적의 핵심은 마지막 축이 `D`, `C`, `V` 중 무엇인지 구분하는 것이다.
- Padding Mask와 Causal Mask는 목적, shape, 값 의미가 다르다.
- Standard Self-Attention Score는 대표적으로 `[B,H,L,L]` 형태다.
- Sequence length가 2배가 되면 L² score cell은 4배로 증가힌디.
- 실제 디버깅에서는 shape, dtype, device, mask 의미를 먼저 기록힌한다.
- 다음 장에서는 BERT와 GPT의 구조와 학습 목적을 비교한다.