---
title: 6-3강 AutoTokenizer와 AutoModel Base Output
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '6-3강 AutoTokenizer와 AutoModel Base Output' 정리
---

## 1. AutoClass는 어떻게 실제 클래스를 찾나

AutoClass 해석 흐름

```
Model ID
-> config.json 확인
-> model_type 식별
-> AutoTokenizer / AutoModel이 실제 클래스 선택
```

예를 들어 DistilBERT Model ID를 `AutoModel.from_pretrained()`에 전달하면 내부적으로 DistilBERT Base Model 클래스가 선택된다


## 2. Base Model과 Task-specific Model



<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_base_model_flow.png
' | relative_url }}" alt="02_base_model_flow.png
" loading="lazy">

Base Model 흐름

| 클래스 | 대표 출력 |
| --- | --- |
| `AutoModel` | `last_hidden_state [B, L, H]` |
| `AutoModelForSequenceClassification` | `logits [B, C]` |
| `AutoModelForMaskedLM` | `logits [B, L, V]` |
| `AutoModelForCausalLM` | `logits [B, L, V]` |

## 3. 실행 환경 준비

```python
# Colab/Jupyter에서 Transformers와 Hub 클라이언트를 설치합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0" "huggingface_hub==1.20.1"
```

```python
# Tensor 연산과 모델 실행을 위해 PyTorch를 불러옵니다.
import torch

# Model ID에 맞는 Tokenizer와 Base Model을 자동 선택하는 클래스를 불러옵니다.
from transformers import AutoModel, AutoTokenizer

# 영어 DistilBERT Base Model ID입니다.
MODEL_ID = "distilbert/distilbert-base-uncased"

# 교육용 기본 revision입니다. 재현이 필요한 운영 환경에서는 승인된 commit hash로 바꾸세요.
REVISION = "main"

# CUDA가 있으면 GPU, 없으면 CPU를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
```

## 4. AutoTokenizer로 Batch 만들기

```python
# 사전학습 때 사용한 tokenization 규칙과 vocabulary를 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
)

# 길이가 다른 두 문장을 하나의 Batch로 처리합니다.
texts = [
    "Transformers build contextual token representations.",
    "Short sentence.",
]

# 문자열을 input_ids와 attention_mask Tensor로 변환합니다.
inputs = tokenizer(
    texts,
    # Batch 안에서 가장 긴 문장에 맞춰 짧은 문장을 padding합니다.
    padding=True,
    # 모델 최대 길이를 넘는 입력은 안전하게 자릅니다.
    truncation=True,
    # PyTorch Tensor를 반환합니다.
    return_tensors="pt",
)

# BatchEncoding에 어떤 입력 key가 만들어졌는지 확인합니다.
print("input keys:", inputs.keys())

# input_ids와 attention_mask의 대표 shape은 모두 [B, L]입니다.
print("input_ids shape:", tuple(inputs["input_ids"].shape))
print("attention_mask shape:", tuple(inputs["attention_mask"].shape))

# 문장별 attention_mask를 출력해 실제 토큰 1과 padding 0을 확인합니다.
for row_index, text in enumerate(texts):
    print("text:", text)
    print("attention_mask:", inputs["attention_mask"][row_index].tolist())
```

## 5. AutoModel forward와 ModelOutput

```python
# Task Head가 없는 Base Model을 불러옵니다.
model = AutoModel.from_pretrained(
    MODEL_ID,
)

# 모델 파라미터를 입력에 사용할 device로 이동합니다.
model = model.to(device)

# 추론이므로 Dropout 등을 평가 모드로 전환합니다.
model.eval()

# BatchEncoding 안의 모든 Tensor를 모델과 같은 device로 이동합니다.
model_inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

# 추론에서는 gradient graph를 만들지 않습니다.
with torch.inference_mode():
    # input_ids와 attention_mask를 모델 forward에 전달합니다.
    outputs = model(**model_inputs)

# ModelOutput이 제공하는 key를 확인합니다.
print("output keys:", outputs.keys())

# 마지막 Transformer layer의 토큰별 표현을 꺼냅니다.
last_hidden_state = outputs.last_hidden_state
print("last_hidden_state shape:", tuple(last_hidden_state.shape))

# Base Model에는 일반적으로 Task Head가 없으므로 logits 속성이 없거나 None입니다.
print("logits:", getattr(outputs, "logits", None))

# DistilBERT는 기본적으로 pooler_output을 제공하지 않습니다.
print("pooler_output:", getattr(outputs, "pooler_output", None))
```

## 6. last_hidden_state 해석


last_hidden_state shape

```
last_hidden_state [B, L, H]
```

- `B`: 문장 수
- `L`: padding 후 토큰 위치 수
- `H`: hidden size, DistilBERT Base는 대표적으로 768

```python
# 첫 번째 문장의 첫 번째 토큰 벡터를 선택합니다.
# Batch index 0, sequence index 0을 고정하므로 hidden 축 [H]만 남습니다.
first_token_vector = last_hidden_state[0, 0, :]

# 벡터 shape은 [H]입니다.
print("first token vector shape:", tuple(first_token_vector.shape))

# 전체 값을 출력하면 길기 때문에 앞의 8개 값만 확인합니다.
print("first 8 values:", first_token_vector[:8].detach().cpu())
```

## 선택 · 7. Pooling 방식

Pooling 방식

문장 벡터를 만드는 방식은 하나가 아닙니다.

- 첫 토큰 사용
- 모델의 pooler output 사용
- 실제 토큰의 평균 사용
- 태스크에 맞춘 별도 pooling layer 학습

### Masked Mean Pooling

```python
# 토큰별 hidden state와 attention mask를 받아 문장별 평균 벡터를 만드는 함수입니다.
def masked_mean_pool(last_hidden_state, attention_mask):
    # attention_mask shape [B, L]에 hidden 축을 추가해 [B, L, 1]로 만듭니다.
    expanded_mask = attention_mask.unsqueeze(-1)

    # hidden state와 곱할 수 있도록 float dtype으로 변환합니다.
    # 실제 토큰 위치는 1, padding 위치는 0이므로 padding 벡터가 제거됩니다.
    expanded_mask = expanded_mask.to(last_hidden_state.dtype)

    # 실제 토큰 위치의 hidden vector만 남기고 sequence 축으로 합합니다.
    summed_hidden = (last_hidden_state * expanded_mask).sum(dim=1)

    # 문장별 실제 토큰 수를 계산합니다.
    token_counts = expanded_mask.sum(dim=1)

    # 0으로 나누는 상황을 막기 위해 최소값을 작은 양수로 제한합니다.
    token_counts = token_counts.clamp(min=1e-9)

    # 실제 토큰 hidden 합을 실제 토큰 수로 나눠 문장 벡터 [B, H]를 만듭니다.
    return summed_hidden / token_counts

# 모델 입력에 사용한 attention_mask를 이용해 padding을 제외한 문장 벡터를 계산합니다.
sentence_vectors = masked_mean_pool(
    last_hidden_state=last_hidden_state,
    attention_mask=model_inputs["attention_mask"],
)

print("sentence vectors shape:", tuple(sentence_vectors.shape))
```

## 선택 · 8. 연습 문제

### 연습 문제 1: 고정 shape 관찰

```python
# 정답을 명확히 비교하기 위해 모든 문장을 길이 12로 맞춥니다.
exercise_inputs = tokenizer(
    [
        "The model returns token representations.",
        "Pooling creates one vector per sentence.",
    ],
    padding="max_length",
    truncation=True,
    max_length=12,
    return_tensors="pt",
)

# 입력 Tensor를 model과 같은 device로 이동합니다.
exercise_inputs = {
    name: tensor.to(device)
    for name, tensor in exercise_inputs.items()
}

# Base Model forward를 실행합니다.
with torch.inference_mode():
    exercise_outputs = model(**exercise_inputs)

# hidden state와 pooling 결과를 확인합니다.
exercise_hidden = exercise_outputs.last_hidden_state
exercise_sentence_vectors = masked_mean_pool(
    exercise_hidden,
    exercise_inputs["attention_mask"],
)

print("input_ids:", tuple(exercise_inputs["input_ids"].shape))
print("last_hidden_state:", tuple(exercise_hidden.shape))
print("sentence_vectors:", tuple(exercise_sentence_vectors.shape))
print("logits:", getattr(exercise_outputs, "logits", None))
```

## 이해도 점검

1. AutoClass가 실제 클래스를 선택할 때 무엇을 읽나
2. Base Model의 대표 출력은 무엇인가
3. `last_hidden_state`의 각 축을 설명
4. masked mean pooling이 필요한 이유는 무엇인가

### 정답 확인

1. Model ID 저장소의 config와 metadata를 읽는다
2. `last_hidden_state`다.
3. Batch, sequence length, hidden size다.
4. padding 위치를 문장 평균에서 제외하기 위해서

## 12. AutoConfig로 모델 구조 먼저 읽기

가중치를 올리기 전에 config만 확인하면 메모리 부담 없이 구조와 입력 계약을 점검할 수 있다.

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION)
fields = [
    "model_type",
    "architectures",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
    "vocab_size",
]
for name in fields:
    print(name, getattr(config, name, None))
```

모델마다 필드 이름이나 존재 여부가 다르므로 `getattr(..., None)`으로 확인한다.

`trust_remote_code=True`가 필요한 모델은 외부 Python 코드를 실행할 수 있으므로, Private 환경에서는 revision을 고정하고 코드를 검토한 뒤 허용한다

## 13. 이번 강의 요약

- AutoClass는 Model ID와 config를 바탕으로 실제 클래스를 선택한다
- Base `AutoModel`은 Task Head 없이 토큰별 문맥 표현을 반환한다
- `last_hidden_state` shape은 `[B, L, H]`다.
- `pooler_output`은 모델마다 없을 수 있으며 pooling 방식은 태스크에 따라 선택한다

