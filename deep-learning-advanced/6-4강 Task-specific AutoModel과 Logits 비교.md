---
title: 6-4강 Task-specific AutoModel과 Logits 비교
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '6-4강 Task-specific AutoModel과 Logits 비교' 정리
---

## 1. Base Model + Task Head

```
Base Model hidden representation
→ Classification Head [ B , C ] / Masked LM Head [ B , L , V ]/ Causal LM Head [ B , L , V]
→ Task-specific logits
```

## 2. 분류 logits `[B, C]`

Classification logits

- `B`: 문장 수
- `C`: 클래스 수
- 각 행은 한 문장의 클래스별 정규화 전 점수다.



## 3. Causal LM logits `[B, L, V]`

Causal LM logits

- `B`: Prompt 수
- `L`: 입력 토큰 위치 수
- `V`: vocabulary 크기
- 각 위치는 다음 토큰 후보 점수다.

## 4. 실행 환경 준비

```python
# Colab/Jupyter에서 아래 예제에 필요한 라이브러리 버전을 설치합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0" "huggingface_hub==1.20.1"
```

```python
# Tensor 연산과 device 관리를 위해 PyTorch를 불러옵니다.
import torch

# 분류 Model, Causal LM, 각 Tokenizer를 자동 선택하는 클래스를 불러옵니다.
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# GPU가 있으면 CUDA, 없으면 CPU를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
```

---

## 5. Sequence Classification 추론

```python
# 영어 SST-2 감성 분류용 Fine-tuned Model ID입니다.
CLASSIFIER_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 분류 모델이 사용한 Tokenizer를 불러옵니다.
classifier_tokenizer = AutoTokenizer.from_pretrained(
    CLASSIFIER_ID,
)

# Base DistilBERT 위에 Sequence Classification Head가 붙은 모델을 불러옵니다.
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    CLASSIFIER_ID,
)

# 모델을 device로 이동하고 평가 모드로 전환합니다.
classifier_model = classifier_model.to(device)
classifier_model.eval()

# 두 문장을 Batch로 처리합니다.
classification_texts = [
    "The explanation was clear and useful.",
    "The documentation was confusing and incomplete.",
]

# 길이가 다른 문장을 padding해 [B, L] Tensor로 만듭니다.
classification_inputs = classifier_tokenizer(
    classification_texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# 입력 Tensor를 모델과 같은 device로 이동합니다.
classification_inputs = {
    name: tensor.to(device)
    for name, tensor in classification_inputs.items()
}

# gradient 없이 forward를 실행합니다.
with torch.inference_mode():
    classification_outputs = classifier_model(**classification_inputs)

# 분류 Head가 만든 [B, C] logits를 꺼냅니다.
classification_logits = classification_outputs.logits
print("classification logits shape:", tuple(classification_logits.shape))
```

### 5-1. label과 probability 해석

```python
# 마지막 축 C에서 softmax를 적용해 문장별 클래스 분포를 만듭니다.
classification_probs = torch.softmax(
    classification_logits,
    dim=-1,
)

# 각 문장에서 가장 큰 클래스 index를 선택합니다.
predicted_ids = classification_probs.argmax(dim=-1)

# 출력용으로 CPU list로 변환합니다.
predicted_ids_list = predicted_ids.detach().cpu().tolist()
classification_probs_cpu = classification_probs.detach().cpu()

# Model config의 id2label mapping을 사용해 정수 ID를 사람이 읽는 label로 바꿉니다.
for row_index, class_id in enumerate(predicted_ids_list):
    label = classifier_model.config.id2label[class_id]
    probability = float(classification_probs_cpu[row_index, class_id])
    print(
        f"text={classification_texts[row_index]!r} | "
        f"label={label} | probability={probability:.4f}"
    )
```

---

## 6. Causal LM forward

```python
# 영어 GPT-2 Causal LM Model ID입니다.
CAUSAL_LM_ID = "openai-community/gpt2"

# GPT-2 Tokenizer와 Causal LM Head가 붙은 모델을 불러옵니다.
causal_tokenizer = AutoTokenizer.from_pretrained(
    CAUSAL_LM_ID,
)
causal_model = AutoModelForCausalLM.from_pretrained(
    CAUSAL_LM_ID,
)

# 모델을 device로 이동하고 추론 모드로 전환합니다.
causal_model = causal_model.to(device)
causal_model.eval()

# 다음 토큰 후보를 관찰할 Prompt입니다.
prompt = "Artificial intelligence can"

# Prompt를 [B, L] input_ids와 attention_mask로 변환합니다.
causal_inputs = causal_tokenizer(
    prompt,
    return_tensors="pt",
)

# 입력 Tensor를 모델과 같은 device로 이동합니다.
causal_inputs = {
    name: tensor.to(device)
    for name, tensor in causal_inputs.items()
}

# generate()가 아니라 forward를 한 번 실행해 위치별 logits를 관찰합니다.
with torch.inference_mode():
    causal_outputs = causal_model(**causal_inputs)

# Causal LM Head의 대표 출력 shape은 [B, L, V]입니다.
causal_logits = causal_outputs.logits
print("causal LM logits shape:", tuple(causal_logits.shape))
```

### 선택 · 6-1. 다음 토큰 후보 확인

```python
# 단일 Prompt이고 padding이 없으므로 마지막 입력 위치의 vocabulary logits를 선택합니다.
next_token_logits = causal_logits[0, -1, :]

# vocabulary 축에 softmax를 적용합니다.
next_token_probs = torch.softmax(
    next_token_logits,
    dim=-1,
)

# 확률이 높은 후보 다섯 개를 선택합니다.
top_probs, top_ids = torch.topk(
    next_token_probs,
    k=5,
)

# 출력용으로 CPU list로 변환합니다.
top_ids_list = top_ids.detach().cpu().tolist()
top_probs_list = top_probs.detach().cpu().tolist()

# 각 ID를 같은 GPT-2 Tokenizer로 decode합니다.
for rank, (token_id, probability) in enumerate(
    zip(top_ids_list, top_probs_list),
    start=1,
):
    token_text = causal_tokenizer.decode([token_id])
    print(
        f"{rank}위 | token={token_text!r} | "
        f"probability={probability:.4f}"
    )
```

## 7. 결과 비교



<img src="{{ '/assets/images/uploads\deep-learning-advanced\04_task_class_selection.png
' | relative_url }}" alt="04_task_class_selection.png
" loading="lazy">

AutoModel 선택

| 항목 | Sequence Classification | Causal LM |
| --- | --- | --- |
| 클래스 | `AutoModelForSequenceClassification` | `AutoModelForCausalLM` |
| Head 출력 | 클래스 점수 | 위치별 vocabulary 점수 |
| logits shape | `[B, C]` | `[B, L, V]` |
| 후처리 | softmax + argmax + id2label | 마지막 위치 + softmax/top-k 또는 generate |


## 참고 · 11. 이해도 점검

1. Task Head의 역할은 무엇인가요?
2. `[B, C]`와 `[B, L, V]`의 차이는 무엇인가
3. GPT에서 다음 토큰 후보를 확인할 때 어느 위치를 선택하나?
4. 분류 label 문자열은 어디에서 확인하나

### 정답 확인

1. Base hidden representation을 특정 문제의 출력으로 바꾼다.
2. 문장별 클래스 점수와 위치별 vocabulary 점수.
3. 마지막 실제 입력 토큰 위치.
4. `model.config.id2label`.

---

## 12. 이번 강의 요약

- Task-specific AutoModel은 Base Model 위에 문제별 Head를 붙입니다.
- 분류 logits는 `[B, C]`, Causal LM logits는 `[B, L, V]`입니다.
- 필요한 출력 shape을 먼저 생각하면 적절한 AutoModel 클래스를 고르기 쉽다
- 다음 강에서는 모델과 Tokenizer를 저장하고 로컬에서 재로드해 추론을 재현한다

