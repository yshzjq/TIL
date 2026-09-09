---
title: 8-4강 PEFT 개념과 적용 기준(선택 정리 필요)
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '8-4강 PEFT 개념과 적용 기준' 정리
---

## 1. PEFT의 큰 그림

대형 언어모델은 파라미터 수가 많기 때문에 전체 Fine-tuning이 부담될 수 있다.

PEFT는 기존 모델 대부분을 고정하고, 작은 학습 모듈만 업데이트하여 메모리와 저장 비용을 줄이는 접근이다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_peft_big_picture.png' | relative_url }}" alt="01_peft_big_picture.png" loading="lazy">

PEFT의 장점은 다음과 같다.

- 전체 모델보다 학습 파라미터 수가 적다.

- 여러 태스크별 adapter를 따로 저장하고 교체할 수 있다.

- 원본 모델을 유지하면서 특정 태스크에 맞춘 조정을 할 수 있다.

단점도 있습니다.

- Full Fine-tuning보다 항상 좋은 것은 아니다.

- 어떤 layer와 모듈에 adapter를 넣을지 결정해야 한다.

- adapter 병합, 저장, 배포 전략을 관리해야 한다.


Adapter = 원본 모델에 붙이는 작은 추가 학습 파라미터이고, 특정 태스크에 더 잘 맞게 출력이 나오도록 조정하는 역할을 한다.

## 2. LoRA 직관

<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_lora_matrix.png' | relative_url }}" alt="02_lora_matrix.png" loading="lazy">

| 그림 | 의미 |
|---|---|
| `기존 가중치 W` | 원래 모델 파라미터, 보통 고정 |
| `작은 A` | LoRA가 추가한 작은 학습 파라미터 |
| `작은 B` | LoRA가 추가한 또 다른 작은 학습 파라미터 |
| `W + BA` | 원래 모델에 LoRA가 학습한 변화를 더한 효과 |

LoRA는 원래 큰 가중치 W를 직접 학습하지 않고, 작은 행렬 A, B만 학습해서 BA라는 변화량을 만든 뒤 원래 가중치에 더하는 방법이다.

큰 모델 전체를 업데이트하지 않고, 특정 layer의 작은 업데이트 방향만 학습한다


LoRA 설정에서 자주 보는 값


| 설정 | 의미 | 직관 |
| --- | --- | --- |
| `r` | LoRA rank | 작을수록 학습 파라미터가 적습니다. |
| `lora_alpha` | LoRA 업데이트 스케일 | 업데이트 영향력을 조절합니다. |
| `target_modules` | LoRA를 붙일 모듈 이름 | attention projection 등에 붙이는 경우가 많습니다. |
| `lora_dropout` | LoRA 경로의 dropout | 과적합 완화에 도움을 줄 수 있습니다. |

LoRA = 기존 파라미터는 고정하고, 작은 추가 파라미터로 보정값을 더한다.

## 3. Prompt Tuning과 LoRA 비교

Prompt Tuning은 입력 앞에 학습 가능한 soft prompt를 붙이는 방식이다.

Soft prompt = 입력 앞에 붙이는, 사람이 읽을 수 없는 학습 가능한 가짜 토큰 embedding

문장 토큰화 → 임베딩 → 앞에 Soft Prompt 벡터 추가 → 모델 입력 → Loss 계산 → Soft Prompt만 학습

LoRA는 모델 내부의 특정 Linear layer에 작은 업데이트 경로를 붙이는 방식이다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_prompt_tuning.png' | relative_url }}" alt="03_prompt_tuning.png" loading="lazy">

Prompt Tuning 직관

| 구분 | Prompt Tuning | LoRA |
| --- | --- | --- |
| 학습 위치 | 입력 앞 soft prompt embedding | 모델 내부 특정 Linear layer |
| 모델 본체 | 보통 고정 | 보통 고정 |
| 직관 | 입력을 더 잘 유도하는 가상 prompt 학습 | 내부 변환을 조금 조정 |
| 자주 쓰는 상황 | 가벼운 task adaptation | 실무형 LLM PEFT 실험 |

## 4. PEFT 적용 기준

PEFT = 큰 모델 전체를 건드리지 않고, 작은 부분만 효율적으로 학습하는 방법

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_method_selection.png' | relative_url }}" alt="04_method_selection.png" loading="lazy">

PEFT 방법 선택 기준

- Full Fine-tuning을 하기에는 GPU 메모리나 저장 비용이 부담될때

- Prompt-only baseline이 부족하고, 반복되는 오류 유형이 있다

- 검수된 학습 데이터가 있다.

- 여러 도메인 또는 고객별 adapter를 따로 관리하고 싶다.

- 원본 모델을 유지하면서 특정 행동만 조정하고 싶다.

PEFT도 학습입니다. 데이터 품질이 낮으면 작은 adapter도 잘못된 패턴을 학습합니다.

## 5. PEFT 코드 읽기



다음 코드는 Hugging Face PEFT에서 LoRA 설정을 읽는 예시입니다. 실제 실행은 GPU 메모리와 모델 크기에 맞는 환경에서 확인하세요.

```python
# ============================================================
# PEFT LoRA 설정 예시
# ============================================================
# 이 코드는 PEFT를 처음 볼 때 어떤 객체와 설정이 등장하는지 읽기 위한 예시입니다.
# 실제 학습은 모델 크기, GPU 메모리, 데이터셋 크기에 따라 조정이 필요합니다.

# Colab에서 필요한 경우 먼저 설치합니다.
# pip install -q "transformers==5.14.1" "peft" "accelerate==1.14.0" "datasets==5.0.1"

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

# 작은 예시 모델을 사용합니다.
# 실제 프로젝트에서는 7장에서 사용한 분류 모델 또는 LLM을 사용할 수 있습니다.
MODEL_ID = "distilbert/distilbert-base-uncased"
NUM_LABELS = 2

# tokenizer는 텍스트를 input_ids와 attention_mask로 바꿉니다.
# PEFT는 모델 학습 방법이고, tokenizer의 역할은 이전 장과 동일합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Sequence Classification head가 붙은 모델을 불러옵니다.
# num_labels는 분류할 클래스 수입니다.
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_LABELS,
)

# LoRA 설정입니다.
# r은 저랭크 행렬의 rank입니다. 작을수록 학습 파라미터가 적습니다.
# lora_alpha는 LoRA 업데이트의 스케일을 조절합니다.
# target_modules는 모델 구조에 따라 달라지므로 실제 모델의 module 이름을 확인해야 합니다.
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # 분류 태스크용 LoRA 설정입니다.
    r=8,                          # LoRA rank입니다.
    lora_alpha=16,                # LoRA 업데이트 스케일입니다.
    lora_dropout=0.05,            # 과적합 완화를 위한 dropout입니다.
    target_modules=["q_lin", "v_lin"],  # DistilBERT attention projection 이름 예시입니다.
)

# get_peft_model은 base_model에 LoRA adapter를 붙인 모델을 반환합니다.
# 이 모델은 원본 파라미터 대부분을 고정하고 LoRA 파라미터만 학습 가능하게 만듭니다.
peft_model = get_peft_model(base_model, lora_config)

# 학습 가능한 파라미터 수를 출력합니다.
# PEFT가 제대로 적용되었다면 trainable parameter 비율이 전체보다 훨씬 작게 나옵니다.
peft_model.print_trainable_parameters()
```
출력

```
trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925
```


### 코드 해설

`LoraConfig`는 LoRA adapter를 어디에, 어떤 크기로 붙일지 정하는 설정이다.

`get_peft_model()`은 기존 모델에 PEFT adapter를 붙인다.

여기서 가장 중요한 확인 포인트는 `print_trainable_parameters()`입니다. 전체 파라미터 대비 학습 가능한 파라미터가 작아야 PEFT의 목적과 맞다.

PEFT가 애초에 전체 모델을 다 학습하지 않고, 아주 적은 수의 파라미터만 학습해서 비용을 줄이려는 방법이기 때문

PEFT의 목적 = 학습할 파라미터 수를 크게 줄이는 것



## 8. Causal LM용 LoRA·QLoRA 체크 (선택 학습)
