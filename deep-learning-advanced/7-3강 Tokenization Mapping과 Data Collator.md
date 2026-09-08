---
title: 7-3강 Tokenization Mapping과 Data Collator
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-3강 Tokenization Mapping과 Data Collator' 정리
---

## 1. 왜 map(tokenize)이 필요한가

모델은 문자열을 직접 계산할 수 없다.
<br>
`Tokenizer`는 문자열을 token id로 바꾸고, 실제 token과 padding 위치를 구분하는 `attention_mask`를 만든다.
<br>
`Dataset.map()`은 이 변환을 데이터셋 전체에 일관되게 적용하는 도구다.

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_tokenization_map.png
' | relative_url }}" alt="01_tokenization_map.png
" loading="lazy">

Tokenization mapping

| 변환 전 | 변환 후 |
| --- | --- |
| `title` | `input_ids` |
| `label` | `labels` 또는 `label` |
| 사람이 읽는 문자열 | 모델이 계산할 정수 Tensor |


## 2. Tokenization 함수 만들기


```python
# ============================================================
# Tokenizer 로드와 tokenization 함수 정의
# ============================================================
from transformers import AutoTokenizer

MODEL_ID = "klue/roberta-small"
TEXT_COL = "title"
LABEL_COL = "label"
MAX_LENGTH = 64

# AutoTokenizer는 MODEL_ID에 맞는 tokenizer class를 자동으로 선택합니다.
# 모델과 tokenizer는 같은 checkpoint 계열을 쓰는 것이 안전합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# tokenize_batch는 Dataset.map(batched=True)에서 호출될 함수입니다.
# examples는 샘플 하나가 아니라 여러 샘플을 모은 dict입니다.
def tokenize_batch(examples):
    # examples[TEXT_COL]은 문자열 list입니다.
    # truncation=True는 MAX_LENGTH를 넘는 문장을 자릅니다.
    # padding은 여기서 하지 않습니다. DataCollatorWithPadding이 batch 단위로 동적 padding을 수행합니다.
    tokenized = tokenizer(
        examples[TEXT_COL],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Trainer는 기본적으로 label 또는 labels 컬럼을 정답으로 사용할 수 있습니다.
    # 여기서는 이후 코드를 명확히 하기 위해 labels 컬럼을 새로 만듭니다.
    tokenized["labels"] = examples[LABEL_COL]
    return tokenized
```

### 코드 해설

이 함수는 한 문장을 처리하는 함수처럼 보이지만, `batched=True`와 함께 사용하면 여러 문장을 한 번에 처리한다.

그래서 `examples[TEXT_COL]`은 문자열 하나가 아니라 문자열 리스트다.

⛔ 주의사항

`padding="max_length"`를 여기서 사용하면 모든 샘플이 `MAX_LENGTH` 길이로 고정된다. 

처음에는 이해하기 쉽지만, 메모리를 더 사용할 수 있다.


## 3. Dataset.map()으로 전체 split 변환

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_dynamic_padding.png
' | relative_url }}" alt="02_dynamic_padding.png
" loading="lazy">

Dynamic padding

```python
# ============================================================
# Dataset.map()으로 train/validation/test 모두 tokenization
# ============================================================
# dataset은 7-2강에서 만든 DatasetDict라고 가정합니다.
# 실제 노트북에서는 7-2강의 split 생성 셀을 먼저 실행하세요.

# 모델 학습에 직접 필요하지 않은 원본 컬럼을 모두 제거합니다.
# tokenize_batch가 새로 반환한 input_ids, attention_mask, labels만 남습니다.
# KLUE YNAT의 guid, url, date 같은 문자열 메타데이터가 collator로 전달되지 않게 합니다.
columns_to_remove = dataset["train"].column_names

# DatasetDict.map()은 train, validation, test split에 같은 변환을 적용합니다.
# batched=True는 여러 샘플을 한 번에 tokenizer에 전달해 처리 속도를 높입니다.
tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=columns_to_remove,
)

print(tokenized_dataset)
print("tokenized train 첫 샘플:", tokenized_dataset["train"][0])
print("컬럼:", tokenized_dataset["train"].column_names)
```

출력

```
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 41110
    })
    validation: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 4568
    })
    test: Dataset({
        features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        num_rows: 9107
    })
})
tokenized train 첫 샘플: {'input_ids': [0, 4989, 4353, 2170, 717, 2208, 29228, 22457, 2810, 2181, 5540, 3917, 2], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'labels': 4}
컬럼: ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
```


### 코드 해설

Dataset.map()은 Dataset 안의 데이터를 하나씩/여러 개씩 꺼내서 tokenize_batch 같은 변환 함수를 적용하는 기능이다.

Trainer = 모델 학습에 필요한 batch 전달, forward, loss, backward, optimizer 업데이트, validation 등을 대신 관리해주는 Hugging Face의 학습 도우미

Dataset.map()을 사용해 tokenize_batch를 데이터셋 전체에 적용하면, 
<br>
사람이 읽는 원본 문장인 title이 모델이 사용할 수 있는 형태로 변환된다.

```
원본 데이터
title, label, url, date ...
        ↓
Dataset.map(tokenize_batch)
        ↓
tokenized 데이터
input_ids, attention_mask, labels

input_ids: 문장을 tokenizer가 변환한 토큰 ID
```

attention_mask: 실제 토큰과 padding 위치를 구분하는 값

labels: 모델이 맞혀야 하는 정답 클래스 ID

이 값들은 이후 Trainer가 batch 단위로 꺼내 모델에 전달해 학습에 사용한다.



Trainer가 모델에 전달할 핵심 입력입니다.

💡 참고사항

디버깅 단계에서는 문제가 생겼을 때 원래 문장을 직접 확인하기 위해 text를 남겨두면 편하기에 원문 text를 남겨두는 것이 유용할 수 있다.

Collator는 여러 샘플을 숫자 Tensor batch로 묶는데, 학습에 필요 없는 문자열 컬럼까지 같이 처리하려 하면 Tensor 변환이나 모델 전달 과정에서 문제가 생길 수 있으므로 제거하는 것이 안전하다.

설정이나 사용하는 collator에 따라 불필요한 문자열 컬럼이 batch 생성 과정까지 넘어가면 Tensor 변환이나 padding 과정에서 오류가 생길 수 있다
<br>
운영 코드에서는 원문을 별도 prediction 파일에 저장하는 방식이 더 깔끔하다(학습용 데이터와 사람이 확인할 데이터를 분리)

## 4. Data Collator와 dynamic padding

Data Collator는 Dataset 샘플 여러 개를 받아 모델이 처리할 수 있는 batch Tensor로 묶는다. 

문장마다 길이가 다르기 때문에, collator는 같은 batch 안에서 길이를 맞추는 padding을 수행할 수 있다.

Data Collator의 역할

```python
# ============================================================
# DataCollatorWithPadding 생성
# ============================================================
from transformers import DataCollatorWithPadding

# tokenizer를 collator에 전달하면 tokenizer의 pad_token_id와 padding 규칙을 사용합니다.
# return_tensors="pt"는 PyTorch Tensor batch를 반환하라는 의미입니다.
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)

# train split에서 샘플 4개를 직접 꺼내 collator에 넣어봅니다.
# 실제 Trainer 내부에서도 이와 비슷한 방식으로 batch가 만들어집니다.
samples = [tokenized_dataset["train"][i] for i in range(4)]
batch = data_collator(samples)

print("batch keys:", batch.keys())
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)
print("input_ids dtype:", batch["input_ids"].dtype)
print("labels dtype:", batch["labels"].dtype)
```
출력
```
batch keys: KeysView({'input_ids': tensor([[    0,  4989,  4353,  2170,   717,  2208, 29228, 22457,  2810,  2181,
          5540,  3917,     2],
        [    0,  4163, 12342,  2079,  5122, 17870,  2118,  1380,  2259,  1852,
             2,     1,     1],
        [    0, 19660,  4623, 12574,     3,  4227,  4110, 21508,     2,     1,
             1,     1,     1],
        [    0,  5610,  2237,  2341, 22563, 27056,  5572,  2232,  6754,  3989,
          8314,     2,     1]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]), 'labels': tensor([4, 2, 4, 0])})
input_ids shape: torch.Size([4, 13])
attention_mask shape: torch.Size([4, 13])
labels shape: torch.Size([4])
input_ids dtype: torch.int64
labels dtype: torch.int64
```


### 코드 해설

`input_ids`와 `attention_mask`의 shape은 보통 `[B, L]`입니다. 여기서 `B`는 batch size, `L`은 현재 batch에서 padding된 sequence length입니다. `labels`의 shape은 `[B]`입니다.

## 5. batch shape 확인

```python
# ============================================================
# 첫 번째 배치의 token과 mask 해석
# ============================================================
# 첫 샘플의 input_ids를 token 문자열로 다시 바꿔 봅니다.
# 이렇게 확인하면 special token, padding, truncation 여부를 디버깅할 수 있습니다.
first_input_ids = batch["input_ids"][0]
first_tokens = tokenizer.convert_ids_to_tokens(first_input_ids)

print("첫 샘플 tokens:")
print(first_tokens)
print("첫 샘플 attention_mask:")
print(batch["attention_mask"][0].tolist())
```

출력

```
첫 샘플 tokens:
['[CLS]', '이란', '영국', '##에', '나', '##포', '유조선', '풀어', '##줘', '##라', '거듭', '요구', '[SEP]']
첫 샘플 attention_mask:
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

### 코드 해설

`attention_mask`에서 1은 실제 token, 0은 padding 위치다. 

학습 중 모델이 padding을 실제 문장 내용처럼 참고하지 않도록 mask가 필요하다.

## 선택 · 6. 연습 문제

1. `Dataset.map(batched=True)`를 사용하는 이유를 설명

2. `padding="max_length"`와 `DataCollatorWithPadding`의 dynamic padding 차이를 설명

3. `remove_columns`를 사용할 때 원본 text를 제거하는 장단점

4. `input_ids`, `attention_mask`, `labels`의 일반적인 shape

---

## 참고 · 7. 정답 확인

1. 여러 샘플을 한 번에 tokenizer에 전달해 처리 속도를 높이고, 같은 전처리를 전체 split에 일관되게 적용하기 위해 사용

2. `padding="max_length"`는 모든 샘플을 지정한 최대 길이까지 padding합니다. 
<br>
Dynamic padding은 현재 batch 안에서 가장 긴 샘플에 맞춰 padding하므로 메모리 사용이 더 효율적일 수 있다.

3. 장점은 Trainer에 불필요한 문자열 컬럼이 전달되어 생길 수 있는 오류를 줄일 수 있다. 
<br>
단점은 학습 중 오류 분석을 할 때 원문을 바로 확인하기 어렵다

4. `input_ids`: `[B, L]`, `attention_mask`: `[B, L]`, `labels`: `[B]`입니다.
