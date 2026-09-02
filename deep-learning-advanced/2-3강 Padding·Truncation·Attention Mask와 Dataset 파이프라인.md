---
title: 2-3강 Padding·Truncation·Attention Mask와 Dataset 파이프라인
date: 2026-09-02
updated: 2026-09-02
description: KANT 강의 '2-3강 Padding·Truncation·Attention Mask와 Dataset 파이프라인' 정리
---

Padding = 짧은 문장 뒤에 빈칸 역할의 PAD를 넣어서 batch 길이를 맞추는 것

## 1. Padding과 truncation은 서로 다른 문제입니다

- **Padding**: 짧은 입력을 PAD token으로 채워 한 batch를 직사각형 Tensor로 만든다
- **Truncation**: 모델·정책의 최대 길이를 넘는 token을 자른다
- **Attention mask**: 실제 key와 PAD key를 구분한다

```
원래 길이:      [7, 4, 5]
동적 padding:   [7, 7, 7]  ← 현재 batch 최장 길이
max_length=6:   [6, 4, 5]  ← 긴 입력은 먼저 절단
최종 batch:     [6, 6, 6]
```

Padding은 정보 손실이 없지만 계산 낭비를 늘릴 수 있다.<br> Truncation은 계산량을 제한하지만 중요한 뒤 문맥을 잃을 수 있다.


<img src="{{ '/assets/images/uploads\deep-learning-advanced\Split먼저한후Padding.png
' | relative_url }}" alt="Split먼저한후Padding.png
" loading="lazy">

split을 먼저 고정해야 같은 문장이 train과 test에 섞이는 일을 막을 수 있다

동적 padding을 batch 직전으로 미뤄야 짧은 batch가 불필요한 PAD를 계산하지 않는다

PyTorch 공식 BetterTransformer 자료는 batch 안의 padding 비율이 커질수록 <br>
불필요한 PAD 계산을 건너뛰는 최적화 효과가 커지는 실측 결과 있다
<br>
이 결과는 “shape를 맞추기 위한 PAD도 계산 비용을 낸다”는 점을 분명히 하며, dynamic padding과 길이별 batching을 쓰는 이유를 뒷받침한다. 공식 출처: PyTorch — A Better Transformer


## 2. 정적 padding과 동적 padding

| 방식 | 설정 위치 | 장점 | 주의점 |
| --- | --- | --- | --- |
| 정적 | tokenizer에서 `padding="max_length"` | shape 고정, 단순 | 짧은 batch에도 PAD가 많음 |
| 동적 | collator에서 `padding=True` | batch별 낭비 감소 | batch마다 `L`이 달라짐 |

동적 Padding =  batch에 들어갈 샘플들”을 고르고 → 그 샘플들끼리 padding한 뒤 → 하나의 Tensor batch로 만든다

Dataset `map`에서는 `padding=False`로 길이를 보존하고, batch를 만들 때 `DataCollatorWithPadding`을 사용한다,
<br>
즉 Dataset에 저장할 때는 문장마다 원래 토큰 길이를 그대로 두고, 실제 batch를 만들 때만 그 batch 안에서 가장 긴 길이에 맞춰 padding한다는 뜻이야.

## 3. DatasetDict를 먼저 분할

```python
from datasets import Dataset, DatasetDict

# 세 클래스가 각각 네 행씩 있는 교육용 원본입니다.
rows = {
    "text": [
        "금리 인상 전망", "수출 증가 발표", "증시 장중 반등",
        "대표팀 결승 진출", "신인 선수 첫 승", "리그 일정 발표",
        "새 AI 반도체 공개", "위성 발사 성공", "보안 업데이트 배포",
        "환율 하락 마감", "감독 전술 공개", "클라우드 보안 강화",
    ],
    "label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2],
}

dataset = Dataset.from_dict(rows)
splits = DatasetDict({
    # 예시에서도 split 간 행을 재사용하지 않습니다. 실제 프로젝트는 고정 sample ID를 저장합니다.
    "train": dataset.select([0, 1, 3, 4, 6, 7]),
    "validation": dataset.select([2, 5, 8]),
    "test": dataset.select([9, 10, 11]),
})

# 같은 원본 행이 두 split에 들어가지 않았는지 index 수준에서도 확인합니다.
split_indices = {"train": {0, 1, 3, 4, 6, 7}, "validation": {2, 5, 8}, "test": {9, 10, 11}}
assert split_indices["train"].isdisjoint(split_indices["validation"])
assert split_indices["train"].isdisjoint(split_indices["test"])
assert split_indices["validation"].isdisjoint(split_indices["test"])
print(splits)
```
결과
```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 6
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 3
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 3
    })
})
```

세 split은 서로 다른 원본 행을 사용

예시는 index를 눈으로 확인할 수 있을 만큼 작게 만들었지만,<br>
실제 프로젝트에서는 sample ID를 별도 열로 보존하고 split 간 교집합이 비었는지 자동 검사해야 한다

## 4. `map(batched=True)`로 같은 함수를 적용

```python
from transformers import AutoTokenizer

MODEL_ID = "monologg/koelectra-small-v3-discriminator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenize_batch(batch):
    # batched=True에서는 batch["text"]가 문자열 하나가 아니라 list입니다.
    encoded = tokenizer(
        batch["text"],
        truncation=True,  # 상한을 넘는 샘플만 자릅니다.
        max_length=16,    # 예제 상한; 실제 값은 아래 길이 감사로 결정합니다.
        padding=False,    # map 단계에서는 원래 token 길이를 보존합니다.
    )
    encoded["labels"] = batch["label"]  # model이 기대하는 복수형 key
    return encoded

tokenized = splits.map(tokenize_batch, batched=True)
print(tokenized["train"].column_names)
print([len(ids) for ids in tokenized["train"]["input_ids"]])
```

출력

```
['text', 'label', 'input_ids', 'token_type_ids', 'attention_mask', 'labels']
[5, 5, 6, 6, 6, 5]
```

`batched=True`이면 함수는 한 행이 아니라 열별 list 묶음을 받는다.<br> 
함수가 반환한 list들의 길이는 입력 batch 행 수와 같아야 한다

## 5. Collator가 실제 batch를 만든다

```python
import torch
from transformers import DataCollatorWithPadding

# 이 호출 시점에 네 샘플 중 최장 길이까지만 PAD를 붙입니다.
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt",
)
# 원문 text·label 문자열은 오류 분석용으로 Dataset에 남기되 Collator에는 숫자 열만 전달합니다.
model_input_keys = ["input_ids", "attention_mask", "labels"]
if "token_type_ids" in tokenized["train"].column_names:
    model_input_keys.append("token_type_ids")
features = [
    {key: tokenized["train"][i][key] for key in model_input_keys}
    for i in range(4)
]  # B=4
batch = collator(features)

print("input:", tuple(batch["input_ids"].shape))
print("mask:", tuple(batch["attention_mask"].shape))
print("labels:", tuple(batch["labels"].shape))

pad_positions = batch["input_ids"].eq(tokenizer.pad_token_id)
# PAD ID가 있는 모든 칸과 attention_mask=0인 칸이 정확히 같아야 합니다.
assert torch.equal(pad_positions, batch["attention_mask"].eq(0))
```

출력

```
input: (4, 6)
mask: (4, 6)
labels: (4,)
```

PAD ID와 mask 0 위치의 일치는 tokenizer/model 입력 계약의 핵심<br>
이 검사가 실패하면 모델이 PAD를 문맥으로 읽거나, 반대로 실제 token을 가릴 수 있다.

PAD 위치와 attention_mask의 0 위치가 정확히 일치해야 모델이 실제 문장은 보고, 길이 맞추기용 PAD는 무시할 수 있다

## 6. 최대 길이는 데이터로 정한다

원본 token 길이를 먼저 측정하고 후보별 절단 비율을 계산한다

```python
# max_length 정책을 정할 때 test 입력을 들여다보지 않고 train 길이부터 측정합니다.
train_texts = splits["train"]["text"]
lengths = [
    len(tokenizer(text, add_special_tokens=True)["input_ids"])
    for text in train_texts
]
for candidate in (8, 16, 32):
    cut = sum(length > candidate for length in lengths)  # 잘릴 샘플 수
    print(candidate, cut, f"{cut / len(lengths):.1%}")

# Train 통계와 validation 성능으로 정책을 확정한 뒤에만 test 절단률을 보고합니다.
fixed_max_length = 16
test_lengths = [
    len(tokenizer(text, add_special_tokens=True)["input_ids"])
    for text in splits["test"]["text"]
]
test_cut = sum(length > fixed_max_length for length in test_lengths)
print("fixed test truncation:", test_cut, f"{test_cut / len(test_lengths):.1%}")
```

평균만 보면 긴 꼬리를 놓칠 수 있으므로 train의 median, p95, max와 절단 후보를 먼저 본다<br>
최종 길이는 validation 성능·메모리와 함께 결정하고, test 절단률은 정책을 바꾸기 위한 입력이 아니라 확정된 정책의 적용 결과로만 기록한다.<br>
`max_length`를 키우면 메모리와 계산량이 증가합니다.

## 7. Split과 전처리에서 지켜야 할 경계

1. 원본을 먼저 감사하고 split ID를 고정한다
2. 학습 데이터에만 vocabulary·통계 fitting이 필요한 변환은 train에서만 학습한다
3. Tokenizer처럼 고정된 사전학습 변환은 세 split에 같은 config로 적용한다
4. Validation으로 길이·학습 설정을 결정한다
5. Test는 선택 완료 후 한 번 평가한다