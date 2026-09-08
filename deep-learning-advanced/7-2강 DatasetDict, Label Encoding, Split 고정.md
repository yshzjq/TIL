---
title: 7-2강 DatasetDict, Label Encoding, Split 고정
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-2강 DatasetDict, Label Encoding, Split 고정' 정리
---

## 1. DatasetDict와 split의 역할

Fine-tuning에서 데이터는 용도에 따라 나누어야 한다

`train`은 모델이 학습하는 데이터이고, 

`validation`은 학습 중 설정을 고르고 과적합을 확인하는 데이터다. 

`test`는 최종 성능을 마지막에 확인하는 데이터입니다.

DatasetDict와 split

| split | 역할 | 파라미터 업데이트 여부 |
| --- | --- | --- |
| train | 모델이 학습하는 데이터 | 업데이트함 |
| validation | 학습 중 성능 확인과 모델 선택 | 업데이트하지 않음 |
| test | 최종 성능 확인 | 업데이트하지 않음 |

validation 성능을 계속 보면서 모델을 고르면 validation도 간접적으로 의사결정에 사용된 것이다.
<br>
그래서 최종 보고에는 가능하면 별도 test 성능을 남기는 것이 좋습니다.

Validation은 모델을 고르는 데 사용했기 때문에 완전히 처음 보는 데이터가 아니므로, 최종 성능 확인은 따로 남겨둔 Test 데이터로 하는 것이 좋다.

## 2. Label Encoding과 매핑 저장


라벨 이름은 사람이 이해하기 좋지만 모델은 정수 id를 학습한다.<br>
따라서 라벨 이름과 정수 id의 대응 관계를 저장해야 한다.


<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_label_encoding.png
' | relative_url }}" alt="02_label_encoding.png
" loading="lazy">

Label Encoding 흐름

```python
# ============================================================
# 데이터셋 로드와 label mapping 생성
# ============================================================
from pathlib import Path
import json
from datasets import load_dataset, DatasetDict

DATASET_ID = "klue/klue"
DATASET_CONFIG = "ynat"
TEXT_COL = "title"
LABEL_COL = "label"
PROJECT_DIR = Path("artifacts/ynat_finetuning")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

raw_ds = load_dataset(DATASET_ID, DATASET_CONFIG)

# label 컬럼의 ClassLabel 메타데이터에서 라벨 이름을 가져옵니다.
label_names = raw_ds["train"].features[LABEL_COL].names
num_labels = len(label_names)

# id2label은 모델 config에 저장해두면 추론 결과를 사람이 읽기 쉬워집니다.
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}

# JSON은 key가 문자열로 저장되므로, 나중에 불러올 때 정수 변환이 필요할 수 있습니다.
label_map_path = PROJECT_DIR / "label_mapping.json"
with label_map_path.open("w", encoding="utf-8") as f:
    json.dump(
        {"id2label": id2label, "label2id": label2id},
        f,
        ensure_ascii=False,
        indent=2,
    )

print("라벨 수:", num_labels)
print("저장 경로:", label_map_path)
print("id2label:", id2label)
```

출력

```
라벨 수: 7
저장 경로: artifacts/ynat_finetuning/label_mapping.json
id2label: {0: 'IT과학', 1: '경제', 2: '사회', 3: '생활문화', 4: '세계', 5: '스포츠', 6: '정치'}
```


### 코드 해설

Label Mapping

**매핑(mapping)**은 두 값을 서로 대응시켜 놓는 것
```
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in id2label.items()}
```
예를 들어 다음과 같이 연결된다.

0 ↔ IT과학<br>
1 ↔ 경제<br>
2 ↔ 사회<br>

id2label: 클래스 ID → 라벨 이름

label2id: 라벨 이름 → 클래스 ID

모델은 "경제" 같은 문자열이 아니라 0, 1, 2와 같은 정수 클래스 ID를 기준으로 예측한다.

따라서 모델을 만들 때 id2label과 label2id를 AutoModelForSequenceClassification의 config에 함께 저장해두면,
<br>
 각 클래스 ID가 어떤 실제 라벨을 의미하는지 쉽게 확인할 수 있다.
```
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
```
예를 들어 모델의 예측 결과가 1 일때

id2label[1]

을 통해 사람이 읽을 수 있는 라벨 이름으로 변환할 수 있다.

1 → "경제"

핵심

Label Mapping은 모델이 사용하는 클래스 번호와 사람이 이해하는 라벨 이름을 연결해 놓은 대응표이다.

## 3. 재현 가능한 split 만들기

KLUE YNAT은 기본적으로 train과 validation split을 제공한다

실습 흐름을 명확히 하기 위해 기존 train을 다시 train/validation으로 나누고, 원래 validation을 최종 test처럼 사용한다



<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_reproducible_split.png
' | relative_url }}" alt="03_reproducible_split.png
" loading="lazy">


재현 가능한 split

```python
# ============================================================
# train/validation/test split 구성
# ============================================================
SEED = 42

# 원래 train split을 다시 train/validation으로 나눕니다.
# test_size=0.1은 원래 train 데이터의 10%를 validation으로 사용한다는 뜻입니다.
# seed를 고정하면 같은 코드와 같은 데이터 버전에서 같은 분할을 다시 만들 수 있습니다.
# stratify_by_column을 사용하면 label 분포를 가능한 한 유지하면서 나눕니다.
split_ds = raw_ds["train"].train_test_split(
    test_size=0.1,
    seed=SEED,
    stratify_by_column=LABEL_COL,
)

# DatasetDict로 split 이름을 명확히 부여합니다.
# 원래 KLUE validation은 여기서 최종 test 역할로 둡니다.
dataset = DatasetDict({
    "train": split_ds["train"],
    "validation": split_ds["test"],
    "test": raw_ds["validation"],
})

print(dataset)
for split_name, split_data in dataset.items():
    print(split_name, "샘플 수:", len(split_data))
```

출력

```
DatasetDict({
    train: Dataset({
        features: ['guid', 'title', 'label', 'url', 'date'],
        num_rows: 41110
    })
    validation: Dataset({
        features: ['guid', 'title', 'label', 'url', 'date'],
        num_rows: 4568
    })
    test: Dataset({
        features: ['guid', 'title', 'label', 'url', 'date'],
        num_rows: 9107
    })
})
train 샘플 수: 41110
validation 샘플 수: 4568
test 샘플 수: 9107
```

## 선택 · 4. split 누수 점검

분할 후에는 같은 text가 여러 split에 동시에 들어가지 않았는지 확인


```python
# ============================================================
# split 간 text 중복 점검
# ============================================================
# 각 split의 title을 set으로 바꿉니다.
# set은 중복을 제거하고 교집합 계산을 빠르게 할 수 있습니다.
train_texts = set(dataset["train"][TEXT_COL])
valid_texts = set(dataset["validation"][TEXT_COL])
test_texts = set(dataset["test"][TEXT_COL])

# split 간 교집합을 계산합니다.
train_valid_overlap = train_texts & valid_texts
train_test_overlap = train_texts & test_texts
valid_test_overlap = valid_texts & test_texts

print("train-validation 중복:", len(train_valid_overlap))
print("train-test 중복:", len(train_test_overlap))
print("validation-test 중복:", len(valid_test_overlap))

# 중복 예시를 일부 출력합니다.
# 실제 프로젝트에서는 중복이 발견되면 분할 전 중복 제거 또는 group split을 검토합니다.
print("중복 예시:", list(train_valid_overlap)[:3])
```

출력

```
train-validation 중복: 0
train-test 중복: 0
validation-test 중복: 0
중복 예시: []
```


### 코드 해설

중복이 모두 0이어야만 항상 좋은 것은 아니다.

문제 성격에 따라 같은 문장이 여러 번 등장하는 것이 자연스러울 수도 있다.

 하지만 train과 test에 같은 문장이 들어가면 모델이 일반화한 것이 아니라 외운 결과가 될 수 있으므로 반드시 확인해야 한다

 ## 선택 · 5. 연습 문제

1. `train`, `validation`, `test`의 역할을 각각 한 문장으로 설명

2. `stratify_by_column=LABEL_COL`을 사용하는 이유

3. split 간 중복 text가 많으면 어떤 문제가 생길 수 있는지 
설명

4. `label_mapping.json`에 저장해야 할 정보 두 가지

---

## 참고 · 6. 정답 확인

1. `train`은 모델 파라미터를 업데이트하는 데이터다. 
<br>
`validation`은 학습 중 모델 상태와 설정을 확인하는 데이터다. 
<br>
`test`는 최종 모델 성능을 마지막에 확인하는 데이터다.

2. 라벨 분포가 train과 validation에 비슷하게 유지되도록 하기 위해 사용한다.
<br>
라벨 분포가 크게 달라지면 validation metric이 실제 성능을 대표하지 못할 수 있다.

3. 같은 문장이 train과 test에 동시에 있으면 모델이 일반화한 것이 아니라 문장을 외워 맞힐 수 있다.
<br>
이 경우 평가 성능이 실제보다 높게 보일 수 있다.

4. `id2label`과 `label2id`를 저장해야 합니다. 
<br>
추가로 라벨 설명, 데이터 버전, split seed도 함께 기록하면 좋다.
