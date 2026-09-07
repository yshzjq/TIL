---
title: 6-5강 Model·Tokenizer 저장 재로드와 추론 재현
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '6-5강 Model·Tokenizer 저장 재로드와 추론 재현' 정리
---

## 1. 저장해야 하는 산출물


<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_save_reload_cycle
' | relative_url }}" alt="01_save_reload_cycle
" loading="lazy">


저장·재로드 흐름

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_artifact_directory
' | relative_url }}" alt="02_artifact_directory
" loading="lazy">


Artifact 디렉터리

```
artifacts/sentiment_model/
├── config.json
├── model.safetensors
├── tokenizer.json 또는 vocab 관련 파일
├── tokenizer_config.json
├── special_tokens_map.json
└── inference_metadata.json   # 우리가 추가할 실행 기록
```

## 2. save/reload 전체 흐름

```
Hub에서 로드
-> 동일 입력 baseline logits 저장
-> model.save_pretrained()
-> tokenizer.save_pretrained()
-> 메모리 객체 삭제
-> local directory에서 from_pretrained()
-> 동일 입력 재추론
-> logits와 prediction 비교
```

## 3. 실행 환경 준비

```python
# Colab/Jupyter에서 아래 예제에 필요한 라이브러리 버전을 설치합니다.
!pip install -q "transformers==5.14.1" "datasets==5.0.1" "accelerate==1.14.0" "huggingface_hub==1.20.1" "safetensors==0.8.0"
```

```python
# 파일 경로를 운영체제에 독립적으로 다루기 위해 Path를 사용합니다.
from pathlib import Path

# 재현성 메타데이터를 JSON으로 저장하기 위해 표준 라이브러리를 불러옵니다.
import json
import platform

# Tensor 비교와 device 관리를 위해 PyTorch를 불러옵니다.
import torch

# 버전 기록을 위해 Transformers 패키지 자체와 모델 클래스를 불러옵니다.
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 빠른 저장·재로드 실습에 사용할 영어 감성 분류 checkpoint입니다.
MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 산출물을 모을 로컬 디렉터리입니다.
SAVE_DIR = Path("artifacts/sentiment_model")

# GPU가 있으면 CUDA, 없으면 CPU를 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
```

---

## 4. 저장 전 baseline 추론

```python
# 모델 학습 때 사용한 Tokenizer를 Hub에서 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
)

# Sequence Classification Head가 Fine-tuning된 모델을 Hub에서 불러옵니다.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
)

# 모델을 device로 이동하고 평가 모드로 전환합니다.
model = model.to(device)
model.eval()

# 저장 전후에 완전히 같은 입력을 사용하기 위해 list를 상수처럼 유지합니다.
texts = [
    "The explanation was clear and useful.",
    "The response was slow and disappointing.",
]

# 두 문장을 padding한 PyTorch Batch로 변환합니다.
baseline_inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# 입력 Tensor를 모델과 같은 device로 이동합니다.
baseline_inputs = {
    name: tensor.to(device)
    for name, tensor in baseline_inputs.items()
}

# 저장 전 logits를 gradient 없이 계산합니다.
with torch.inference_mode():
    baseline_outputs = model(**baseline_inputs)

# 이후 객체를 삭제해도 비교할 수 있도록 logits를 CPU에 복사합니다.
# clone()은 현재 값을 별도 Tensor로 보관합니다.
baseline_logits = baseline_outputs.logits.detach().cpu().clone()

# 각 문장의 가장 큰 클래스 index를 저장합니다.
baseline_predictions = baseline_logits.argmax(dim=-1)

print("baseline logits shape:", tuple(baseline_logits.shape))
print("baseline predictions:", baseline_predictions.tolist())
```

---

## 5. Model과 Tokenizer 저장

```python
# 상위 디렉터리가 없어도 생성하고, 이미 존재해도 오류가 나지 않게 설정합니다.
SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# 모델 weight와 config를 Hugging Face 표준 형식으로 저장합니다.
# safe_serialization=True이면 가능한 경우 safetensors 형식을 사용합니다.
model.save_pretrained(
    SAVE_DIR,
    safe_serialization=True,
)

# vocabulary, Tokenizer config, Special Token 정보를 같은 디렉터리에 저장합니다.
tokenizer.save_pretrained(
    SAVE_DIR,
)

# 어떤 파일이 생성되었는지 정렬해 출력합니다.
saved_files = sorted(
    path.name
    for path in SAVE_DIR.iterdir()
)
print("saved files:", saved_files)
```

---

## 6. 객체 삭제와 로컬 재로드

```python
# 실제 재로드 상황을 흉내 내기 위해 기존 모델·Tokenizer·중간 출력 객체를 삭제합니다.
del model
_del_names = ["tokenizer", "baseline_outputs", "baseline_inputs"]
for name in _del_names:
    # globals()에 객체가 있을 때만 삭제해 재실행 시 NameError를 피합니다.
    if name in globals():
        del globals()[name]

# Python garbage collector로 참조가 사라진 객체를 정리합니다.
import gc
gc.collect()

# CUDA를 사용했다면 PyTorch의 미사용 캐시를 비웁니다.
# 이 호출은 살아 있는 Tensor를 삭제하지 않습니다.
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

```python
# Hub Model ID가 아니라 로컬 디렉터리를 from_pretrained()에 전달합니다.
# local_files_only=True는 네트워크를 사용하지 않고 로컬 파일만 읽게 합니다.
reloaded_tokenizer = AutoTokenizer.from_pretrained(
    SAVE_DIR,
    local_files_only=True,
)

# 저장한 config와 model.safetensors를 사용해 분류 모델을 복원합니다.
reloaded_model = AutoModelForSequenceClassification.from_pretrained(
    SAVE_DIR,
    local_files_only=True,
)

# 재로드한 모델도 같은 device로 이동하고 평가 모드로 전환합니다.
reloaded_model = reloaded_model.to(device)
reloaded_model.eval()

print("reloaded model class:", reloaded_model.__class__.__name__)
print("reloaded tokenizer class:", reloaded_tokenizer.__class__.__name__)
```

---

## 7. 저장 전후 결과 비교

```python
# 저장 전과 동일한 문자열과 전처리 옵션으로 다시 Tokenization합니다.
reloaded_inputs = reloaded_tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)

# 입력 Tensor를 재로드한 모델과 같은 device로 이동합니다.
reloaded_inputs = {
    name: tensor.to(device)
    for name, tensor in reloaded_inputs.items()
}

# 재로드한 모델로 logits를 계산합니다.
with torch.inference_mode():
    reloaded_outputs = reloaded_model(**reloaded_inputs)

# 비교를 위해 logits를 CPU에 복사합니다.
reloaded_logits = reloaded_outputs.logits.detach().cpu()

# 가장 큰 클래스 index를 재계산합니다.
reloaded_predictions = reloaded_logits.argmax(dim=-1)

# 같은 dtype/device 조건에서는 매우 가까운 logits를 기대합니다.
# atol과 rtol은 부동소수점의 작은 오차를 허용하는 기준입니다.
torch.testing.assert_close(
    baseline_logits,
    reloaded_logits,
    rtol=1e-5,
    atol=1e-6,
)

# 클래스 prediction도 정확히 같은지 확인합니다.
assert torch.equal(
    baseline_predictions,
    reloaded_predictions,
)

print("logits comparison: PASS")
print("predictions comparison: PASS")
print("reloaded predictions:", reloaded_predictions.tolist())
```


다른 하드웨어, dtype, 커널을 사용하면 bit 단위로 완전히 같지 않을 수 있다. 

그래서 실수 Tensor는 `assert_close()`로 비교하고, 최종 label도 함께 비교한다

## 8. 재현성 메타데이터 기록

목표는 "왜 결과가 달라졌는지 설명할 수 있는상태"를 만드는 것

재현성 계층

```python
# Hub에서 실제 commit hash를 확인하기 위해 model_info를 불러옵니다.
from huggingface_hub import model_info

# 현재 Hub 저장소의 metadata를 조회합니다.
info = model_info(MODEL_ID)

# 재로드에 필요한 핵심 환경과 설정을 dict로 정리합니다.
metadata = {
    "source_model_id": MODEL_ID,
    "source_revision": info.sha,
    "python_version": platform.python_version(),
    "pytorch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "device_type": device.type,
    "padding": True,
    "truncation": True,
    "texts": texts,
}

# 메타데이터를 저장할 파일 경로입니다.
metadata_path = SAVE_DIR / "inference_metadata.json"

# ensure_ascii=False는 한국어가 있을 때 Unicode escape 대신 읽기 쉬운 문자열로 저장합니다.
# indent=2는 사람이 검토하기 좋은 들여쓰기 형식입니다.
metadata_path.write_text(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("metadata path:", metadata_path)
print(metadata_path.read_text(encoding="utf-8"))
```

## 참고 · 12. 이해도 점검

1. 모델과 Tokenizer를 함께 저장해야 하는 이유는 무엇인가
2. `local_files_only=True`의 역할
3. 실수 logits를 `torch.equal()`보다 `assert_close()`로 비교하는 이유는 무엇인가
4. 재현성 메타데이터에 어떤 항목을 기록해야 하나

### 정답 확인

1. 같은 문자열을 같은 token ID와 입력 형식으로 재현하기 위해서
2. 네트워크 없이 로컬 파일만 사용하도록 제한
3. 하드웨어와 dtype에 따른 작은 부동소수점 오차를 허용하기 위해서
4. Model ID, revision, 라이브러리 버전, 전처리 설정, device/dtype, 입력 또는 평가 조건 등을 기록합니다.

## 14. 이번 강의 요약

- `save_pretrained()`는 모델과 Tokenizer를 Hugging Face 표준 형식으로 저장한다
- Model과 Tokenizer는 같은 디렉터리에 한 쌍으로 관리하는 것이 안전하다
- 로컬 재로드 후 동일 입력의 logits와 prediction을 비교해 추론 재현을 확인한다
- 재현성은 seed 하나가 아니라 revision, 환경, 전처리, dtype/device, 산출물 기록의 묶음이다
