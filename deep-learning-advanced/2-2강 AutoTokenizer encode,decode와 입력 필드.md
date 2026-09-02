---
title: 2-2강 AutoTokenizer encode,decode와 입력 필드
date: 2026-09-02
updated: 2026-09-02
description: KANT 강의 '2-2강 AutoTokenizer encode,decode와 입력 필드' 정리
---

## 1. `AutoTokenizer`가 해결하는 일

`AutoTokenizer.from_pretrained(model_id)`는 checkpoint의 tokenizer 설정을 읽어 적절한 구현을 선택한다.<br>
`Auto`는 임의의 tokenizer를 섞는 기능이 아니라 config 기반 factory입니다.

```python
from transformers import AutoTokenizer

MODEL_ID = "monologg/koelectra-small-v3-discriminator"
# checkpoint의 tokenizer_config를 읽어 맞는 구현을 선택합니다.
# 운영 실험에서는 revision을 commit SHA로 고정하는 편이 안전합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("class:", type(tokenizer).__name__)
print("vocab size:", tokenizer.vocab_size)
print("special tokens:", tokenizer.special_tokens_map)
```

운영 환경에서는 모델 revision과 package version을 함께 고정한다.<br>
캐시만 사용할 때는 `local_files_only=True`를 추가할 수 있다.


## 2. 다섯 API의 경계를 구분한다

| API | 대표 반환 | 주요 용도 |
| --- | --- | --- |
| `tokenize(text)` | token 문자열 list | 분할 결과 관찰 |
| `convert_tokens_to_ids(tokens)` | ID list | token↔ID 대응 확인 |
| `encode(text)` | ID list | 한 문장을 간단히 ID로 변환 |
| `tokenizer(texts, ...)` | `BatchEncoding` | 실제 model 입력 생성 |
| `decode(ids)` | 문자열 | ID를 읽을 수 있는 텍스트로 복원 |

실제 학습·추론에는 batch, padding, truncation, Tensor 반환을 함께 다루는 `tokenizer(...)` 호출이 중심


<img src="{{ '/assets/images/uploads\deep-learning-advanced\Tokenizer API 5가지 역할.png
' | relative_url }}" alt="Tokenizer API 5가지 역할.png
" loading="lazy">


위 도식은 `tokenize()`와 `encode()`를 관찰용 경로로, `tokenizer(...)`를 실제 모델 입력 경로로 구분한다.<br>
`decode()`는 ID를 사람이 읽을 수 있게 되돌리는 도구이지, 원문의 공백과 정규화를 바이트 단위로 복원하는 장치가 아니다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\Tokenizer 내부 파이프라인.png
' | relative_url }}" alt="Tokenizer 내부 파이프라인
" loading="lazy">

Tokenizer 내부 파이프라인: 정규화 → 사전 토큰화 → 토큰화 모델 → 후처리 → 디코딩

Hugging Face 공식 tokenizer pipeline 도식은 normalization, pre-tokenization, tokenizer model, post-processing, decoding을 분리해 보여 준다

AutoTokenizer의 편리한 한 번 호출 안에서도 이 단계들이 순서대로 작동, post-processing에서 special token과 입력 필드가 추가될 수 있다

## 3. 한 문장 입력을 끝까지 추적

```python
text = "한국어 모델은 문장을 토큰으로 나눕니다."

# 같은 원문을 경계 token 전후로 비교해 API별 반환 계약을 추적합니다.
tokens = tokenizer.tokenize(text)  # 사람이 분할 결과를 읽기 위한 token 문자열
ids_without_special = tokenizer.convert_tokens_to_ids(tokens)  # 경계 token 없는 ID
ids_with_special = tokenizer.encode(
    text,
    add_special_tokens=True,  # 모델이 기대하는 [CLS]/[SEP]를 tokenizer가 추가
)
decoded = tokenizer.decode(
    ids_with_special,
    skip_special_tokens=True,  # 사용자에게 보여 줄 문자열에서는 경계 token 제거
)

print("tokens:", tokens)
print("without special:", ids_without_special)
print("with special length:", len(ids_with_special))
print("decoded:", decoded)
```
결과
```
tokens: ['한국어', '모델', '##은', '문장', '##을', '토큰', '##으로', '나', '##눕', '##니다', '.']
without special: [11229, 6918, 4112, 9611, 4292, 32436, 10749, 2236, 4983, 6216, 18]
with special length: 13
decoded: 한국어 모델은 문장을 토큰으로 나눕니다.
```

decode 결과가 원문과 글자 단위로 똑같은지만 확인하지 말고, 토큰들이 제대로 복원되었는지와 의미가 유지되는지를 봐야 한다

universal contract는 모든 tokenizer가 반드시 지켜야 하는 공통 규칙

## 4. `BatchEncoding`과 shape

```python
texts = [
    "금리 인상 가능성에 시장이 긴장했다.",
    "대표팀이 결승에 진출했다.",
]

batch = tokenizer(
    texts,
    padding=True,       # 이 batch의 최장 문장까지만 PAD를 추가
    truncation=True,    # max_length를 넘는 입력은 명시적으로 절단
    max_length=16,      # 교육용 작은 상한; 실제 값은 길이 분포로 결정
    return_tensors="pt",  # list가 아닌 PyTorch Tensor [B,L] 반환
)

for name, value in batch.items():
    # token_type_ids처럼 선택적인 필드도 실제 반환값을 순회해 확인합니다.
    print(name, tuple(value.shape), value.dtype)
```

대표적으로 `input_ids`와 `attention_mask`는 `[B, L]`입니다. <br>
KoELECTRA 계열은 `token_type_ids`도 반환할 수 있다. <br>
모델·tokenizer에 따라 이 필드는 없을 수 있으므로 무조건 가정하지 않는다

token_type_ids는 각 토큰이 첫 번째 문장인지 두 번째 문장인지 구분해 주는 표시값

```
B = 문장 수
L = 이 batch에서 padding 후 token 길이(단, max_length 이하)
```

## 5. Special token은 tokenizer가 넣게 한다

`add_special_tokens=True`가 기본인 호출에서 `[CLS]`, `[SEP]` 같은 모델별 경계 token이 추가된다<br>
 직접 문자열로 붙이면 중복 special token이나 잘못된 ID가 생길 수 있다.

 ```python
ids = batch["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(ids)
# 각 token 옆에 1(실제 입력) 또는 0(PAD)을 붙여 눈으로 확인합니다.
print(list(zip(tokens, batch["attention_mask"][0].tolist())))
```

```
[
('[CLS]', 1), ('금리', 1), ('인상', 1), ('가능', 1), ('##성', 1), ('##에', 1), ('시장', 1),
 ('##이', 1), ('긴장', 1), ('##했', 1), ('##다', 1), ('.', 1), ('[SEP]', 1)
]
```

Mask가 0인 PAD 위치까지 `decode(..., skip_special_tokens=False)`하면 special token 문자열이 보일 수 있다.<br>
사람에게 보여줄 최종 텍스트는 보통 `skip_special_tokens=True`를 사용한다

PAD = batch 길이 맞추기용 빈칸 토큰
attention_mask의 0 = 이 위치는 PAD니까 모델이 신경 쓰지 말라는 표시


-**skip_special_tokens**<br>
False → 내부 구조 확인, 디버깅, 토큰화 결과 점검<br>
True → 사람에게 보여줄 자연스러운 최종 텍스트

## 6. 출력 계약을 검증하는 assert

```python
import torch

# input ID와 mask는 같은 [B,L] 격자를 설명해야 합니다.
assert batch["input_ids"].shape == batch["attention_mask"].shape
assert batch["input_ids"].dtype == torch.long  # embedding index는 정수형
assert batch["attention_mask"].max().item() <= 1  # 대표적인 binary mask 계약
assert batch["attention_mask"].min().item() >= 0
```

실제 코드에서는 `batch["input_ids"].dtype == torch.long`처럼 확인한다.<br>
필드 이름·shape·dtype·special token ID를 함께 검증해야 조용한 전처리 오류를 줄일 수 있다.

## 요약

- `AutoTokenizer`는 checkpoint config에 맞는 tokenizer 구현을 선택하는 factory입니다.
- `tokenize`는 분할 관찰, `tokenizer(...)`는 batch·mask를 포함한 모델 입력 생성에 적합합니다.
- `input_ids`와 `attention_mask`의 shape는 보통 `[B,L]`이며 선택 field는 모델마다 다를 수 있다.
- Decode의 공백 완전 일치보다 checkpoint·special token·field·shape·dtype 계약을 검증한다.