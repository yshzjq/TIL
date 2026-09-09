---
title: 9-2강 Greedy Decoding vs Sampling
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '9-2강 Greedy Decoding vs Sampling' 정리
---

## 1. 다음 토큰 선택이 왜 중요한가

Causal LM은 다음 토큰 후보에 대한 logits를 만듭니다. 하지만 logits가 있다고 바로 문장이 만들어지는 것은 아니다

그 후보 중 어떤 토큰을 선택할지 정해야 합니다. 이 선택 규칙이 decoding strategy입니다.

Decoding strategy = 모델이 계산한 다음 토큰 후보들 중 실제 출력 토큰을 선택하는 규칙

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_greedy_vs_sampling.png' | relative_url }}" alt="01_greedy_vs_sampling.png" loading="lazy">

## 2. Greedy decoding

Greedy decoding은 매 단계에서 가장 높은 확률의 토큰을 선택한다.

장점은 안정적이고 재현하기 쉽다. <br>
단점은 결과가 단조롭거나, 한 번 잘못된 선택을 하면 이후 경로가 제한될 수 있다

Greedy가 적절한 경우는 다음과 같다.

- 정보 추출처럼 출력이 안정적이어야 한다.
- 같은 입력에 같은 답변이 나오는 것이 중요하다
- 창의성보다 일관성과 예측 가능성이 중요하다.

## 3. Sampling

Sampling은 확률 분포에서 토큰을 뽑는다.

확률이 높은 토큰이 선택될 가능성이 크지만, 항상 1등 토큰만 선택하지는 않습니다.

따라서 더 다양한 결과가 나올 수 있다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_determinism.png' | relative_url }}" alt="02_determinism.png" loading="lazy">

Sampling이 적절한 경우는 다음과 같습니다.

- 아이디어 생성, 브레인스토밍, 창작처럼 다양성이 중요하다.

- 같은 prompt에서 여러 후보 답변을 받고 싶다.

- 단일 정답이 없는 open-ended generation다.

Sampling은 다양성을 높일 수 있지만, 사실성이나 형식 안정성을 보장하지 않는다.

중요한 업무에서는 사람 검토나 후처리 검증이 필요하다


## 4. Beam search는 언제 보나

Beam search는 여러 후보 경로를 동시에 유지한다

번역이나 요약처럼 비교적 목적이 명확한 생성에서 사용될 수 있다.

하지만 LLM 챗봇형 답변에서는 반복적이거나 다양성이 낮은 결과를 만들 수도 있으므로 항상 정답은 아닙니다.

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_beam_search.png' | relative_url }}" alt="03_beam_search.png" loading="lazy">

Beam Search 직관

## 5. 코드로 비교하기

```python
# ============================================================
# Greedy decoding과 Sampling 비교
# ============================================================
# 같은 prompt에 대해 greedy와 sampling이 어떻게 다른 결과를 만드는지 확인합니다.
# 실제 출력 품질은 모델 크기와 checkpoint에 크게 영향을 받습니다.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "sshleifer/tiny-gpt2"
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

prompt = "A helpful AI assistant explains"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

# Greedy decoding: 매 단계 가장 높은 점수의 토큰을 선택합니다.
with torch.no_grad():
    greedy_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

# Sampling: 확률 분포에서 다음 토큰을 샘플링합니다.
# do_sample=True가 핵심 설정입니다.
set_seed(42)
with torch.no_grad():
    sampled_ids = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_k=50,
    )

print("[Greedy]")
print(tokenizer.decode(greedy_ids[0], skip_special_tokens=True))

print("\n[Sampling]")
print(tokenizer.decode(sampled_ids[0], skip_special_tokens=True))
```

출력

```
[Greedy]
A helpful AI assistant explains factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors

[Sampling]
A helpful AI assistant explainsozygGy incarcer448Mini praying Television Redux Singapore Boone omega Bend soy representations653 factors clearer brutality Pocket skillet Redux Bendacious courtyardMini Bend deflectived Singapore Tre
```

### 코드 해설

`do_sample=False`이면 Greedy decoding입니다.

`do_sample=True`이면 sampling이 활성화됩니다.

 Sampling에서는 `temperature`, `top_k`, `top_p` 같은 설정이 함께 영향을 준다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_setting_table.png' | relative_url }}" alt="04_setting_table.png" loading="lazy">

디코딩 설정표




## 8. do_sample과 생성 파라미터의 적용 조건

`do_sample=False`이면 기본적으로 greedy 또는 beam 기반 선택을 사용하므로 `temperature`, `top_k`, `top_p`를 바꿔도 sampling 결과에 반영 않는다!

```python
# 결정적 비교
model.generate(**inputs, do_sample=False, max_new_tokens=64)

# Sampling 비교
model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=64,
)
```

Sampling 실험에서는 seed 하나의 결과만 비교하지 않는다.

여러 seed와 반복 실행에서 품질 분포, 실패율, 길이를 함께 본다.

Greedy 결과가 매번 같아도 GPU 연산과 라이브러리 설정까지 포함한 완전한 결정성이 자동으로 보장되는 것은 아니다
