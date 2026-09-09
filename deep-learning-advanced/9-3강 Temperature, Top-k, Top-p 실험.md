---
title: 9-3강 Temperature, Top-k, Top-p 실험
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '9-3강 Temperature, Top-k, Top-p 실험' 정리
---

## 1. 생성 파라미터를 왜 조절할까

같은 모델과 같은 prompt라도 생성 파라미터가 달라지면 결과가 달라진다.

낮은 다양성이 필요한 업무와 높은 다양성이 필요한 업무는 적절한 설정이 다르다

## 2. Temperature

Temperature는 다음 토큰 확률 분포의 모양을 조절한다

낮은 temperature는 높은 확률 후보에 더 집중하게 만들고, 

높은 temperature는 더 다양한 후보가 선택될 가능성을 키운다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_temperature.png' | relative_url }}" alt="01_temperature.png" loading="lazy">

| 값 | 경향 | 주의점 |
| --- | --- | --- |
| 낮음 | 안정적, 보수적 | 다양성이 부족할 수 있습니다. |
| 중간 | 안정성과 다양성 균형 | 태스크별 실험 필요 |
| 높음 | 다양하고 예측 불가능 | 의미가 흐려질 수 있습니다. |

## 3. Top-k

Top-k는 확률이 높은 상위 k개 후보만 남깁니다. 예를 들어 `top_k=10`이면 다음 토큰 후보 중 상위 10개만 남기고 그 안에서 sampling합니다.


<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_top_k.png' | relative_url }}" alt="02_top_k.png" loading="lazy">

Top-k Sampling


## 4. Top-p

Top-p는 누적 확률이 p에 도달할 때까지 후보를 남긴다. 이 방식은 상황에 따라 후보 수가 달라진다

확률이 한두 개 후보에 몰려 있으면 적은 후보만 남고, 분포가 넓으면 더 많은 후보가 남는다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_top_p.png' | relative_url }}" alt="03_top_p.png" loading="lazy">

Top-p Sampling

## 5. 실험표 만들기

```python
# ============================================================
# Temperature, Top-k, Top-p 실험표 만들기
# ============================================================
# 같은 prompt에 대해 generation parameter를 바꾸고 결과를 표로 저장합니다.
# 작은 모델은 출력 품질이 낮을 수 있으므로, 여기서는 파라미터 효과 관찰에 집중합니다.

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

prompt = "In the future, artificial intelligence will"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

# 실험할 generation 설정 목록입니다.
# 각 설정은 같은 prompt에 대해 서로 다른 decoding 조건을 의미합니다.
settings = [
    {"name": "low_temperature", "temperature": 0.3, "top_k": 50, "top_p": 1.0},
    {"name": "medium_temperature", "temperature": 0.8, "top_k": 50, "top_p": 1.0},
    {"name": "top_k_10", "temperature": 0.8, "top_k": 10, "top_p": 1.0},
    {"name": "top_p_0_9", "temperature": 0.8, "top_k": 0, "top_p": 0.9},
]

records = []
for config in settings:
    # sampling 결과 비교를 위해 seed를 고정합니다.
    # 실제 운영에서는 여러 seed로 반복 생성해 품질 변동도 보는 것이 좋습니다.
    set_seed(42)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    records.append({
        "setting": config["name"],
        "temperature": config["temperature"],
        "top_k": config["top_k"],
        "top_p": config["top_p"],
        "output": output_text,
    })

result_df = pd.DataFrame(records)
display(result_df)
```
출력

| index | setting | temperature | top_k | top_p | output |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | `low_temperature` | 0.3 | 50 | 1.0 | In the future, artificial intelligence willoho... |
| 1 | `medium_temperature` | 0.8 | 50 | 1.0 | In the future, artificial intelligence willoho... |
| 2 | `top_k_10` | 0.8 | 10 | 1.0 | In the future, artificial intelligence will Jr... |
| 3 | `top_p_0_9` | 0.8 | 0 | 0.9 | In the future, artificial intelligence will Ju... |


### 코드 해설

실험표의 핵심은 설정과 출력을 함께 저장하는 것이다.

한 번의 출력만 보고 결론을 내리면 우연에 영향을 많이 받는다.

실제 비교에서는 같은 설정으로 여러 번 생성하고, 반복, 의미 흐림, 형식 실패, 사실성 문제를 함께 기록해야 한다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_experiment_grid.png' | relative_url }}" alt="04_experiment_grid.png" loading="lazy">

생성 실험 Grid

