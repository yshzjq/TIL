---
title: 8-5강 Prompt-only vs Fine-tuned Model 결과 비교(선택 정리 필요)
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '8-5강 Prompt-only vs Fine-tuned Model 결과 비교' 정리
---

## 1. 비교 실험의 원칙

Prompt-only와 Fine-tuned 모델을 비교할 때는 반드시 조건을 맞춰야 한다

평가셋이 다르거나, 후처리 규칙이 다르거나, metric이 다르면 결과를 해석할 수 없다.

Prompt-only vs Fine-tuned 비교 설계

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_comparison_design.png' | relative_url }}" alt="01_comparison_design.png" loading="lazy">

label = 클래스 이름 자체

gold = 그 샘플의 정답으로 지정된 label



공정 비교를 위한 조건

| 조건 | 이유 |
| --- | --- |
| 같은 평가셋 | 샘플 난이도를 동일하게 맞춥니다. |
| 같은 정답 라벨 | 라벨 기준이 달라지면 비교가 불가능합니다. |
| 같은 metric | 성능 해석 기준을 통일합니다. |
| 같은 parsing 규칙 | 출력 형식 실패를 같은 기준으로 처리합니다. |
| 같은 기록 방식 | 재현성과 리포트 작성이 쉬워집니다. |


## 2. 비교할 지표 정하기

성능 비교에는 여러 지표가 필요하다

!02_metric_panel.png

비교에 사용할 지표

- **Accuracy**: 전체 샘플 중 맞힌 비율이다.
- **Macro-F1**: 라벨별 F1을 평균합니다. 라벨 불균형이 있을 때 Accuracy보다 유용할 수 있다.
- **Format success**: 출력이 요구한 형식을 지킨 비율이다.
- **Latency**: 응답까지 걸리는 시간
- **Cost**: API 비용, GPU 비용, 운영 비용을 포함한다.
- **Qualitative review**: 실제 출력을 사람이 읽고 품질과 위험을 확인한다.

## 3. 결과표 만들기

```python
# ============================================================
# Prompt-only vs Fine-tuned 결과 비교 예시
# ============================================================
# 실제 모델을 호출하지 않고, 이미 저장된 예측 결과가 있다고 가정합니다.
# 목적은 같은 평가셋에서 두 접근을 어떻게 비교하는지 확인하는 것입니다.

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 평가셋의 정답 라벨입니다.
# 실제 프로젝트에서는 CSV 또는 Dataset에서 gold label을 읽어옵니다.
gold = ["refund", "delivery", "account", "refund", "other", "delivery"]

# Prompt-only baseline의 예측입니다.
# 일부 샘플은 형식을 못 지켜 None으로 기록되었다고 가정합니다.
prompt_pred = ["refund", "other", "account", None, "other", "delivery"]

# Fine-tuned 또는 PEFT 모델의 예측입니다.
# label-only 출력이라고 가정합니다.
finetuned_pred = ["refund", "delivery", "account", "refund", "other", "other"]

# 비교를 위해 DataFrame으로 묶습니다.
# 행 하나는 같은 입력 샘플에 대한 두 모델의 결과를 의미합니다.
compare_df = pd.DataFrame({
    "id": range(1, len(gold) + 1),
    "gold": gold,
    "prompt_pred": prompt_pred,
    "finetuned_pred": finetuned_pred,
})

# prompt_pred가 None이면 format 실패로 봅니다.
compare_df["prompt_format_success"] = compare_df["prompt_pred"].notna()
compare_df["prompt_correct"] = compare_df["prompt_pred"] == compare_df["gold"]
compare_df["finetuned_correct"] = compare_df["finetuned_pred"] == compare_df["gold"]

# sklearn metric은 None 값을 라벨처럼 처리할 수 있으므로,
# metric 계산에서는 parsing 성공 샘플만 보거나 None을 별도 라벨로 처리하는 정책을 명확히 해야 합니다.
parsed_rows = compare_df[compare_df["prompt_format_success"]]

prompt_accuracy = accuracy_score(parsed_rows["gold"], parsed_rows["prompt_pred"])
finetuned_accuracy = accuracy_score(compare_df["gold"], compare_df["finetuned_pred"])

prompt_macro_f1 = f1_score(parsed_rows["gold"], parsed_rows["prompt_pred"], average="macro")
finetuned_macro_f1 = f1_score(compare_df["gold"], compare_df["finetuned_pred"], average="macro")

summary = pd.DataFrame([
    {
        "method": "prompt_only",
        "accuracy": prompt_accuracy,
        "macro_f1": prompt_macro_f1,
        "format_success": compare_df["prompt_format_success"].mean(),
    },
    {
        "method": "fine_tuned_or_peft",
        "accuracy": finetuned_accuracy,
        "macro_f1": finetuned_macro_f1,
        "format_success": 1.0,
    },
])

display(compare_df)
display(summary)
```

출력

```
### 샘플별 비교 결과

| id | gold | prompt_pred | finetuned_pred | prompt_format_success | prompt_correct | finetuned_correct |
|---:|---|---|---|---|---|---|
| 1 | refund | refund | refund | True | True | True |
| 2 | delivery | other | delivery | True | False | True |
| 3 | account | account | account | True | True | True |
| 4 | refund | None | refund | False | False | True |
| 5 | other | other | other | True | True | True |
| 6 | delivery | delivery | other | True | True | False |

### 방법별 성능 비교

| method | accuracy | macro_f1 | format_success |
|---|---:|---:|---:|
| prompt_only | 0.800000 | 0.833333 | 0.833333 |
| fine_tuned_or_peft | 0.833333 | 0.833333 | 1.000000 |
```

### 코드 해설

이 코드는 실제 모델 성능을 주장하는 코드가 아니라 비교 결과표를 구성하는 방식을 보여준다

Prompt-only는 출력 형식 실패가 있을 수 있으므로 format_success를 따로 기록한다

Fine-tuned 모델도 실제 운영에서는 잘못된 라벨 id, label mapping 오류, confidence 해석 오류가 있을 수 있으므로 후처리 규칙을 확인해야 한다

## 4. 오류 분석과 regression 확인

숫자 지표가 좋아져도 일부 중요한 샘플에서 성능이 나빠질 수 있습니다. 이를 regression이라고 한다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_error_analysis_loop.png' | relative_url }}" alt="03_error_analysis_loop.png" loading="lazy">

오류 분석 루프

```python
# ============================================================
# 개선 사례와 regression 사례 찾기
# ============================================================
# 개선 사례: Prompt-only는 틀렸지만 Fine-tuned 모델은 맞힌 샘플
# Regression 사례: Prompt-only는 맞혔지만 Fine-tuned 모델은 틀린 샘플

improved = compare_df[
    (compare_df["prompt_correct"] == False) &
    (compare_df["finetuned_correct"] == True)
]

regressed = compare_df[
    (compare_df["prompt_correct"] == True) &
    (compare_df["finetuned_correct"] == False)
]

print("개선 사례 수:", len(improved))
display(improved)

print("Regression 사례 수:", len(regressed))
display(regressed)
```

### 코드 해설

모델을 교체할 때는 평균 성능뿐 아니라 regression 사례를 확인해야 한다. 

특히 고객 문의, 의료, 법률, 보안처럼 특정 오류가 큰 리스크를 만들 수 있는 경우에는
<br>
 평균 지표가 조금 좋아져도 중요한 클래스의 recall이 낮아지면 배포하면 안 된다

 ## 선택 · 5. 비교 리포트 작성

<img src="{{ '/assets/images/uploads/deep-learning-advanced/04_report_structure.png' | relative_url }}" alt="04_report_structure.png" loading="lazy">

최종 비교 리포트 구조

비교 리포트에는 다음 항목을 포함하세요.

1. 문제 정의: 입력, 출력, 라벨 목록, 성공 기준

2. 실험 조건: 모델, prompt 버전, 데이터 split, seed, 후처리 규칙

3. 정량 결과: Accuracy, Macro-F1, format success, latency, 비용

4. 정성 결과: 좋은 예측, 나쁜 예측, 애매한 사례

5. 오류 분석: 오류 유형과 원인

6. 결론: Prompt-only 유지, PEFT 적용, 데이터 보강, Fine-tuning 확대 중 선택

7. 한계: 데이터 크기, 라벨 품질, 최신성, 안전성


## 8. 생성 모델 평가와 Release Gate (선택 학습)
