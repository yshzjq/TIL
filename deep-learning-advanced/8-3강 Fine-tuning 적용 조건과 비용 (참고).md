---
title: 8-3강 Fine-tuning 적용 조건과 비용 (참고)
date: 2026-09-09
updated: 2026-09-09
description: KANT 강의 '8-3강 Fine-tuning 적용 조건과 비용 (참고)' 정리
---

## 1. Fine-tuning을 너무 빨리 선택하면 생기는 문제

Fine-tuning은 강력한 방법이지만 항상 첫 번째 선택지는 아니다.

데이터가 부족하거나 라벨 기준이 불명확한 상태에서 학습을 시작하면,<br> 모델은 잘못된 기준을 더 잘 외우게 될 수 있다.

또한 평가셋이 없으면 학습 후 모델이 실제로 개선되었는지 알 수 없습니다.


예를 들어 고객 문의 분류에서 배송 지연과 배송 조회의 라벨 기준이 불명확하다면 Fine-tuning으로 해결하기 어렵다.

먼저 라벨 정책을 고치고, 예시를 정리하고, Prompt-only baseline으로 오류 유형을 확인해야 한다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/01_when_finetune.png' | relative_url }}" alt="01_when_finetune.png" loading="lazy">

## 2. Fine-tuning이 필요한 신호

| 신호 | 설명 | 먼저 확인할 것 |
| --- | --- | --- |
| 반복 오류 | 특정 라벨이나 형식에서 계속 실패합니다. | 오류 유형이 평가셋에서 반복되는가 |
| 도메인 행동 | 회사만의 분류 기준이나 답변 스타일이 필요합니다. | 기준 문서와 라벨 예시가 있는가 |
| 형식 안정성 | 출력 형식이 자주 깨져 후처리가 어렵습니다. | Prompt와 schema 검증으로 해결 가능한가 |
| 충분한 데이터 | 검수된 train/valid/test가 있습니다. | 중복, leakage, 라벨 품질을 확인했는가 |

최신 지식이 없어서 틀리는 문제는 Fine-tuning보다 RAG가 먼저일 수 있다.

Fine-tuning은 모델이 새 지식을 정확히 검색하게 만드는 방법이 아니다

## 3. 비용을 네 층으로 나누어 보기

Fine-tuning 비용은 GPU 사용량만 의미하지 않는다

실제 프로젝트에서는 데이터 비용과 평가 비용이 더 크게 느껴질 수 있다.


<img src="{{ '/assets/images/uploads/deep-learning-advanced/02_cost_layers.png' | relative_url }}" alt="02_cost_layers.png" loading="lazy">

Fine-tuning 비용의 층

| 비용 층 | 포함되는 작업 | 놓치기 쉬운 점 |
| --- | --- | --- |
| 데이터 비용 | 수집, 라벨링, 검수, 중복 제거 | 라벨 기준이 흔들리면 학습 결과도 흔들립니다. |
| 학습 비용 | GPU, 시간, 실험 반복, checkpoint 저장 | 한 번에 끝나지 않고 여러 실험이 필요합니다. |
| 평가 비용 | metric, 오류 분석, 사람 검토 | 자동 metric만으로는 충분하지 않을 수 있습니다. |
| 운영 비용 | 배포, 모니터링, 버전 관리, 재학습 | 모델이 바뀌면 추론 환경도 다시 검증해야 합니다. |

## 4. RAG, Prompt, PEFT, Fine-tuning 선택

문제 원인에 따라 선택지가 달라진다

<img src="{{ '/assets/images/uploads/deep-learning-advanced/03_rag_prompt_finetune.png' | relative_url }}" alt="03_rag_prompt_finetune.png" loading="lazy">

- **지식 부족**: 모델이 회사 내부 문서나 최신 정보를 몰라서 틀린다면 RAG를 우선 검토한다
- **출력 형식 문제**: JSON 형식이나 label-only 규칙을 어긴다면 Prompt와 schema 검증을 먼저 개선한다
- **반복되는 판단 기준 문제**: 도메인 라벨 기준이나 스타일을 계속 틀린다면 PEFT/Fine-tuning을 검토한다
- **자원 제약**: Full Fine-tuning이 부담되면 LoRA 같은 PEFT부터 시작한다

schema(스키마) 는 쉽게 말하면 출력 데이터가 어떤 구조와 규칙을 가져야 하는지 정해놓은 형식

Schema = 데이터의 구조와 형식을 정해둔 규칙표입니다.

## 5. 의사결정 점수표 만들기

다음 코드는 Fine-tuning 의사결정을 돕는 간단한 점수표입니다. 

실제 비용을 계산하는 모델이 아니라, 학습 전에 필요한 조건이 얼마나 준비되었는지 점검하기 위한 도구다

```python
# ============================================================
# Fine-tuning 적용 조건 점수표
# ============================================================
# 이 코드는 실제 비용을 원 단위로 계산하지 않습니다.
# 목적은 Fine-tuning 전에 준비 상태를 구조적으로 점검하는 것입니다.

import pandas as pd

criteria = [
    {
        "criterion": "문제 정의가 명확한가",
        "description": "입력, 출력, 라벨, 성공 기준이 문서화되어 있는지 확인합니다.",
        "score": 4,
    },
    {
        "criterion": "검수된 데이터가 충분한가",
        "description": "train/valid/test split과 라벨 품질 검수가 되어 있는지 확인합니다.",
        "score": 3,
    },
    {
        "criterion": "Prompt-only baseline이 있는가",
        "description": "학습 전 기준 성능과 오류 유형을 알고 있는지 확인합니다.",
        "score": 5,
    },
    {
        "criterion": "평가 metric이 정해졌는가",
        "description": "Accuracy, Macro-F1, format success 등 개선 기준이 있는지 확인합니다.",
        "score": 4,
    },
    {
        "criterion": "운영 계획이 있는가",
        "description": "저장, 배포, rollback, 모니터링 계획이 있는지 확인합니다.",
        "score": 2,
    },
]

df = pd.DataFrame(criteria)

# 평균 점수는 의사결정의 보조 지표입니다.
# 점수가 낮은 항목은 Fine-tuning 전에 보완해야 할 위험 신호입니다.
average_score = df["score"].mean()

display(df)
print("준비도 평균 점수:", round(average_score, 2))

if average_score < 3.5:
    print("권장: Fine-tuning 전에 데이터/평가/운영 계획을 보완하세요.")
else:
    print("권장: PEFT 또는 작은 실험부터 시작할 수 있습니다.")
```

| criterion | description | score |
|---|---|---:|
| 문제 정의가 명확한가 | 입력, 출력, 라벨, 성공 기준이 문서화되어 있는지 확인합니다. | 4 |
| 검수된 데이터가 충분한가 | train/valid/test split과 라벨 품질 검수가 되어 있는지 확인합니다. | 3 |
| Prompt-only baseline이 있는가 | 학습 전 기준 성능과 오류 유형을 알고 있는지 확인합니다. | 5 |
| 평가 metric이 정해졌는가 | Accuracy, Macro-F1, format success 등 개선 기준이 있는지 확인합니다. | 4 |
| 운영 계획이 있는가 | 저장, 배포, rollback, 모니터링 계획이 있는지 확인합니다. | 2 |

출력

```
준비도 평균 점수: 3.6
권장: PEFT 또는 작은 실험부터 시작할 수 있습니다.
```

