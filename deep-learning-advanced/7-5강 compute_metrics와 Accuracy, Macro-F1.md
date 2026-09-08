---
title: 7-5강 compute_metrics와 Accuracy, Macro-F1
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-5강 compute_metrics와 Accuracy, Macro-F1' 정리
---

## 1. Accuracy만으로 충분하지 않은 이유

Accuracy는 전체 샘플 중 맞힌 비율입니다. 직관적이지만, 라벨 불균형이 있을 때는 한계가  크다. 

예를 들어 전체 샘플의 90%가 `배송 문의`라면, 모델이 모든 샘플을 `배송 문의`로 예측해도 Accuracy는 90%가 될 수 있다.


<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_metric_choice.png
' | relative_url }}" alt="01_metric_choice.png
" loading="lazy">

비즈니스 문제에서는 특정 소수 클래스가 더 중요할 수 있다.
<br>
예를 들어 개인정보 포함 여부, 안전하지 않은 답변, 환불 요청 같은 라벨은 샘플 수가 적어도 놓치면 큰 문제가 될 수 있다.

비즈니스에서는 샘플 수가 적은 클래스라도 놓쳤을 때 위험이나 비용이 크다면 그 클래스의 성능을 더 중요하게 봐야 한다.

## 2. Precision, Recall, F1 직관

Precision은 “모델이 해당 클래스로 예측한 것 중 얼마나 맞았는가”.

Recall은 “실제 해당 클래스 중 얼마나 찾아냈는가”입니다. 

F1은 Precision과 Recall의 균형을 봅니다.

| 지표 | 질문 | 중요한 상황 |
| --- | --- | --- |
| Precision | 모델이 이 클래스라고 한 것들이 얼마나 정확한가요? | 잘못된 경고를 줄이고 싶을 때 |
| Recall | 실제 이 클래스인 것들을 얼마나 놓치지 않았나요? | 위험 사례를 놓치면 안 될 때 |
| F1 | Precision과 Recall의 균형은 어떤가요? | 둘 다 중요할 때 |

## 3. Macro-F1이 필요한 상황

Macro-F1은 각 클래스별 F1을 계산한 뒤 단순 평균<br>따라서 샘플 수가 적은 클래스도 같은 비중으로 반영된다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_macro_f1.png
' | relative_url }}" alt="02_macro_f1.png
" loading="lazy">

Macro-F1 직관

Macro-F1이 항상 최종 목표라는 뜻이 아니다

문제에 따라 특정 클래스 Recall을 더 중요하게 보거나, 비용 기반 metric을 따로 설계할 수도 있다.

## 4. compute_metrics 함수 작성

```python
# ============================================================
# Trainer에 전달할 compute_metrics 함수
# ============================================================
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# compute_metrics는 Trainer가 evaluation 또는 prediction 단계에서 호출하는 함수입니다.
# eval_pred는 보통 predictions와 label_ids를 담고 있습니다.
def compute_metrics(eval_pred):
    # predictions는 모델 출력 logits입니다.
    # 일반적인 sequence classification에서는 shape이 [B, C]입니다.
    predictions, labels = eval_pred

    # Trainer 설정이나 모델에 따라 predictions가 tuple로 오는 경우가 있습니다.
    # 이때 첫 번째 요소가 logits인 경우가 많으므로 안전하게 처리합니다.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # logits에서 가장 큰 클래스 index를 예측 label로 선택합니다.
    # axis=-1은 클래스 축 C에서 argmax를 수행한다는 뜻입니다.
    preds = np.argmax(predictions, axis=-1)

    # Accuracy는 전체 샘플 중 맞힌 비율입니다.
    accuracy = accuracy_score(labels, preds)

    # Macro-F1은 클래스별 F1을 단순 평균합니다.
    # zero_division=0은 특정 클래스에 예측이 하나도 없을 때 발생할 수 있는 경고를 줄이고 0으로 처리합니다.
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    # precision_recall_fscore_support는 precision, recall, f1, support를 함께 계산합니다.
    # 여기서는 macro 평균만 리턴하지만, 오류 분석에서는 클래스별 값을 따로 보는 것이 좋습니다.
    precision, recall, _, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    # Trainer는 이 dict를 evaluation log로 기록합니다.
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }
```

### 코드 해설

evaluation : 평가

Trainer는 evaluation 중 모델 예측을 모아 compute_metrics`에 전달한다

이 함수는 logits와 labels를 받아 사람이 해석할 수 있는 metric dict를 반환한다
<br>
반환된 key는 로그와 checkpoint 선택 기준에 사용할 수 있다.

## 5. Confusion matrix로 오류 유형 보기



<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_confusion_matrix.png
' | relative_url }}" alt="03_confusion_matrix.png
" loading="lazy">

Confusion Matrix 읽기

```python
# ============================================================
# Confusion matrix와 classification report 예시
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

# 예시용 labels와 predictions입니다.
# 실제 학습 후에는 trainer.predict(test_dataset)의 결과를 사용합니다.
true_labels = np.array([0, 0, 1, 1, 2, 2, 2])
pred_labels = np.array([0, 1, 1, 1, 2, 0, 2])

# classification_report는 클래스별 precision, recall, f1-score, support를 표 형태로 보여줍니다.
print(classification_report(
    true_labels,
    pred_labels,
    target_names=["class_0", "class_1", "class_2"],
    zero_division=0,
))

# confusion_matrix의 행은 실제 label, 열은 예측 label입니다.
cm = confusion_matrix(true_labels, pred_labels)
print("confusion matrix:\n", cm)
```

출력

```
              precision    recall  f1-score   support

     class_0       0.50      0.50      0.50         2
     class_1       0.67      1.00      0.80         2
     class_2       1.00      0.67      0.80         3

    accuracy                           0.71         7
   macro avg       0.72      0.72      0.70         7
weighted avg       0.76      0.71      0.71         7

confusion matrix:
 [[1 1 0]
 [0 2 0]
 [1 0 2]]
```

실제 class_0 → class_0으로 예측: 1개  ← 맞음
실제 class_0 → class_1으로 예측: 1개  ← 틀림
실제 class_0 → class_2로 예측: 0개

- 행(row) = 실제 정답
- 열(column) = 모델의 예측

### 코드 해설

Confusion matrix는 “실제 정답이 무엇이었는데, 모델이 무엇이라고 예측했는지”를 표로 나타낸 것이다

Confusion matrix에서 대각선은 맞힌 샘플, 비대각선은 틀린 샘플입니다. 예를 들어 행 2, 열 0 값이 크다면 실제 class_2를 class_0으로 자주 헷갈린다는 뜻이다


## 선택 · 6. 연습 문제

1. Accuracy가 높아도 Macro-F1이 낮을 수 있는 상황을 설명

2. Precision과 Recall의 차이를 고객 문의 분류 예시로 설명

3. `compute_metrics`에서 `np.argmax(predictions, axis=-1)`을 사용하는 이유를 설명

4. Confusion matrix에서 비대각선 값이 의미하는 것을 설명

---

## 참고 · 7. 정답 확인

1. 라벨 불균형이 심하면 다수 클래스를 많이 맞혀 Accuracy는 높지만, 소수 클래스 성능이 낮아 Macro-F1은 낮을 수 있다.

2. 예를 들어 `환불 요청` 라벨에서 Precision은 모델이 환불이라고 예측한 문의 중 실제 환불 문의 비율<br> Recall은 실제 환불 문의 중 모델이 환불이라고 찾아낸 비율

3. 모델 출력 logits에서 클래스 축의 가장 큰 점수를 가진 index를 예측 label로 선택하기 위해 사용한다.

4. 비대각선 값은 실제 label과 예측 label이 다른 오류 샘플 수입니다. 어떤 클래스 쌍을 자주 헷갈리는지 
보여준다


np.argmax(predictions, axis=-1)은 각 샘플의 클래스 점수 중 가장 큰 점수를 가진 클래스 ID를 최종 예측값으로 선택


---
