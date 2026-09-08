---
title: 7-8강 Error Analysis와 Fine-tuning 리포트(출력 추가 예정)
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-8강 Error Analysis와 Fine-tuning 리포트' 정리
---

## 1. Error Analysis가 필요한 이유

Metric은 모델 성능을 숫자로 요약하지만, 숫자만으로는 무엇을 고쳐야 하는지 알기 어렵다.

Error Analysis는 틀린 샘플을 직접 확인해 라벨 경계, 데이터 품질, 모델 한계, 전처리 문제를 찾는 과정

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_error_analysis.png
' | relative_url }}" alt="01_error_analysis.png
" loading="lazy">

Error Analysis 흐름

성능 개선은 “모델을 더 크게 바꾸기”보다 “어떤 오류가 반복되는지 확인하기”에서 시작하는 경우가 많다.

## 2. test prediction 생성

```python
# ============================================================
# test split 예측 생성
# ============================================================
# trainer.predict는 주어진 dataset에 대해 logits, labels, metrics를 반환합니다.
# 여기서는 최종 평가용 test split에 대해 예측을 수행합니다.
test_output = trainer.predict(tokenized_dataset["test"])

# predictions는 logits입니다. 일반적인 shape은 [B, C]입니다.
test_logits = test_output.predictions

# label_ids는 정답 label id입니다. shape은 [B]입니다.
test_label_ids = test_output.label_ids

# metrics에는 test loss와 compute_metrics 결과가 포함될 수 있습니다.
test_metrics = test_output.metrics

print("logits shape:", test_logits.shape)
print("labels shape:", test_label_ids.shape)
print("test metrics:", test_metrics)
```
출력

```

```

### 코드 해설

`trainer.predict()`는 학습을 하지 않습니다.

모델 파라미터를 업데이트하지 않고, 주어진 데이터에 대한 예측과 metric을 계산한다


## 3. prediction DataFrame 만들기

Confidence 기반 검토

```python
# ============================================================
# prediction 결과를 사람이 읽는 표로 만들기
# ============================================================
# logits를 softmax로 변환해 클래스별 확률처럼 해석할 수 있는 값을 만듭니다.
# 여기서는 numpy로 계산합니다.
exp_logits = np.exp(test_logits - test_logits.max(axis=-1, keepdims=True))
probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

# 가장 높은 확률의 class id를 예측 id로 사용합니다.
pred_ids = probs.argmax(axis=-1)

# 예측 confidence는 선택된 class의 확률값입니다.
# confidence가 높다고 항상 맞는 것은 아니지만, 오류 우선순위 분석에 유용합니다.
confidence = probs.max(axis=-1)

# 원본 test 데이터에서 title과 정답 label을 가져옵니다.
# tokenized_dataset에서는 title을 제거했을 수 있으므로 dataset 원본을 사용합니다.
test_texts = dataset["test"][TEXT_COL]

prediction_df = pd.DataFrame({
    "text": test_texts,
    "true_id": test_label_ids,
    "pred_id": pred_ids,
    "confidence": confidence,
})

# 사람이 읽는 라벨 이름을 추가합니다.
prediction_df["true_label"] = prediction_df["true_id"].map(lambda x: id2label[int(x)])
prediction_df["pred_label"] = prediction_df["pred_id"].map(lambda x: id2label[int(x)])
prediction_df["is_correct"] = prediction_df["true_id"] == prediction_df["pred_id"]

# prediction 파일 저장
prediction_path = PROJECT_DIR / "test_predictions.csv"
prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

print("prediction 저장 경로:", prediction_path)
display(prediction_df.head())
```

### 코드 해설

prediction 파일은 error analysis의 기본 재료입니다. 

`text`, `true_label`, `pred_label`, `confidence`, `is_correct`가 있으면 수동 검토와 보고서 작성이 쉬워진다

## 4. classification report와 confusion matrix

```python
# ============================================================
# classification report와 confusion matrix 생성
# ============================================================
from sklearn.metrics import classification_report, confusion_matrix

report_text = classification_report(
    prediction_df["true_id"],
    prediction_df["pred_id"],
    target_names=label_names,
    zero_division=0,
)

cm = confusion_matrix(
    prediction_df["true_id"],
    prediction_df["pred_id"],
)

print(report_text)
print("confusion matrix:\n", cm)

# 텍스트 리포트와 confusion matrix를 파일로 저장합니다.
(PROJECT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
np.save(PROJECT_DIR / "confusion_matrix.npy", cm)
```

### 코드 해설

classification report는 클래스별 precision, recall, F1, support를 보여준다.

confusion matrix는 어떤 클래스 쌍을 헷갈리는지 보여줍니다. 

이 두 정보를 함께 보면 metric 숫자를 구체적인 오류 패턴으로 바꿀 수 있다.

## 5. 오류 샘플 해석과 개선안

```python
# ============================================================
# 오류 샘플 추출
# ============================================================
# 틀린 샘플만 필터링합니다.
errors = prediction_df[~prediction_df["is_correct"]].copy()

# confidence가 높은 오류는 모델이 강하게 확신했지만 틀린 사례입니다.
# 이런 오류는 라벨 경계 문제, 데이터 편향, 모델이 잘못 배운 패턴을 드러낼 수 있습니다.
high_conf_errors = errors.sort_values("confidence", ascending=False).head(20)

# confidence가 낮은 오류는 모델도 애매하게 판단한 사례일 수 있습니다.
low_conf_errors = errors.sort_values("confidence", ascending=True).head(20)

print("전체 오류 수:", len(errors))
print("높은 확신 오류 예시")
display(high_conf_errors[["text", "true_label", "pred_label", "confidence"]])

print("낮은 확신 오류 예시")
display(low_conf_errors[["text", "true_label", "pred_label", "confidence"]])
```


### 코드 해설

높은 확신 오류는 모델이 잘못된 규칙을 강하게 배웠을 가능성을 보여준다.

낮은 확신 오류는 문장이 애매하거나 라벨 기준이 불명확할 가능성을 보여준다

두 유형을 나눠 보면 개선 방향을 더 쉽게 찾을 수 있다.

## 선택 · 6. Fine-tuning 리포트 템플릿



<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_report_structure.png
' | relative_url }}" alt="03_report_structure.png
" loading="lazy">


Fine-tuning 리포트 구조

```python
# ============================================================
# Fine-tuning 리포트 markdown 생성
# ============================================================
report_md = f"""
# YNAT Text Classification Fine-tuning Report

## 1. 문제 정의

- 입력: 한국어 뉴스 제목
- 출력: 뉴스 topic label
- 성공 기준: Macro-F1, Accuracy, 클래스별 Recall

## 2. 데이터

- Dataset ID:{DATASET_ID}
- Config:{DATASET_CONFIG}
- Text column:{TEXT_COL}
- Label column:{LABEL_COL}
- Label names:{label_names}
- Split seed:{SEED}

## 3. 모델과 Tokenizer

- Base model:{MODEL_ID}
- Max length:{MAX_LENGTH}
- Task: Sequence Classification

## 4. 학습 설정

- Epoch:{training_args.num_train_epochs}
- Train batch size:{training_args.per_device_train_batch_size}
- Eval batch size:{training_args.per_device_eval_batch_size}
- Learning rate:{training_args.learning_rate}
- Weight decay:{training_args.weight_decay}

## 5. 평가 결과

아래에는 test metric을 JSON 문자열로 기록합니다.

{json.dumps(test_metrics, ensure_ascii=False, indent=2)}

## 6. 오류 분석

- 전체 오류 수:{len(errors)}
- 높은 확신 오류: confidence가 높지만 틀린 샘플을 우선 검토합니다.
- 낮은 확신 오류: 라벨 경계가 애매한 샘플을 검토합니다.

## 7. 한계와 개선안

- 라벨 경계가 애매한 샘플은 라벨 가이드 보완이 필요합니다.
- 소수 클래스 성능이 낮으면 데이터 보강 또는 class-weighted loss를 검토할 수 있습니다.
- 도메인이 바뀌면 같은 모델 성능을 보장할 수 없으므로 추가 평가 데이터가 필요합니다.
"""

report_path = PROJECT_DIR / "fine_tuning_report.md"
report_path.write_text(report_md, encoding="utf-8")
print("리포트 저장 경로:", report_path)
```

## 선택 · 7. 연습 문제

1. Error Analysis가 metric만 보는 것보다 좋은 점

2. 높은 confidence 오류와 낮은 confidence 오류를 각각 해석

3. Fine-tuning 리포트에 반드시 들어가야 할 항목을 5개 이상 적으세요.

4. confusion matrix에서 특정 두 클래스가 자주 헷갈린다면 어떤 개선안을 생각할 수 있나

---

## 참고 · 8. 정답 확인

1. Error Analysis는 어떤 샘플에서 왜 틀리는지 볼 수 있어 구체적인 개선 방향을 찾을 수 있다. 
<br>
Metric은 전체 성능 요약에는 좋지만 오류 원인을 직접 알려주지는 않습니다.

Confidence는 “예측이 맞냐 틀리냐”가 아니라, 모델이 자기 예측을 얼마나 확신하느냐

2. 높은 confidence 오류는 모델이 잘못된 패턴을 강하게 배웠거나 라벨 오류가 있을 수 있음을 의미한다.<br>
 낮은 confidence 오류는 문장이 애매하거나 라벨 경계가 불분명할 수 있음을 의미한다

3. 문제 정의, 데이터 출처와 split, 라벨 규칙, 모델 ID, tokenizer, 학습 설정, metric, checkpoint 경로, 오류 분석, 한계와 개선안이 들어가야 한다.

4. 두 클래스의 라벨 정의를 더 명확히 하고, 헷갈리는 샘플을 추가 검토하거나 데이터 보강을 수행할 수 있습니다. 필요하면 클래스 통합이나 라벨 체계 재설계를 검토할 수도 있다.

---

## 9. Validation·Test·Confidence를 구분해서 보고하기

Validation은 checkpoint와 hyperparameter를 고르는 데 사용한다.

Test는 선택이 끝난 뒤 한 번 평가하는 최종 일반화 자료입니다. Test 결과를 보며 설정을 반복해서 바꾸면 사실상 test가 validation 역할을 하게 된다

Softmax의 가장 큰 값은 모델의 상대 점수이지 자동으로 보정된 신뢰도가 아닙니다. 선택 학습에서는 임계값을 정할 때 Validation에서 다음 항목을 함께 확인한다

- class별 precision·recall·F1
- confusion matrix와 대표 오류
- confidence 구간별 accuracy 또는 calibration 지표
- 임계값을 바꿨을 때 false positive·false negative 비용
- 분포가 달라진 데이터에서의 성능

리포트에는 “최종 test를 언제 몇 번 사용했는지”, “임계값은 어느 split에서 정했는지”, “confidence를 어떻게 해석했는지”를 명시한다