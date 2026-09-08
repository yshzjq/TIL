---
title: 7-7강 Fine-tuning 실행·평가·Checkpoint 관리(출력 결과 추가 예정)
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-7강 Fine-tuning 실행·평가·Checkpoint 관리' 정리
---

## 1. Fine-tuning 실행 전 확인


Fine-tuning은 사전학습 모델을 특정 라벨 분류 문제에 맞게 업데이트하는 과정이다.

실행 전에는 데이터 컬럼, 모델 라벨 수, metric key, 저장 경로, GPU 메모리를 확인해야 한다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_training_loop.png
' | relative_url }}" alt="001_training_loop.png
" loading="lazy">

Fine-tuning 실행 흐름

```python
# ============================================================
# 실행 전 최종 점검
# ============================================================
# tokenized_dataset의 컬럼이 Trainer가 기대하는 형태인지 확인합니다.
print("train columns:", tokenized_dataset["train"].column_names)
print("validation columns:", tokenized_dataset["validation"].column_names)

# 모델의 라벨 수와 데이터셋 라벨 수가 일치하는지 확인합니다.
assert model.config.num_labels == num_labels

# metric_for_best_model이 compute_metrics 반환 key와 일치해야 합니다.
# 여기서는 macro_f1을 best model 선택 기준으로 사용합니다.
print("best model metric:", training_args.metric_for_best_model)

# GPU 사용 여부를 다시 확인합니다.
print("device:", device)
```

### 코드 해설

실행 전 점검은 시간을 아끼는 단계입니다. 

Fine-tuning은 몇 분에서 몇 시간까지 걸릴 수 있으므로, 
<br>
컬럼명이나 라벨 수 같은 기본 오류를 먼저 잡는 것이 중요하다

## 2. trainer.train() 실행

```python
# ============================================================
# Fine-tuning 실행
# ============================================================
# trainer.train()은 내부적으로 다음 과정을 반복합니다.
# 1. train_dataset에서 batch를 만듭니다.
# 2. model(**batch)로 forward pass를 실행합니다.
# 3. outputs.loss로 loss를 계산합니다.
# 4. loss.backward()로 gradient를 계산합니다.
# 5. optimizer.step()으로 파라미터를 업데이트합니다.
# 6. 설정된 시점마다 evaluation과 checkpoint 저장을 수행합니다.

train_result = trainer.train()

# train_result에는 학습 시간, step 수, train loss 등 요약 정보가 들어 있습니다.
print(train_result)
```

출력

```

```

### 코드 해설

Trainer가 학습 루프를 대신 실행하지만, 내부 원리는 PyTorch train loop와 같습니다.

`forward → loss → backward → optimizer.step()` 흐름이 그대로 들어 있다.

Colab에서 OOM이 발생하면 batch size를 줄이고, max_length를 줄이고, 필요하면 런타임을 재시작해 GPU 메모리를 비운 뒤 다시 실행

## 3. evaluation 결과 확인

```python
# ============================================================
# validation 평가
# ============================================================
# evaluate()는 eval_dataset에 대해 모델 예측을 만들고 compute_metrics를 호출합니다.
validation_metrics = trainer.evaluate()
print("validation metrics:")
print(validation_metrics)

# metrics를 JSON 파일로 저장합니다.
# 실험 결과를 노트북 출력에만 남기면 나중에 비교하기 어렵습니다.
metrics_path = PROJECT_DIR / "validation_metrics.json"
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(validation_metrics, f, ensure_ascii=False, indent=2)
print("저장 경로:", metrics_path)
```

출력

```

```

## 4. checkpoint와 best model 관리



<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_checkpoint_lifecycle.png
' | relative_url }}" alt="02_checkpoint_lifecycle.png
" loading="lazy">

Checkpoint lifecycle

```python
# ============================================================
# checkpoint 폴더 확인
# ============================================================
# output_dir 아래에는 checkpoint-숫자 형태의 폴더가 생길 수 있습니다.
checkpoint_root = Path(training_args.output_dir)
checkpoints = sorted(checkpoint_root.glob("checkpoint-*"))

print("checkpoint root:", checkpoint_root)
print("checkpoint 개수:", len(checkpoints))
for ckpt in checkpoints:
    print("-", ckpt.name)

# Trainer state에는 best_model_checkpoint 정보가 저장될 수 있습니다.
# load_best_model_at_end=True를 사용하면 학습 종료 후 trainer.model이 best model로 로드됩니다.
print("best model checkpoint:", trainer.state.best_model_checkpoint)
```
출력
```

```

### 코드 해설

checkpoint는 중간 저장 상태입니다. 

best checkpoint는 검증 metric 기준으로 선택된 중간 상태입니다. 

final model은 추론과 제출을 위해 따로 저장하는 산출물입니다.


## 5. 최종 모델과 tokenizer 저장


<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_save_reload.png
' | relative_url }}" alt="03_save_reload.png
" loading="lazy">

저장과 재로드

```python
# ============================================================
# 최종 모델과 tokenizer 저장
# ============================================================
final_model_dir = PROJECT_DIR / "final_model"
final_model_dir.mkdir(parents=True, exist_ok=True)

# trainer.save_model은 현재 trainer.model을 Hugging Face 형식으로 저장합니다.
# load_best_model_at_end=True였다면, 일반적으로 best model 상태가 저장됩니다.
trainer.save_model(final_model_dir)

# tokenizer도 같은 폴더에 저장해야 합니다.
# 모델만 저장하고 tokenizer를 저장하지 않으면 추론 시 tokenization 규칙이 달라질 수 있습니다.
tokenizer.save_pretrained(final_model_dir)

# label mapping과 주요 설정도 함께 저장합니다.
run_config = {
    "dataset_id": DATASET_ID,
    "dataset_config": DATASET_CONFIG,
    "model_id": MODEL_ID,
    "text_col": TEXT_COL,
    "label_col": LABEL_COL,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "id2label": id2label,
    "label2id": label2id,
    "best_model_checkpoint": trainer.state.best_model_checkpoint,
}

with (final_model_dir / "run_config.json").open("w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

print("final model 저장 경로:", final_model_dir)
print("저장 파일:", sorted(path.name for path in final_model_dir.iterdir()))
```

### 코드 해설

모델, tokenizer, config, label mapping, metric이 함께 있어야 추론과 재현이 가능하다.

실제 프로젝트에서는 여기에 model card와 data card도 추가하는 것이 좋다.

## 선택 · 6. 연습 문제

1. checkpoint와 final model의 차이를 설명

2. `load_best_model_at_end=True`가 필요한 이유를 설명

3. 모델 저장 시 tokenizer를 함께 저장해야 하는 이유

4. Fine-tuning 중 validation loss는 좋아지는데 test 성능이 낮다면 어떤 가능성을 의심해야 하나요?

---

## 참고 · 7. 정답 확인

1. checkpoint는 학습 중간 상태 저장본이고, final model은 추론·제출·공유를 위해 최종적으로 정리해 저장한 모델

2. 학습 마지막 epoch의 모델이 항상 가장 좋은 모델은 아닐 수 있습니다. 검증 metric 기준으로 가장 좋은 checkpoint를 최종 모델로 사용하기 위해 필요합니다.

3. tokenizer의 vocab, special token, truncation/padding 규칙이 모델 입력과 맞아야 하므로 함께 저장한다

4. validation에 과도하게 맞춘 과적합, test 분포 차이, data leakage, metric 설정 오류, 라벨 품질 문제를 의심할 수 있다.

## 8. 학습이 실패할 때 진단 순서

| 증상 | 먼저 확인할 것 | 대표 조치 |
| --- | --- | --- |
| Train loss가 거의 줄지 않음 | label 분포·mapping, 학습 대상 parameter, learning rate, batch 입력 | 작은 데이터 overfit 시험, gradient norm 확인 |
| Train은 좋아지고 validation은 악화 | split 누수 반대편의 과적합, class별 성능 | early stopping, 데이터·정규화 점검 |
| Loss가 NaN/Inf | 입력·label 범위, learning rate, mixed precision, gradient | 문제 batch 격리, LR 하향, gradient clipping |
| CUDA OOM | sequence length, batch, activation, optimizer state | batch 축소, gradient accumulation/checkpointing, 짧은 길이 |
| Metric이 비정상적으로 높음 | 중복·누수, test 사용, label 순서 | group/time split 재검토, test 격리 |

가장 빠른 sanity check는 아주 작은 샘플을 의도적으로 overfit하는 것이다
 
이것조차 실패하면 데이터·loss·gradient·optimizer 연결을 먼저 의심한다

반대로 작은 샘플만 외우고 validation이 나쁘다면 구현보다 일반화와 데이터 문제를 본다

일반화는 학습할 때 보지 못한 새로운 데이터에서도 잘 맞히는 능력


기록할 최소 항목:

- step별 train/eval loss와 learning rate
- gradient norm, 최대 GPU memory
- best checkpoint 기준과 저장 경로
- 중단된 step, seed, 문제 batch ID

## 9. 이번 강의 요약

