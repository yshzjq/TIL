---
title: 7-6강 TrainingArguments와 Trainer 설정
date: 2026-09-08
updated: 2026-09-08
description: KANT 강의 '7-6강 TrainingArguments와 Trainer 설정' 정리
---

## 1. TrainingArguments는 무엇을 기록하나

TrainingArguments는 학습 실행 조건을 한 곳에 모아둔 설정 객체다.
<br>
batch size, learning rate, epoch, logging 간격, evaluation 간격, checkpoint 저장 정책 같은 값이 여기에 들어갑니다.

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_trainingarguments.png
' | relative_url }}" alt="01_trainingarguments.png
" loading="lazy">

TrainingArguments 설정

TrainingArguments는 단순 코드 옵션이 아니라 **실험 재현성 문서**다

나중에 성능 차이가 생겼을 때 어떤 설정이 달랐는지 추적하는 기준이 된다.

## 2. Trainer는 어떤 부품을 연결하나

Trainer는 지금까지 만든 부품을 하나로 연결한다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\02_trainer_components.png
' | relative_url }}" alt="02_trainer_components.png
" loading="lazy">

Trainer 구성요소

| 구성요소 | 역할 |
| --- | --- |
| `model` | 학습할 Sequence Classification 모델 |
| `args` | TrainingArguments 설정 |
| `train_dataset` | 학습 split |
| `eval_dataset` | 검증 split |
| `processing_class` 또는 `tokenizer` | 저장·재로드 시 전처리 규칙 연결 |
| `data_collator` | batch 생성과 dynamic padding |
| `compute_metrics` | 평가 metric 계산 |


## 3. TrainingArguments 설정 코드

```python
# ============================================================
# TrainingArguments 생성
# ============================================================
from transformers import TrainingArguments
import inspect

# transformers 버전에 따라 evaluation_strategy 대신 eval_strategy를 쓰는 경우가 있습니다.
# 아래 코드는 현재 설치된 TrainingArguments가 어떤 인자 이름을 지원하는지 확인해 호환성을 높입니다.
training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
strategy_arg_name = "eval_strategy" if "eval_strategy" in training_arg_params else "evaluation_strategy"

training_args_kwargs = {
    # 모델, checkpoint, 로그가 저장될 디렉터리입니다.
    "output_dir": str(PROJECT_DIR / "checkpoints"),

    # 학습 epoch 수입니다. 교육용 실습에서는 시간을 줄이기 위해 작게 설정할 수 있습니다.
    "num_train_epochs": 2,

    # GPU 하나에서 한 번에 처리할 학습 batch size입니다.
    # OOM이 나면 이 값을 줄이거나 gradient_accumulation_steps를 사용할 수 있습니다.
    "per_device_train_batch_size": 16,

    # evaluation batch size입니다. 평가에서는 gradient를 저장하지 않아 학습보다 크게 잡을 수도 있습니다.
    "per_device_eval_batch_size": 32,

    # learning rate는 파라미터를 업데이트하는 보폭입니다.
    # Transformer fine-tuning에서는 보통 작은 값을 사용합니다.
    "learning_rate": 2e-5,

    # weight_decay는 가중치가 지나치게 커지는 것을 완화하는 정규화 설정입니다.
    "weight_decay": 0.01,

    # logging_steps마다 loss 등 로그를 출력합니다.
    "logging_steps": 50,

    # checkpoint 저장 개수를 제한해 저장 공간을 관리합니다.
    "save_total_limit": 2,

    # 가장 좋은 모델을 학습 종료 후 로드합니다.
    "load_best_model_at_end": True,

    # 어떤 metric이 가장 좋은 모델을 고르는 기준인지 지정합니다.
    "metric_for_best_model": "macro_f1",

    # metric 값이 클수록 좋은지 여부입니다. macro_f1은 클수록 좋습니다.
    "greater_is_better": True,

    # 리포트 도구를 끕니다. wandb 계정 설정 없이 실습이 바로 실행되도록 하기 위한 설정입니다.
    "report_to": "none",
}

# evaluation과 checkpoint 저장을 epoch마다 수행합니다.
# save_strategy와 evaluation strategy가 맞아야 load_best_model_at_end가 정상 동작합니다.
training_args_kwargs[strategy_arg_name] = "epoch"
training_args_kwargs["save_strategy"] = "epoch"

training_args = TrainingArguments(**training_args_kwargs)
print(training_args)
```

출력

```
TrainingArguments(
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
average_tokens_across_devices=True,
batch_eval_metrics=False,
bf16=False,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
dataloader_prefetch_factor=None,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=None,
ddp_static_graph=None,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=False,
do_eval=True,
do_predict=False,
do_train=False,
enable_jit_checkpoint=False,
eval_accumulation_steps=None,
eval_delay=0,
eval_do_concat_batches=True,
eval_on_start=False,
eval_steps=None,
eval_strategy=IntervalStrategy.EPOCH,
eval_use_gather_object=False,
fp16=False,
fp16_full_eval=False,
fsdp=None,
fsdp_config=None,
full_determinism=False,
gradient_accumulation_steps=1,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=True,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=None,
hub_revision=None,
hub_strategy=HubStrategy.EVERY_SAVE,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_for_metrics=[],
include_num_input_tokens_seen=no,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=2e-05,
length_column_name=length,
liger_kernel_config=None,
load_best_model_at_end=True,
local_rank=-1,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=None,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=50,
logging_strategy=IntervalStrategy.STEPS,
lr_scheduler_kwargs=None,
lr_scheduler_type=SchedulerType.LINEAR,
max_grad_norm=1.0,
max_steps=-1,
metric_for_best_model=macro_f1,
neftune_noise_alpha=None,
num_train_epochs=2,
optim=OptimizerNames.ADAMW_TORCH_FUSED,
optim_args=None,
optim_target_modules=None,
output_dir=artifacts/ynat_finetuning/checkpoints,
parallelism_config=None,
per_device_eval_batch_size=32,
per_device_train_batch_size=16,
prediction_loss_only=False,
project=huggingface,
push_to_hub=False,
remove_unused_columns=True,
report_to=[],
restore_callback_states_from_checkpoint=False,
resume_from_checkpoint=None,
run_name=None,
save_on_each_node=False,
save_only_model=False,
save_steps=500,
save_strategy=SaveStrategy.EPOCH,
save_total_limit=2,
seed=42,
skip_memory_metrics=True,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torch_empty_cache_steps=None,
trackio_bucket_id=None,
trackio_space_id=None,
trackio_static_space_id=None,
train_sampling_strategy=random,
use_cache=False,
use_cpu=False,
use_liger_kernel=False,
warmup_ratio=None,
warmup_steps=0,
weight_decay=0.01,
)
```

### 코드 해설

이 코드는 버전 호환성을 고려해 `eval_strategy`와 `evaluation_strategy` 중 현재 설치된 버전이 지원하는 이름을 선택한다

사소한 버전 차이가 오류로 이어질 수 있으므로 이런 방어 코드가 도움이 된다


## 4. Trainer 생성 코드

<img src="{{ '/assets/images/uploads\deep-learning-advanced\03_logging_eval_save.png
' | relative_url }}" alt="03_logging_eval_save.png
" loading="lazy">

logging·evaluation·save

```python
# ============================================================
# Trainer 생성
# ============================================================
from transformers import Trainer

# transformers 버전에 따라 Trainer가 tokenizer 인자를 deprecated 처리하고
# processing_class를 권장하는 경우가 있습니다.
# 현재 설치된 Trainer가 processing_class를 지원하는지 확인합니다.
trainer_params = inspect.signature(Trainer.__init__).parameters
processor_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_dataset["train"],
    "eval_dataset": tokenized_dataset["validation"],
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
    processor_key: tokenizer,
}

trainer = Trainer(**trainer_kwargs)
print("Trainer 생성 완료")
print("train 샘플 수:", len(tokenized_dataset["train"]))
print("eval 샘플 수:", len(tokenized_dataset["validation"]))
```

### 코드 해설

Trainer 생성 코드는 모델, 학습 설정, train/validation 데이터, batch 처리 방식, 평가 방법 등을 Trainer에게 전달해서 이후 학습과 평가를 자동으로 수행할 준비를 하는 과정이다.

Trainer는 내부적으로 batch 생성, forward pass, loss 계산, backward, optimizer step, evaluation, checkpoint 저장을 처리한다. 

하지만 어떤 데이터와 모델을 쓰고 어떤 metric을 계산할지는 우리가 명시해야 한다


## 5. 설정 점검 체크리스트

학습 실행 전 확인

| 항목 | 확인 질문 |
| --- | --- |
| 라벨 수 | `model.config.num_labels`가 실제 label 수와 같나요? |
| 데이터 컬럼 | `input_ids`, `attention_mask`, `labels`가 있나요? |
| device | GPU 사용 가능 여부를 확인했나요? |
| metric | `metric_for_best_model`이 `compute_metrics` 반환 key와 일치하나요? |
| 저장 정책 | evaluation과 save strategy가 호환되나요? |
| batch size | Colab GPU 메모리에서 실행 가능한가요? |


## 선택 · 6. 연습 문제

1. TrainingArguments에 learning rate, batch size, epoch를 기록해야 하는 이유를 설명
2. `metric_for_best_model="macro_f1"`을 쓰려면 `compute_metrics`가 어떤 key를 반환해야 하나요?
3. OOM 오류가 발생하면 조정할 수 있는 설정 두 가지를 적으세요.
4. Trainer에 data_collator를 전달하는 이유를 설명하세요.

---

## 참고 · 7. 정답 확인

1. 이 값들은 학습 결과에 직접 영향을 주는 hyperparameter이므로 실험 재현과 비교를 위해 기록해야 한다
2. `compute_metrics`가 `macro_f1`이라는 key를 포함한 dict를 반환해야 한다
3. `per_device_train_batch_size`를 줄이거나, `gradient_accumulation_steps`를 사용하거나, `max_length`를 줄일 수 있다.
4. 샘플들을 batch Tensor로 묶고, 길이가 다른 문장들을 padding해 모델 입력 shape을 맞추기 위해 필요하다.


## 8. Trainer 전에 모델·Label Mapping 확인


```python
from transformers import AutoModelForSequenceClassification

id2label = {0: "negative", 1: "positive"}
label2id = {name: idx for idx, name in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)
```

Trainer를 만들기 전에 다음을 확인한다

- dataset의 label ID가 `0 ~ num_labels-1` 범위인가

- `model.config.id2label`과 보고서의 class 이름이 같은가

- validation·test에 train에 없던 label이 들어오지 않나

- Tokenizer와 model checkpoint·revision이 맞나

- 한 batch의 logits shape이 `[B,num_labels]`인가


Label Mapping이 어긋나면 학습 자체는 진행돼도 예측 이름과 리포트가 조용히 뒤바뀔 수 있다.

