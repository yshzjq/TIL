---
title: 1-2강 미니 LLM 워크플로우와 누적 산출물
date: 2026-09-01
updated: 2026-09-01
description: KANT 강의 '1-2강 미니 LLM 워크플로우와 누적 산출물' 정리
---

## 1. 한 줄 워크플로우

```
문제 정의 → 데이터 감사·split → tokenize·collate → model·objective
→ train → validation으로 선택 → test 1회 평가 → 저장·재로드
→ 새 입력 inference → error analysis → 최종 실험 기록
```

이 순서는 도구 이름이 바뀌어도 유지되는 실험 계약입니다.
<br>
먼저 “무엇을 예측하고 어떤 기준으로 성공이라 할지”를 정하고, 
<br>
마지막에는 다른 사람이 같은 환경과 설정으로 재현할 수 있게 기록한다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\미니LLM워크플로우.png' | relative_url }}" alt="미니LLM워크플로우.png" loading="lazy">

도식의 윗줄은 분류와 생성이 공유하는 입력 경로이고,<br>
아랫줄은 학습·평가·추론에서 달라지는 행동이다

특히 `model forward`가 같아 보여도 학습에서는 loss와 parameter update가,<br>
평가에서는 고정된 checkpoint와 metric이, 추론에서는 실제 서비스 입력과 후처리가 필요하다 즉,<br>
학습이 끝난 모델에 사용자가 새로 입력한 데이터를 넣어 예측하고, 모델이 만든 logits나 token ID를 사람이 사용할 수 있는 결과로 변환해야 한다.

모델이 낸 원시 출력을 사람이 사용할 수 있는 형태로 바꾸는 과정이 후처리(post-processing)


<img src="{{ '/assets/images/uploads\deep-learning-advanced\전체 NLP 파이프라인.png
' | relative_url }}" alt="전체 NLP 파이프라인.png
" loading="lazy">

Hugging Face 공식 LLM Course의 전체 NLP pipeline 도식은 raw text가 token과 ID를 거쳐 Transformer와 task head의 예측으로 바뀌는 구간을 보여 준다.<br>
이 그림을 볼 때는 “pipeline 한 줄이면 끝난다”보다,<br>
각 화살표의 입력·출력 계약을 task별로 먼저 고정해야 한다는 점이 중요하다

ID는 Tokenizer가 각 token에 붙여 둔 숫자 번호

Tokenizer = 텍스트를 token으로 나누고, 각 token을 모델이 사용할 숫자 ID로 변환하는 도구

## 2. Raw text가 모델 입력이 되는 과정

```
raw text ── tokenizer ──┬─ input_ids       [B, L] ─┐
                        └─ attention_mask  [B, L] ─┤
                                                    ├─ model ─┬─ loss
분류 정답 ─ label mapping ─ labels            [B] ─┤         └─ logits
tokenized LM input_ids ─ collator/objective ─ labels [B, L] ─┘
```

- `input_ids`: vocabulary의 정수 ID
- `attention_mask`: 실제 token은 1, PAD 위치는 0인 대표적 입력 mask
- `labels`: tokenizer의 산출물이 아니라 Dataset의 분류 정답 또는 LM collator·학습 objective가 만드는 학습 목표
- `logits`: 정규화 전 점수
- `loss`: 학습할 때 gradient를 만드는 목적 함수

Vocabulary는 모델이 사용할 수 있는 token들과 그 token에 대응하는 ID를 모아둔 사전입니다.

Tokenizer와 model은 같은 checkpoint 계열을 사용해야 한다<br>
ID는 tokenizer의 vocabulary에 종속되기 때문이다

## 3. Task별 출력 계약

| Task | 대표 model class | Logits shape | 대표 평가 |
| --- | --- | --- | --- |
| 문장 분류 | `AutoModelForSequenceClassification` | `[B, C]` | accuracy, macro-F1 |
| Masked LM | `AutoModelForMaskedLM` | `[B, L, V]` | mask 위치 top-k |
| Causal LM | `AutoModelForCausalLM` | `[B, L, V]` | 생성 품질, 지연, 안전성 |

같은 `AutoTokenizer` API를 써도 출력 head와 평가 기준은 다르다


```python
import torch

# 예제는 샘플 2개(B=2), 클래스 3개(C=3)의 분류 logits입니다.
# Logit은 확률이 아닌 실수 점수이므로 합이 1일 필요가 없습니다.
logits = torch.tensor([[-1.2, 2.7, 0.4], [1.8, 0.2, -0.5]])
probabilities = torch.softmax(logits, dim=-1)  # 클래스 축 C에만 softmax
predictions = logits.argmax(dim=-1)            # 각 샘플에서 가장 큰 클래스 ID

print("prob shape:", tuple(probabilities.shape))
print("predictions:", predictions.tolist())
print("row sums:", probabilities.sum(dim=-1))  # 각 행이 약 1인지 확인
```

출력

```
prob shape: (2, 3)
predictions: [1, 0]
row sums: tensor([1.0000, 1.0000])
```

Softmax는 클래스 축 `dim=-1`에 적용힌디<br>
배치 축에 적용하면 서로 다른 샘플이 경쟁하는 잘못된 계산이 된다

## 4. 학습·평가·추론을 구분

| 단계 | 정답 label | 파라미터 갱신 | 목적 |
| --- | --- | --- | --- |
| Train | 있음 | 있음 | loss를 줄이도록 학습 |
| Validation | 있음 | 없음 | hyperparameter·checkpoint 선택 |
| Test | 있음 | 없음 | 모든 선택 후 최종 일반화 성능 확인 |
| Inference | 보통 없음 | 없음 | 새 입력에 실제 결과 생성 |

Test 결과를 보며 epoch나 설정을 다시 고르면 test가 validation 역할을 하게 된다.

최종 실험 기록에는 test를 몇 번 확인했는지 남긴다

## 5. 생성에서 forward와 `generate()`는 다르다

Causal LM의 forward 한 번은 각 위치의 다음-token logits를 계산한다.<br>
`generate()`는 마지막 위치에서 token을 선택하고 입력에 붙이는 과정을 종료 조건까지 반복한다

```
forward:  input → [B, L, V] logits
generate: forward → token 선택 → append → forward → ... → decode
```

`max_new_tokens`는 새로 만들 token 수만 제한하고, `max_length`는 입력과 출력을 합친 전체 길이를 제한한다.

## 6. 재현 가능한 실행 기록

최소 기록 단위

```python
# 비교 실험마다 같은 key를 사용하면 누락을 빠르게 찾을 수 있습니다.
manifest = {
    "problem": "한국어 뉴스 제목 3분류",
    "dataset_version": "news-mini-v1",
    "split_seed": 42,  # 같은 split을 다시 만들기 위한 seed
    "model_id": "monologg/koelectra-small-v3-discriminator",
    "tokenizer_id": "monologg/koelectra-small-v3-discriminator",
    "max_length": 64,  # 입력 길이 분포를 본 뒤 고른 절단 상한
    "selection_metric": "validation_macro_f1",
    "test_policy": "best checkpoint 확정 후 1회",
}
```

Seed만 남겨도 충분하지 않다
<br> 
package version, dataset fingerprint, model revision, 실제 checkpoint 경로와 평가 코드도 함께 기록해야 한다

## 7. 실험 기록 구조

| 장 | 남길 실행 증빙 |
| --- | --- |
| 1 | 문제 정의·구조 선택 근거·실험 manifest |
| 2 | token 예시·길이 분포·padding/truncation 정책 |
| 3 | Attention score/weight/context와 mask 검증 |
| 4 | block·mask·memory shape 표 |
| 5 | BERT/GPT objective·출력 비교 |
| 6 | Model Card·license·저장/재로드 증빙 |
| 7 | split·metric·best checkpoint·test·오류 분석 |
| 8 | prompt-only·PEFT·fine-tuned 비교 조건 |
| 9 | decoding·Chat Template 결과와 최종 결론 |

각 표와 로그에는 “실제 실행값”과 “해석”을 분리한다.
<br>예상값을 실행값처럼 쓰지 않습니다.

## 오류·주의사항

- `model_id`와 tokenizer의 checkpoint를 다르게 섞지 않는다.
- Accuracy와 loss를 같은 개념으로 설명하지 않는다.
- Prompt-only를 prompt tuning이라고 부르지 않는다
- 마지막 checkpoint를 자동으로 best checkpoint라 가정하지 않는다
- 생성 문자열이 자연스럽다는 이유로 사실성이 검증됐다고 결론 내리지 않는다

## 이번 강의 요약

- LLM 실험은 문제 정의에서 시작해 split·tokenize·model·학습·평가·저장·추론·오류 분석으로 이어진다
- 분류와 생성은 tokenizer 입구를 공유할 수 있지만 model head, logits shape, metric, 후처리가 다르다.
- Validation은 선택, test는 선택 완료 뒤 최종 확인에 사용한다
- 실행값·환경·설정·산출물 경로를 함께 기록해야 다른 사람이 재현할 수 있는 실험 문서가 된다

