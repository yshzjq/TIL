---
title: 3-3강 Attention Weight와 Context Vector 해석
date: 2026-09-03
updated: 2026-09-03
description: KANT 강의 '3-3강 Attention Weight와 Context Vector 해석' 정리
---

## 1. Attention weight는 참고 비율

Attention weight는 한 Query가 여러 Key를 얼마나 참고할지 나타낸다

```
"먹었다" Query의 예시 weight
민수는     0.25
식당에서   0.15
라면을     0.45
먹었다     0.15
합계       1.00
```

이 예시에서는 `먹었다`가 `라면을` 가장 많이 참고한다.<br>
하지만 나머지 토큰의 정보도 완전히 버리는 것은 아니다

Weight는 정답 확률이 아니다. <br>
현재 Attention layer의 특정 head에서 정보를 섞는 비율입니다.

## 2. Heatmap을 한 행씩 읽는 방법

Attention weight는 행렬이므로 색으로 표현하면 Heatmap이 된다

<img src="{{ '/assets/images/uploads\deep-learning-advanced\01_heatmap_reading.png
' | relative_url }}" alt="01_heatmap_reading.png
" loading="lazy">

Attention Heatmap 읽기

### 2-1. 행과 열

- 행: 현재 Query 토큰
- 열: 참고 후보 Key 토큰
- 진한 칸: 해당 Query가 그 Key에 더 큰 weight를 줌


`먹었다` 행만 고정해서 보면 다음처럼 읽을 수 있다
```
먹었다 -> 라면을     0.45
먹었다 -> 민수는     0.25
먹었다 -> 식당에서   0.15
먹었다 -> 먹었다     0.15
```

### 2-2. 대각선이 진하면 무엇을 의미하나요?

대각선은 토큰이 자기 자신을 참고하는 위치다.<br>
대각선이 진한 패턴은 흔히 볼 수 있지만, 항상 진해야 하는 규칙은 아니다

## 3. Context vector는 Value의 가중합이다

Attention weight를 만들었으면 각 weight를 같은 위치의 Value에 곱해 모두 더한다

Value의 가중합으로 Context 만들기

```
Context
= 0.25 × 민수 정보
+ 0.15 × 식당 정보
+ 0.45 × 라면 정보
+ 0.15 × 행동 정보
```

Context vector는 특정 단어 하나를 그대로 복사한 것이 아니다<br>
여러 위치에서 가져온 정보가 비율에 따라 섞인 새로운 벡터다


### 3-1. Key와 Value 구분

| 역할 | 질문 |
| --- | --- |
| Key | 이 위치를 얼마나 참고할까요? |
| Value | 이 위치에서 어떤 정보를 가져올까요? |

Weight는 Key와 비교해 만들지만, 마지막에는 Value에 곱한다

## 4. Context vector가 토큰 표현을 어떻게 바꾸나

문맥에 따라 달라지는 표현

```
민수는 라면을 먹었다.
-> 사람이 음식을 섭취하는 의미

불길이 건물을 먹었다.
-> 불길이 건물을 집어삼킨다는 비유적 의미
```

Self-Attention은 주변 토큰을 참고해 `먹었다`의 표현을 문맥에 맞게 바꾼다.<br>
이처럼 주변 문맥이 반영된 표현을 **contextual representation**이라고 부른다

같은 단어라도 문장에 따라 뜻이 달라지니까, Attention이 주변 단어들을 참고해서 그 단어의 벡터 표현을 다르게 만든다는 뜻

contextual representation = 주변 문맥을 고려해서 새롭게 만들어진 토큰의 벡터 표현

## 5. Attention weight 해석 시 주의할 점

Attention weight는 모델 내부를 살펴보는 데 유용하지만, 최종 예측의 모든 이유를 설명하지는 않는다

### 5-1. 유용한 관찰

- 어떤 위치에 큰 weight가 있는지 확인한다
- Padding이나 mask가 제대로 적용됐는지 확인한다
- Head별 패턴이 어떻게 다른지 비교한다
- 예상과 다른 참고 관계를 오류 분석의 단서로 사용한다

### 5-2. 피해야 할 단정

- Weight가 높으면 반드시 인간이 생각하는 핵심 단어라고 단정
- Weight가 낮으면 그 토큰이 최종 예측에 아무 영향도 없다고 단정
- Head 하나의 heatmap만 보고 모델 전체의 판단 이유를 설명

Attention heatmap은 **관찰 도구**다. 
<br>
입력을 바꿨을 때 예측이 어떻게 변하는지, 오류 사례가 무엇인지와 함께 해석해야 한다

## 6. 연습 문제: Weight와 Context 계산

다음 코드는 이미 만들어진 Attention weight와 Value를 사용해 context vector를 계산

```python
import torch

# 한 Query가 네 개 Key에 준 Attention weight입니다.
# Shape [B=1, L_query=1, L_key=4]입니다.
weights = torch.tensor(
    [[[0.25, 0.15, 0.45, 0.15]]],
    dtype=torch.float32,
)

# 네 개 토큰 위치의 Value입니다.
# 각 Value는 2차원 벡터라고 가정합니다.
# Shape [B=1, L_key=4, D_value=2]입니다.
values = torch.tensor(
    [[[1.0, 0.0],   # 민수 관련 정보
      [0.0, 1.0],   # 식당 관련 정보
      [1.0, 1.0],   # 라면 관련 정보
      [0.5, 0.5]]], # 행동 관련 정보
    dtype=torch.float32,
)

# Attention weight와 Value를 행렬곱합니다.
# [1,1,4] @ [1,4,2] -> [1,1,2]
context = torch.matmul(weights, values)

print("weights shape:", weights.shape)
print("values shape:", values.shape)
print("context shape:", context.shape)
print("context:", context)

# Weight 한 행의 합이 1인지 확인합니다.
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(1, 1),
    atol=1e-6,
)

# 출력은 Query 하나에 대한 2차원 새 표현입니다.
assert context.shape == (1, 1, 2)
```

출력

```
context ≈ [[[0.775, 0.675]]]
```

이 값은 다음 가중합의 결과

```
0.25×[1,0] + 0.15×[0,1] + 0.45×[1,1] + 0.15×[0.5,0.5]
```

**weight가 Value를 섞는 비율**이라는 점을 인지



## 8. 이해도 점검

1. Attention weight 한 행은 무엇을 나타내나?
2. Context vector는 어떻게 만들어지나?
3. Heatmap의 행과 열은 각각 무엇인가?
4. Context vector와 원래 Value 하나의 차이는 무엇인가?
5. Attention weight를 완전한 설명으로 보면 안 되는 이유는 무엇인가?

### 정답 확인

1. Query 하나가 모든 Key를 참고하는 비율을 나타낸다.
2. Weight를 같은 위치의 Value에 곱해 모두 더해서 만든다.
3. 행은 Query, 열은 Key다.
4. Context vector는 여러 Value가 weight에 따라 섞인 새로운 표현.
5. Weight는 특정 layer/head의 참고 패턴이며, 최종 예측에 영향을 주는 모든 계산을 단독으로 설명하지 않기 때문



## 요약

- Attention weight는 한 Query가 각 Key를 참고하는 비율.
- Heatmap은 행을 Query, 열을 Key로 읽는다
- Context vector는 Attention weight에 따른 Value의 가중합이다
- Context vector는 주변 문맥이 반영된 새로운 토큰 표현이다
- Attention weight는 유용한 관찰 도구이지만 완전한 예측 설명은 아니다