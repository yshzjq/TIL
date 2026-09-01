---
title: 1-1강 RNN,LSTM의 한계와 Transformer 필요성
date: 2026-09-01
updated: 2026-09-01
description: KANT 강의 '1-1강 RNN,LSTM의 한계와 Transformer 필요성' 정리
---

## 1. RNN은 이전 상태를 기다린다

시퀀스 입력 `x_t`는 이전 hidden state `h_{t-1}`와 함께 현재 상태를 만든다

```
h_t = tanh(x_t W_x + h_(t-1) W_h + b)
x_1 → h_1 → h_2 → ... → h_L
```

`h_t`가 `h_{t-1}`에 의존하므로 같은 시퀀스의 위치 `t+1`은 위치 `t`의 계산이 끝나야 진행할 수 있다.<br>
배치 안의 샘플과 한 시점의 행렬 연산은 병렬화할 수 있지만, **시간축 의존성 자체**는 남는다

```python
import torch

# batch_first=True이므로 입력 축 순서는 [B, L, D_in]입니다.
rnn = torch.nn.RNN(
    input_size=4,   # token마다 들어오는 feature 수 D_in
    hidden_size=6,  # 각 시점이 내보내는 hidden feature 수 H
    batch_first=True,
)

# 샘플 2개, 길이 5, 입력 feature 4인 작은 batch입니다.
x = torch.randn(2, 5, 4)  # [B=2, L=5, D_in=4]
output, h_n = rnn(x)

# output은 다섯 시점 전체, h_n은 마지막 시점의 층·방향별 상태입니다.
print("output:", tuple(output.shape))
print("h_n:", tuple(h_n.shape))
```

출력

```
output: (2, 5, 6)
h_n: (1, 2, 6)
```

`output`은 모든 시점의 표현이고, `h_n`은 층·방향별 마지막 상태입니다.<br> 
다층·양방향에서는 `output[:, -1]`와 `h_n[-1]`을 같은 뜻이라고 단정하면 안 된다

**RNN은 앞의 정보를 hidden state를 통해 차례로 전달하고, Transformer는 Attention을 이용해 필요한 위치의 정보를 직접 가져온다.**

대신 Transformer는 모든 토큰 쌍의 관계를 계산해야 해서 문장이 길어질수록 비용이 크게 증가한다

Attention score가 L²만큼 커짐, 각 토큰이 다른 모든 토큰과 얼마나 관련 있는지 비교하기 때문

Attention score는 현재 토큰이 다른 토큰을 얼마나 중요하게 봐야 하는지 나타내는 점수

L은 **시퀀스의 길이(sequence length)**예요. 쉽게 말하면 토큰 개수


Transformer 는 RNN처럼 이전 hidden state를 한 단계씩 전달하지 않고, Attention을 통해 서로 떨어진 토큰들도 직접 정보를 주고받는다

RNN
→ 정보를 hidden state로 순서대로 전달
→ recurrence 있음

Transformer
→ Attention으로 다른 위치를 직접 참고
→ recurrence 없음


## 2. RNN 긴 문맥에서 세 문제가 겹친다

### 2-1. 장기 의존성

초반 단어가 마지막 예측에 중요하면 정보가 많은 recurrent step을 통과해야 한다,<br> 
첫 위치에서 마지막 위치까지의 최소 경로는 대략 `L-1`다.

### 2-2. Vanishing/Exploding gradient

시간축 역전파에서는 여러 Jacobian이 반복해서 곱해집니다.<br>
작은 값이 계속 곱해지면 학습 신호가 사라지고, 큰 값이 곱해지면 폭주할 수 있습니다.<br>
Gradient clipping은 폭주를 완화하지만 장거리 관계를 자동으로 학습하게 해 주지는 않는다

Jacobian을 반복해서 곱하는 것은 RNN의 시간축 역전파 과정이고,<br> 그 곱의 결과 gradient가 지나치게 커지는 현상이 Exploding Gradient이다.

Gradient clipping = 역전파로 계산된 gradient가 너무 커졌을 때 최대 크기를 제한해서 학습이 폭주하지 않게 하는 방법입니다.

#### 시간축 역전파(Backpropagation Through Time, BPTT)

RNN은 같은 계산을 시간 순서대로 반복한다

 x1 → h1 → h2 → h3 → h4 → 출력

예를 들어 마지막 출력에서 loss가 발생했다고 하면, 학습할 때는 반대로 gradient를 전달해야 한다

loss<br>
 ↓<br>
h4<br>
 ↓<br>
h3<br>
 ↓<br>
h2<br>
 ↓<br>
h1<br>

즉 시간 순서로 펼쳐진 RNN을 뒤에서 앞으로 역전파하는 것을 시간축 역전파라고 합니다.

#### Jacobian


h_(t-1) = [a, b, c]

h_t     = [d, e, f]

이전 hidden state의 각 값이 현재 hidden state의 각 값에 얼마나 영향을 주는지 전부 알아야 합니다.

a가 d에 얼마나 영향?<br>
a가 e에 얼마나 영향?<br>
a가 f에 얼마나 영향?<br>

b가 d에 얼마나 영향?<br>
b가 e에 얼마나 영향?<br>
...
이 여러 미분값을 하나의 행렬로 모은 것이 Jacobian(야코비안) 행렬

### 2-3. 고정 크기 상태의 정보 병목

초기 RNN Encoder–Decoder는 입력 전체를 마지막 상태 하나에 압축했다.<br>
입력이 길어질수록 필요한 세부 정보가 희석될 수 있다.(정보 병목 현상)

Transformer 이전의 Attention도 이 병목을 줄이기 위해 Encoder의 여러 상태를 직접 참고하도록 도입되었다.

모든 정보를 마지막 hidden state 하나에 몰아넣는 부담을 줄이고, <br>Decoder가 Encoder의 여러 위치 정보를 직접 볼 수 있게 한다

Attention은 여러 위치의 정보 중 현재 작업에 중요한 정보를 더 많이 참고하도록 가중치를 계산하는 방법

#### RNN Encoder–Decoder

RNN Encoder–Decoder는 쉽게 말하면 RNN 두 개를 이어서 입력을 읽고, 그 내용을 바탕으로 다른 시퀀스를 만드는 구조

Encoder의 역할은:
입력 문장을 읽어서 내용을 hidden state에 담는 것

Decoder의 역할은:
Encoder가 전달한 정보를 이용해서 출력 시퀀스를 생성하는 것

*전체 흐름*

입력 문장
   ↓
Encoder
   ↓
문장 정보를 담은 hidden state
   ↓
Decoder
   ↓
출력 문장

Encoder = 입력을 읽고 표현으로 변환<br>
Decoder = 그 표현을 이용해 출력을 생성<br>
둘 다 RNN을 사용하면 RNN Encoder–Decoder<br>


## 3. LSTM은 gradient 흐름을 개선하지만 순차성은 유지한다

LSTM은 cell state와 gate를 사용한다

| 구성 | 역할 |
| --- | --- |
| Forget gate | 이전 정보 중 버릴 비율 결정 |
| Input gate | 새 정보를 반영할 비율 결정 |
| Output gate | 현재 출력에 드러낼 정보 결정 |
| Cell state | 비교적 직접적인 정보 전달 경로 |

LSTM은 vanilla RNN보다 장기 정보를 잘 보존하지만, <br>
`t`가 `t-1`을 기다리는 구조와 제한된 상태에 문맥을 축약하는 부담은 남는다.<br>
시계열·스트리밍·작은 모델에서는 여전히 좋은 baseline이 될 수 있다.


## 4. Transformer는 위치 쌍을 직접 연결한다.


Self-Attention은 각 token이 같은 시퀀스의 다른 token을 직접 참고하게 한다

```
RNN:            1 → 2 → 3 → ... → L       경로 O(L)
Self-Attention: 1 ─────────────────→ L      한 층 경로 O(1)
```

한 층의 모든 위치를 큰 행렬 연산으로 계산하므로 학습 시 sequence 위치의 병렬화가 쉽다.<br>
대신 기본 score 행렬은 `[L, L]`이고 원소 수가 `L²`으로 증가한다<br>
생성 시에는 새 token이 이전 token에 의존하므로 Transformer도 token별 순차 생성이 필요하다


```python
# 길이가 커질 때 두 비용 지표가 어떤 속도로 변하는지 비교합니다.
for length in (4, 16, 128):
    rnn_path = length - 1          # 첫 위치에서 마지막 위치까지 거치는 recurrent edge 수
    score_elements = length ** 2   # head 하나의 [L,L] score 행렬 원소 수
    print(length, "RNN path=", rnn_path, "Attention scores=", score_elements)
```

출력

```
4 RNN path= 3 Attention scores= 16
16 RNN path= 15 Attention scores= 256
128 RNN path= 127 Attention scores= 16384
```

## 5. 비교표: 우열이 아니라 제약의 차이

| 관점 | RNN/LSTM | Transformer |
| --- | --- | --- |
| 같은 layer 안 정보 경로 | 위치 수에 따라 길어짐 | 먼 위치도 직접 연결 |
| 학습 시 시간축 계산 | 순차 의존 | 행렬 단위 병렬 계산 |
| 긴 문맥 | 상태에 계속 축약 | 모든 위치를 직접 비교 가능 |
| 주요 비용 | 긴 직렬 경로 | 기본 Attention의 `L²` 메모리·연산 |
| 유리할 수 있는 조건 | 스트리밍, 작은 모델, 짧은 시계열 | 대규모 학습, 긴 문맥 관계, 전이학습 |

## 오류·주의사항

- “RNN은 병렬 처리가 전혀 불가능하다”가 아니라 **한 시퀀스의 시간축 의존**이 핵심
- LSTM이 장기 의존성을 완전히 해결한다고 표현하지 않는다.
- Self-Attention의 짧은 경로와 `L²` 비용을 함께 설명한다.
- 양방향 RNN 출력은 실시간 미래 예측처럼 미래 입력을 볼 수 없는 문제에 그대로 적용할 수 없다.


## 요약

- RNN은 hidden state를 시간 순서대로 전달해 순서를 반영하지만 긴 직렬 경로를 만듭니다.
- LSTM은 cell state와 gate로 gradient·기억 문제를 완화하지만 시간축 순차성은 유지합니다.
- Self-Attention은 먼 위치를 한 층에서 직접 연결하고 학습 시 행렬 병렬화를 가능하게 합니다.
- Transformer의 짧은 정보 경로는 기본 score 행렬의 `L²` 비용과 함께 평가해야 합니다.