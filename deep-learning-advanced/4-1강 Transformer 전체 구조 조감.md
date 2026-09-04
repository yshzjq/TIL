---
title: 4-1강 Transformer 전체 구조 조감(선택 정리 필요)
date: 2026-09-04
updated: 2026-09-04
description: KANT 강의 '4-1강 Transformer 전체 구조 조감' 정리
---

## 1. Transformer란 무엇인가

**Transformer는 토큰들이 서로를 참고해 문맥이 반영된 표현을 만들고, 그 표현을 분류나 생성 같은 목적에 맞는 출력으로 바꾸는 딥러닝 아키텍처다.**

Transformer = 토큰을 벡터로 바꾸고 → 토큰끼리 서로 참고하게 하고 → 그 정보를 가공해서 → 원하는 답을 출력하는 신경망 구조

Transformer는 한 개의 연산 이름이 아니다. 다음 요소를 연결한 **전체 모델 구조**입니다.

- Token Embedding: 정수 Token ID를 벡터로 바꾼다.
- Position 정보: 토큰의 순서를 알려준다
- Multi-Head Attention: 토큰들이 서로의 정보를 참고힌다
- FFN: 각 토큰 표현을 비선형적으로 가공한다
- Residual Connection과 LayerNorm: 깊은 Block의 학습 흐름을 돕는다
- Output Head: 문맥 표현을 클래스 점수 또는 어휘 점수로 바꾼다

Transformer는 특정 모델 하나의 이름이라기보다 **Attention 기반 Block을 쌓는 아키텍처 계열**이다.<br>
BERT, GPT, T5는 모두 Transformer 계열이다



## 2. Self-Attention과 Transformer는 어떻게 다른가

| 구분 | 의미 |
| --- | --- |
| Self-Attention | 한 시퀀스 안에서 토큰들이 서로를 참고해 새 표현을 만드는 연산 |
| Transformer Block | Attention, FFN, Residual, LayerNorm을 묶은 한 층 |
| Transformer Model | Embedding, 여러 Block, Output Head까지 연결한 전체 모델 |

비유하면 Self-Attention은 자동차의 엔진 부품이고, Transformer는 엔진·차체·바퀴·제어 장치를 연결한 자동차 전체에 가깝다

## 3. 원래 Transformer의 Encoder-Decoder 흐름

원래 Transformer는 기계 번역처럼 **입력 시퀀스를 다른 출력 시퀀스로 변환하는 문제**를 위해 제안되었다


<img src="{{ '/assets/images/uploads\deep-learning-advanced\Encoder와 Decoder 정보 흐름.png
' | relative_url }}" alt="Encoder와 Decoder 정보 흐름.png
" loading="lazy">


Encoder와 Decoder 정보 흐름

### 3-1. Encoder

Encoder는 입력 전체를 읽고, 각 입력 위치마다 문맥이 반영된 표현을 만든다.

```
입력 표현:      [B, L_source, D]
Encoder 출력:  [B, L_source, D]
```

Encoder 출력은 문장 전체를 한 벡터로 압축한 값이 아니라, **입력의 각 위치별 문맥 표현 집합**이다.<br>
원래 Encoder-Decoder 문맥에서는 이 출력을 흔히 `memory`라고 부른다.

### 3-2. Decoder

원래 Decoder는 두 정보를 함께 사용합니다.

1. 지금까지 제공된 Target 토큰
2. Encoder가 만든 `memory`

그래서 원래 Decoder에는 다음 연산이 포함

- Causal Self-Attention: Target의 과거와 현재만 참고
- Cross-Attention: Decoder가 Encoder의 `memory`를 참고
- FFN: 각 Target 위치의 표현을 가공

## 4. 전체 구조를 네 구간으로 나누기

| 구간 | 핵심 질문 | 대표 요소 |
| --- | --- | --- |
| 입력 준비 | 정수 Token ID를 어떤 표현으로 바꿀까 | Token Embedding, Position |
| 문맥화 | 토큰이 다른 토큰을 어떻게 참고할까 | Multi-Head Attention |
| 표현 가공 | 각 토큰의 특징을 어떻게 확장·변환할까 | FFN, Residual, LayerNorm |
| Task 출력 | 문맥 표현을 어떤 문제의 점수로 바꿀까 | Classification Head, LM Head |



<img src="{{ '/assets/images/uploads\deep-learning-advanced\Transformer 공개 구조도.png
' | relative_url }}" alt="Transformer 공개 구조도.png
" loading="lazy">


### 선택 · 4-1. 원 논문의 Figure 1로 전체 구조 읽기(정리 예정)

#### 1단계. 입력과 출력 토큰을 벡터로 바꿉니다

#### 2단계. Encoder가 입력 전체를 문맥화합니다

#### 3단계. Decoder는 미래를 가린 채 현재까지의 출력을 읽습니다

#### 4단계. Decoder가 Encoder의 정보를 Cross-Attention으로 참고합니다

#### 5단계. Linear와 Softmax가 다음 토큰 후보를 만듭니다

## 5. Transformer 계열의 세 가지 구조

### 5-1. Encoder-only

Encoder-only 모델은 문장의 각 토큰이 앞뒤의 모든 토큰을 참고하여 문맥이 반영된 표현을 만든다.<br>
이 표현을 이용해 문장 분류, 검색, 개체명 인식 같은 입력 이해 중심 작업을 수행한다

입력 전체를 양방향으로 참고해 **이해 중심 표현**을 만든다
```
나는 은행에서 돈을 찾았다.
```
은행이라는 단어를 처리할 때 Encoder-only 모델은 보통 은행의 앞쪽 토큰과 뒤쪽 토큰을 모두 참고할 수 있다.

즉 은행을 해석할 때

- 앞의 나는
- 뒤의 돈을 찾았다

앞뒤 문장을 참고해서
<br>
여기서 은행은 강둑(bank)이 아니라 금융기관이구나

같은 문맥을 반영한 표현을 만들 수 있다는 뜻

이게 **양방향(Bidirectional)**

**이해 중심 표현** 은 문장이나 단어의 의미와 문맥을 담은 벡터
```
나는 은행에서 돈을 찾았다.
```
위 문장이 Transformer Encoder를 지나면 각 토큰이 단순한 단어 벡터가 아니라 문맥이 반영된 벡터로 바뀐다

예를 들어 은행의 벡터에는 단순히 "은행"이라는 단어 정보뿐 아니라
```
금융기관이라는 의미
돈과 관련됨
문장에서 장소 역할
주변 단어와의 관계
```
같은 정보가 반영될 수 있다.

이것을 **“이해 중심 표현”**이라고 표현

분류, 검색, 개체명 인식 등에 유용하도록 문맥 정보가 잘 담긴 벡터를 만든다

- 문장 분류
- 검색 임베딩
- 개체명 인식
- 대표 계열: BERT



### 5-2. Decoder-only

각 위치에서 왼쪽의 과거 토큰만 참고하며 다음 토큰을 예측한다

- 텍스트 생성
- 대화
- 코드 생성
- 대표 계열: GPT

### 5-3. Encoder-Decoder

Encoder가 입력을 이해하고, Decoder가 그 입력을 조건으로 출력 시퀀스를 생성한다

Encoder가 입력 내용을 문맥이 반영된 표현으로 만들고, Decoder가 그 표현을 참고하여 출력 토큰을 순서대로 생성

출력 시퀀스를 생성 = 토큰을 한 개씩 순서대로 만들어서 최종 문장이나 결과를 만든다

- 번역
- 요약
- 대표 계열: T5

## 6. Hidden state와 logits 구분

Transformer 본체의 대표 출력은 문맥이 반영된 hidden state다.

Hidden state는 Transformer가 문맥을 반영해서 만든 내부 표현

Logits는 그 내부 표현을 실제 문제의 정답 후보 점수로 바꾼 것

```
hidden state: [B, L, D]
```

하지만 실제 문제에서 필요한 출력은 다르다

### 문장 분류

```
대표 문장 표현
-> Classification Head
-> logits [B, C]
```

### Causal Language Modeling

```
각 위치의 hidden state
-> LM Head
-> logits [B, L, V]
```

- `D`: hidden dimension
- `C`: class 수
- `V`: vocabulary 크기


예시
```
나는 학교에
```
문장이 들어보면 다음 토큰 후보가 Vocabulary 전체에 있다

Vocabulary 크기가 50,000이라면 모델은 이런 식으로 점수를 낸다
```
간다    -> 8.4
갔다    -> 5.1
먹는다  -> 1.2
자동차  -> -2.4
...
```

즉 각 토큰 위치마다 Vocabulary 전체에 대한 점수가 필요

> `[B,L,D]`와 `[B,L,V]`는 겉으로 세 축이지만 마지막 축의 의미가 완전히 다르다. `D`는 모델 내부 표현이고, `V`는 다음 토큰 후보 전체의 점수다.


## 선택 · 7. 연습 문제: PyTorch에서 전체 Shape 관찰(정리 예정)



## 참고 · 10. 이해도 점검

1. Transformer를 한 문장으로 설명.
2. Self-Attention과 Transformer Model의 차이는 무엇인가
3. Encoder가 만드는 `memory`의 대표 shape은 무엇인가
4. Decoder-only 구조에 Cross-Attention이 없는 이유는 무엇인가
5. `[B,L,D]` hidden state와 `[B,L,V]` logits의 마지막 축은 각각 무엇을 의미하나



### 정답 확인

1. 토큰들이 서로를 참고해 문맥 표현을 만들고 목적에 맞는 출력을 만드는 Attention 기반 신경망 아키텍처입니다.
2. Self-Attention은 토큰 관계를 계산하는 연산이고, Transformer Model은 여러 구성요소를 연결한 전체 구조입니다.
3. 일반적으로 `[B,L_source,D]`입니다.
4. 별도의 Encoder가 없으므로 참고할 Encoder memory가 없기 때문입니다.
5. `D`는 hidden representation 차원, `V`는 vocabulary 크기입니다.

