---
title: 5-4강 GPT Decoder-only와 Causal LM .md
date: 2026-09-07
updated: 2026-09-07
description: KANT 강의 '5-4강 GPT Decoder-only와 Causal LM .md' 정리
---

## 1. GPT란 무엇인가?

GPT는 **Generative Pre-trained Transformer**의 약자다

| 단어 | 의미 |
| --- | --- |
| Generative | 이전 문맥을 바탕으로 새 토큰을 생성합니다. |
| Pre-trained | 대규모 텍스트에서 먼저 언어 패턴을 학습합니다. |
| Transformer | Attention 기반 Transformer Block을 사용합니다. |

**GPT 계열은 Transformer Decoder-only Block을 여러 층 쌓고,<br> 이전 토큰을 바탕으로 다음 토큰을 예측하도록 사전학습된 생성형 언어모델 계열이다.**



