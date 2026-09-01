---
title: 2-1강 NLP 데이터 구조와 Token·Subword
date: 2026-09-01
updated: 2026-09-01
description: KANT 강의 '2-1강 NLP 데이터 구조와 Token·Subword' 정리
---

## 1. Tokenizer보다 먼저 데이터 계약을 고정

분류 샘플의 최소 schema는 다음과 같습니다.

```python
# 각 dict가 데이터셋의 한 행입니다. id는 중복 추적, text는 모델 입력,
# label은 사람이 읽는 원래 클래스 이름으로 사용합니다.
samples = [
    {"id": "n001", "text": "금리 인상 가능성에 시장 긴장", "label": "economy"},
    {"id": "n002", "text": "대표팀, 연장전 끝에 결승 진출", "label": "sports"},
]

required = {"id", "text", "label"}
for row in samples:
    assert required <= row.keys()  # 필수 열 누락을 가장 먼저 차단합니다.
    assert row["id"] and row["text"].strip() and row["label"]  # 빈 값 방어

print("rows:", len(samples))
print("fields:", sorted(samples[0]))
```

출력

```
rows: 2
fields: ['id', 'label', 'text']
```

빈 문자열, 중복 ID, 예상 밖 label, 동일 텍스트의 split 간 중복을 먼저 확인.<br>
Tokenization 이후에는 사람이 읽는 원문에서 오류를 찾기 더 어렵다.


<img src="{{ '/assets/images/uploads\deep-learning-advanced\데이터 계약 후 토큰분할.png
' | relative_url }}" alt="데이터 계약 후 토큰분할.png
" loading="lazy">

text 와 label 계약을 확인한 뒤에만 subword 분할과 vocabulary 조회로 넘어가는 순서를 보여준다

오른쪽의 token ID와 label ID는 모두 정수지만 서로 다른 표에서 나온 값이므로, 코드와 분석표에서 이름을 분명히 나눠야 한다

Subword 분할 = 문장을 tokenizer가 아는 작은 토큰 조각으로 나누는 과정
Vocabulary 조회 = 나눈 토큰을 vocabulary에서 찾아 숫자 ID로 바꾸는 과정

Subword 토큰화는 드문 단어를 의미 있는 하위 단위로 나누어 작은 vocabulary로 표현 범위를 넓힌다

## 2. Token과 vocabulary

- **Token**: tokenizer가 모델 입력의 기본 단위로 취급하는 조각
- **Vocabulary**: token 문자열과 정수 ID의 고정 대응표
- **Token ID**: vocabulary에서 해당 token을 가리키는 정수
- **UNK/OOV**: vocabulary로 표현하지 못한 입력 또는 미등록 단위


ID 숫자 자체에는 보편적 의미가 없다. <br>
`1234`가 어떤 token인지는 특정 tokenizer와 vocabulary 버전에 따라 달라집니다.

## 3. 왜 subword를 사용하나

| 단위 | 장점 | 비용·한계 |
| --- | --- | --- |
| Word | 사람이 읽기 쉽고 sequence가 짧음 | 어휘가 매우 커지고 신조어/OOV가 많음 |
| Character | 작은 어휘로 거의 모든 문자열 표현 | sequence가 길어지고 의미 단위가 잘게 깨짐 |
| Subword | 재사용되는 조각으로 OOV와 길이의 균형 | 분할이 직관적 단어 경계와 다를 수 있음 |

`인공지능연구소`가 하나의 word token으로 없더라도 `인공`, `##지능`, `연구`, `##소`처럼 기존 조각을 조합할 수 있다. 실제 분할 표시는 WordPiece, BPE, SentencePiece 구현마다 다릅니다.