---
title: 2-1강 NLP 데이터 구조와 Token·Subword
date: 2026-09-01
updated: 2026-09-02
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

`인공지능연구소`가 하나의 word token으로 없더라도 `인공`, `##지능`, `연구`, `##소`처럼 기존 조각을 조합할 수 있다. 실제 분할 표시는 WordPiece, BPE, SentencePiece 구현마다 다르다

```
문자열 → tokenizer의 정규화·pre-tokenization·subword 알고리즘
      → token 문자열 → vocabulary lookup → input_ids
```

## 4. Vocabulary 크기와 sequence 길이의 trade-off

Vocabulary를 키우면 자주 쓰는 긴 문자열을 하나의 token으로 담아 길이가 짧아질 수 있지만 embedding·LM head 파라미터가 늘어난다<br> 
Vocabulary를 줄이면 희귀 문자열을 더 작은 조각으로 표현해 sequence가 길어질 수 있다.

Vocabulary 크기는 token 수에 영향을 주지만, <br>
실제 분할 결과는 Tokenizer의 분할 방식, 학습에 사용한 데이터, 문자열 정규화 규칙에도 영향을 받기 때문에 Vocabulary 크기만으로 <br>
Tokenizer의 성능이나 sequence 길이를 판단할 수 없다.

Vocabulary를 키우면 자주 쓰는 긴 문자열을 하나의 token으로 담을 수 있어 sequence 길이가 짧아질 수 있다.<br>
반면 저장하고 구분해야 하는 token 종류가 많아지므로 모델의 메모리 부담도 커질 수 있다.


## 5. Special token은 모델 입력의 구조를 표시

| Token 역할 | 대표 표기 | 용도 |
| --- | --- | --- |
| Padding | `[PAD]` | batch 길이 맞춤 |
| Unknown | `[UNK]` | 표현하지 못한 문자열 |
| Classification/Beginning | `[CLS]`, `<s>` | 문장 대표 또는 시작 |
| Separator/End | `[SEP]`, `</s>` | 문장 경계 또는 끝 |
| Mask | `[MASK]` | Masked LM 학습·추론 |

모든 모델이 같은 표기와 역할을 쓰지 않는다.<br>
Token ID를 하드코딩하지 말고 `tokenizer.pad_token_id`, `tokenizer.mask_token`처럼 객체에서 읽는다

## 6. 데이터 누수와 중복

같은 기사 제목이 train과 test에 동시에 있으면 모델이 일반화한 것이 아니라 본 문장을 기억했을 수 있다. <br>
다음 항목을 tokenization 전에 확인

1. ID 유일성
2. 텍스트 정규화 후 중복
3. label 집합과 `label2id`
4. 클래스 분포
5. 출처·시간 단위 split 필요성
6. 개인정보·라이선스·민감 정보

## 7. 출력 예시를 읽는 방법

Toy tokenizer가 다음 결과를 냈다고 가정한다

```
text: 인공지능연구소 출범
tokens: [CLS], 인공, ##지능, 연구, ##소, 출범, [SEP]
ids:    2,     4102, 7821, 991,   441,  7300, 3
```

*계약(contract)**은 “이 데이터가 어떤 규칙과 형식을 지켜야 하는가”

여기서 확인할 것은 특정 ID 숫자가 아니라 계약이다.<br>
Token 7개와 ID 7개가 1:1로 대응하고, `##`는 앞 조각에 이어지는 subword임을 표시하며, 문장 경계 special token이 앞뒤에 추가되었다.<br>
숫자 ID는 예시일 뿐 실제 KoELECTRA vocabulary에서 다시 조회해야 한다.

## 8. 실무 판단: label과 token 계약을 분리

`label2id`는 프로젝트 task의 계약이고 token ID는 tokenizer checkpoint의 계약입니다. 둘 다 정수지만 섞어 해석하면 안 됩니다.

```python
# label ID는 task가 정하는 별도 namespace입니다.
label2id = {"economy": 0, "sports": 1, "tech": 2}
id2label = {value: key for key, value in label2id.items()}  # 예측 ID를 이름으로 복원

# 서로 다른 label이 같은 ID를 공유하지 않는지, 왕복 변환이 되는지 검사합니다.
assert len(label2id) == len(set(label2id.values()))
assert all(id2label[label2id[name]] == name for name in label2id)
```

모델 예측의 `0`은 economy label을 뜻할 수 있지만 tokenizer `input_ids`의 `0`은 PAD 같은 전혀 다른 token을 뜻할 수 있다.<br>
어느 namespace의 ID인지 변수명과 표 제목에 명시한다

## 오류·주의사항

- “token=단어”라고 고정하지 않는다
- `len(text)`는 문자 수이지 token 수가 아니다.
- `[UNK]`가 적다고 무조건 좋은 tokenizer는 아니다. 길이·언어 범위·모델 호환성을 함께 봐야한다
- 원문 label 문자열과 학습용 정수 ID의 매핑을 양방향으로 저장한다.

