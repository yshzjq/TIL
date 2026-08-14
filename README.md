# TIL

공부하며 새롭게 이해한 내용을 기록하고, GitHub Pages로 보여주는 개인 학습 노트입니다.

## 사이트

- TIL 웹사이트: <https://yshzjq.github.io/TIL/>
- 저장소: <https://github.com/yshzjq/TIL>

> 저장소 주소에서는 README가 보이는 것이 정상입니다. 위의 TIL 웹사이트 주소에서 실제 사이트를 볼 수 있습니다.

## 새 글 작성하기

주제에 맞는 폴더 안에 Markdown 파일을 만듭니다. 경로와 파일명은 URL로 사용되므로 영문 소문자와 하이픈 사용을 권장합니다.

```text
TIL/
├─ machine-learning/
│  └─ linear-regression.md
└─ math/
```

`_templates/TIL.md`를 복사하거나 아래 형식으로 새 파일을 만듭니다. 각 글의 맨 위에는 글 정보를 작성합니다. `layout`, `category`, `til`은 폴더별 기본값이 설정되어 있어 생략할 수 있습니다.

```markdown
---
title: 선형 회귀의 기본 원리
date: 2026-08-14
description: 선형 회귀가 데이터를 학습하는 기본 원리를 정리했다.
---

# 선형 회귀의 기본 원리

본문을 작성합니다.
```

지원하는 폴더는 `machine-learning`, `math`입니다. 다른 폴더를 추가하려면 `_config.yml`의 `defaults`와 `_data/categories.yml`에도 해당 카테고리를 추가합니다.

## GitHub Pages 배포

GitHub 저장소의 **Settings → Pages → Build and deployment**에서 다음과 같이 설정합니다.

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/(root)`

설정 후 `main` 브랜치에 변경 사항을 push하면 사이트가 자동으로 갱신됩니다.
