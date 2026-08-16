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

지원하는 폴더는 `machine-learning`, `math`, `python`입니다. 다른 폴더를 추가하려면 `_config.yml`의 `defaults`, `_data/categories.yml`, `admin/config.yml`에도 해당 카테고리를 추가합니다.

## 브라우저에서 글 관리하기

배포된 사이트의 `/admin/` 주소에서 콘텐츠 편집기를 열 수 있습니다.

- 관리 주소: <https://yshzjq.github.io/TIL/admin/>
- 새 글 작성, 기존 글 수정과 삭제를 지원합니다.
- 본문 편집기에서 이미지를 붙여 넣거나 업로드할 수 있습니다.
- 저장하면 Markdown과 이미지가 `main` 브랜치에 커밋되고 GitHub Pages가 자동으로 사이트를 갱신합니다.

처음 로그인할 때는 GitHub 토큰이 필요합니다. 편집기의 **Sign In with Token** 화면에서 안내하는 GitHub 링크로 필요한 권한이 선택된 토큰을 만들 수 있습니다. 가능한 짧은 만료 기간을 사용하고, 토큰을 공개하거나 Markdown 파일에 저장하지 않습니다. 토큰은 현재 브라우저에 보관되므로 공용 컴퓨터에서는 작업 후 반드시 로그아웃합니다.

## GitHub Pages 배포

GitHub 저장소의 **Settings → Pages → Build and deployment**에서 다음과 같이 설정합니다.

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/(root)`

설정 후 `main` 브랜치에 변경 사항을 push하면 사이트가 자동으로 갱신됩니다.
