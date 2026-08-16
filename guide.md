---
title: 작성 안내
permalink: /guide/
description: 새로운 TIL 글을 추가하는 방법을 안내합니다.
---

<header class="page-header">
  <p class="eyebrow">WRITING GUIDE</p>
  <h1>TIL 작성하고 관리하기</h1>
  <p>브라우저 편집기에서 글을 작성·수정하고 이미지를 넣을 수 있습니다.</p>
  <div class="hero-actions">
    <a class="button button-primary" href="{{ '/admin/' | relative_url }}" rel="nofollow">콘텐츠 편집기 열기</a>
  </div>
</header>

<div class="guide-layout">
  <aside class="guide-summary">
    <strong>이 순서로 작성해요</strong>
    <ol>
      <li>편집기에 로그인합니다.</li>
      <li>카테고리를 고릅니다.</li>
      <li>글과 이미지를 작성합니다.</li>
      <li>저장하고 배포를 기다립니다.</li>
    </ol>
  </aside>

  <div class="guide-content" markdown="1">
    <section>
      <span class="step-number">01</span>
      <h2>콘텐츠 편집기 로그인</h2>
      <p>위의 <strong>콘텐츠 편집기 열기</strong>를 누르고 GitHub 토큰으로 로그인합니다. 이 저장소에 글을 저장할 권한이 있는 계정만 사용할 수 있습니다.</p>
      <p>공용 컴퓨터에서는 사용을 마친 뒤 반드시 로그아웃합니다.</p>
    </section>

    <section>
      <span class="step-number">02</span>
      <h2>카테고리와 글 선택</h2>
      <p><code>머신러닝</code>, <code>수학</code>, <code>Python</code> 중 알맞은 카테고리를 고른 뒤 새 글을 만듭니다. 기존 글을 고르면 내용을 수정하거나 삭제할 수 있습니다.</p>
      <p>새 글의 URL 이름은 편집 화면의 더보기 메뉴에서 바꿀 수 있습니다. 영문 소문자와 하이픈 사용을 권장합니다.</p>
    </section>

    <section>
      <span class="step-number">03</span>
      <h2>내용과 이미지 작성</h2>
      <p>제목, 작성일, 한 줄 설명과 본문을 입력합니다. 본문은 일반 문서처럼 편집하거나 원본 Markdown 모드로 전환할 수 있습니다.</p>
      <p>본문 도구 모음의 이미지 버튼을 사용하거나 이미지 파일을 끌어다 놓고 붙여 넣을 수 있습니다. 한 이미지의 최대 크기는 5MB입니다.</p>

```markdown
## 오늘 배운 내용

내용을 자유롭게 작성합니다.

![이미지 설명](/TIL/assets/images/uploads/example.png)
```
    </section>

    <section>
      <span class="step-number">04</span>
      <h2>저장하고 공개</h2>
      <p>저장하면 글과 이미지가 GitHub에 반영됩니다. GitHub Pages가 사이트를 다시 만든 뒤 홈, 카테고리와 검색 결과에 자동으로 표시됩니다.</p>
      <p>공개된 글 아래의 <strong>이 글 수정</strong> 링크를 누르면 다음 수정 때 해당 글을 바로 열 수 있습니다.</p>
    </section>
  </div>
</div>
