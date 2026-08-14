---
title: 작성 안내
permalink: /guide/
description: 새로운 TIL 글을 추가하는 방법을 안내합니다.
---

<header class="page-header">
  <p class="eyebrow">WRITING GUIDE</p>
  <h1>새 TIL 작성하기</h1>
  <p>주제 폴더에 Markdown 파일 하나만 추가하면 홈과 카테고리에 자동으로 표시됩니다.</p>
</header>

<div class="guide-layout">
  <aside class="guide-summary">
    <strong>3단계면 충분해요</strong>
    <ol>
      <li>주제 폴더를 고릅니다.</li>
      <li>Markdown 파일을 만듭니다.</li>
      <li>내용을 작성하고 push합니다.</li>
    </ol>
  </aside>

  <div class="guide-content" markdown="1">
    <section>
      <span class="step-number">01</span>
      <h2>주제 폴더 선택</h2>
      <p><code>python</code>, <code>git</code>, <code>data</code>, <code>machine-learning</code>, <code>llm</code>, <code>math</code> 중 알맞은 폴더를 고릅니다.</p>
      <p>경로가 그대로 URL의 일부가 되므로 폴더명과 파일명에는 <strong>영문 소문자와 하이픈</strong> 사용을 권장합니다.</p>
    </section>

    <section>
      <span class="step-number">02</span>
      <h2>글 정보 작성</h2>
      <p><code>_templates/TIL.md</code>를 복사한 뒤 파일 맨 위에 제목, 날짜, 한 줄 설명을 작성합니다. 폴더에 따라 레이아웃과 카테고리는 자동으로 지정됩니다.</p>

```markdown
---
title: 리스트 컴프리헨션 이해하기
date: 2026-08-14
description: 반복문으로 리스트를 간결하게 만드는 방법을 정리했다.
---

# 리스트 컴프리헨션 이해하기

오늘 배운 내용을 자유롭게 작성합니다.
```
    </section>

    <section>
      <span class="step-number">03</span>
      <h2>GitHub에 반영</h2>
      <p>변경 내용을 <code>main</code> 브랜치에 push하면 GitHub Pages가 사이트를 다시 만들고 새 글을 자동으로 공개합니다.</p>
    </section>
  </div>
</div>
