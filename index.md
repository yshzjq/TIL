---
title: 홈
description: 매일 배운 내용을 차곡차곡 기록하는 개발 학습 노트입니다.
---

{% assign til_pages = site.pages | where: "til", true | sort: "date" | reverse %}

<section class="hero">
  <p class="eyebrow">LEARN · RECORD · GROW</p>
  <h1>오늘의 배움을<br><span>내일의 지식</span>으로.</h1>
  <p class="hero-description">공부하며 새롭게 이해한 내용을 짧고 명확하게 기록합니다.<br class="desktop-only"> 작은 기록이 쌓여 단단한 지식이 되는 공간입니다.</p>
  <div class="hero-actions">
    <a class="button button-primary" href="#recent">최근 기록 보기</a>
  </div>
  <div class="hero-stat" aria-label="작성한 TIL 수">
    <strong>{{ til_pages | size }}</strong>
    <span>개의 배움 기록</span>
  </div>
</section>

<section class="section" aria-labelledby="category-title">
  <div class="section-heading">
    <div>
      <p class="eyebrow">CATEGORIES</p>
      <h2 id="category-title">관심 분야별로 살펴보기</h2>
    </div>
    <a class="text-link" href="{{ '/categories/' | relative_url }}">전체 글 보기 <span aria-hidden="true">→</span></a>
  </div>

  <div class="category-grid">
    {% for item in site.data.categories %}
      {% assign category_posts = til_pages | where: "category", item.name %}
      <a class="category-card" href="{{ '/categories/' | relative_url }}#{{ item.slug }}">
        <span class="category-icon" aria-hidden="true">{{ item.icon }}</span>
        <span class="category-info">
          <strong>{{ item.name }}</strong>
          <small>{{ item.description }}</small>
        </span>
        <span class="category-count">{{ category_posts | size }}</span>
      </a>
    {% endfor %}
  </div>
</section>

<section class="section" id="recent" aria-labelledby="recent-title">
  <div class="section-heading">
    <div>
      <p class="eyebrow">RECENT NOTES</p>
      <h2 id="recent-title">최근 TIL</h2>
    </div>
  </div>

  {% if til_pages.size > 0 %}
    <div class="post-list">
      {% for page in til_pages limit: 8 %}
        <article class="post-card">
          <a href="{{ page.url | relative_url }}" aria-label="{{ page.title }} 읽기">
            <div class="post-meta">
              <span class="category-label">{{ page.category }}</span>
              <time datetime="{{ page.date | date_to_xmlschema }}">{{ page.date | date: "%Y.%m.%d" }}</time>
            </div>
            <h3>{{ page.title }}</h3>
            <p>{{ page.description | default: page.excerpt | strip_html | truncate: 120 }}</p>
            <span class="read-more">기록 읽기 <span aria-hidden="true">→</span></span>
          </a>
        </article>
      {% endfor %}
    </div>
  {% else %}
    <div class="empty-state">
      <p>아직 작성된 TIL이 없습니다.</p>
    </div>
  {% endif %}
</section>
