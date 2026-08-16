---
title: 카테고리
permalink: /categories/
description: 분야별로 정리한 모든 TIL 기록입니다.
---

{% assign til_pages = site.pages | where: "til", true | sort: "date" | reverse %}

<header class="page-header">
  <p class="eyebrow">ALL NOTES</p>
  <h1>배움 기록 모아보기</h1>
  <p>왼쪽에서 관심 분야를 고르면 해당 게시글만 빠르게 살펴볼 수 있습니다.</p>
</header>

<div class="category-browser" data-category-browser>
  <aside class="category-sidebar" aria-labelledby="category-filter-title">
    <div class="category-sidebar-heading">
      <h2 id="category-filter-title">카테고리</h2>
    </div>

    <div class="category-filter-list" aria-label="게시글 카테고리 선택">
      <button class="category-filter-button active" type="button" data-category-filter="all" data-category-label="전체" data-category-description="모든 분야의 배움 기록" aria-controls="category-posts" aria-pressed="true">
        <span class="category-filter-icon" aria-hidden="true">✦</span>
        <span class="category-filter-name">전체</span>
        <span class="category-filter-count">{{ til_pages | size }}</span>
      </button>

      {% for item in site.data.categories %}
        {% assign category_posts = til_pages | where: "category", item.name %}
        <button class="category-filter-button" type="button" data-category-filter="{{ item.slug }}" data-category-label="{{ item.name | escape }}" data-category-description="{{ item.description | escape }}" aria-controls="category-posts" aria-pressed="false">
          <span class="category-filter-icon" aria-hidden="true">{{ item.icon }}</span>
          <span class="category-filter-name">{{ item.name }}</span>
          <span class="category-filter-count">{{ category_posts | size }}</span>
        </button>
      {% endfor %}
    </div>
  </aside>

  <section class="category-results" aria-labelledby="category-results-title">
    <div class="category-results-heading">
      <div>
        <p class="eyebrow">ARCHIVE</p>
        <h2 id="category-results-title" data-category-title>전체 게시글</h2>
        <p class="category-results-summary" data-category-summary aria-live="polite">모든 분야의 배움 기록 · 총 {{ til_pages | size }}개</p>
      </div>
      <span class="category-results-total" aria-hidden="true"><strong data-visible-count>{{ til_pages | size }}</strong> NOTES</span>
    </div>

    {% if til_pages.size > 0 %}
      <div class="category-board" data-category-board role="table" aria-label="TIL 게시글 목록">
        <div class="category-board-header" role="row">
          <span role="columnheader">번호</span>
          <span role="columnheader">제목</span>
          <span role="columnheader">작성일</span>
          <span role="columnheader">수정일</span>
        </div>

        <div id="category-posts" class="category-board-body" role="rowgroup">
          {% for post in til_pages %}
            {% assign post_number = til_pages.size | minus: forloop.index0 %}
            {% assign modified_date = post.updated | default: post.date %}
            <article class="category-board-row" data-category-row data-category="{{ post.category | slugify }}" role="row">
              <span class="category-board-number" data-post-number role="cell">{{ post_number }}</span>
              <div class="category-board-title-cell" role="cell">
                <a class="category-board-title" href="{{ post.url | relative_url }}">
                  <span class="category-board-title-line">
                    <strong>{{ post.title }}</strong>
                    <span class="category-board-arrow" aria-hidden="true">→</span>
                  </span>
                  <span class="category-board-description">{{ post.description }}</span>
                  <span class="category-board-label">{{ post.category }}</span>
                </a>
              </div>
              <time class="category-board-created" datetime="{{ post.date | date_to_xmlschema }}" role="cell">{{ post.date | date: "%Y.%m.%d" }}</time>
              <time class="category-board-updated" datetime="{{ modified_date | date_to_xmlschema }}" role="cell">{{ modified_date | date: "%Y.%m.%d" }}</time>
            </article>
          {% endfor %}
        </div>
      </div>
      <p class="category-filter-empty" data-category-empty hidden>아직 이 카테고리에 작성된 글이 없습니다.</p>
    {% else %}
      <p class="category-filter-empty">아직 작성된 TIL이 없습니다.</p>
    {% endif %}
  </section>
</div>

{% assign asset_version = site.github.build_revision | default: '1' %}
<script src="{{ '/assets/js/categories.js' | relative_url }}?v={{ asset_version }}" defer></script>
