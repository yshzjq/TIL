---
title: 카테고리
permalink: /categories/
description: 분야별로 정리한 모든 TIL 기록입니다.
---

{% assign til_pages = site.pages | where: "til", true | sort: "date" | reverse %}
{% assign date_groups = til_pages | group_by_exp: "post", "post.date | date: '%Y-%m-%d'" %}

<header class="page-header">
  <p class="eyebrow">ALL NOTES</p>
  <h1>배움 기록 모아보기</h1>
  <p>왼쪽에서 관심 분야나 작성일을 고르면 해당 게시글만 빠르게 살펴볼 수 있습니다.</p>
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

    {% if date_groups.size > 0 %}
      <div class="category-sidebar-heading category-date-heading">
        <h2 id="date-filter-title">작성일</h2>
      </div>

      <div class="category-filter-list category-date-list" aria-labelledby="date-filter-title">
        {% for date_group in date_groups %}
          <button class="category-filter-button" type="button" data-date-filter="{{ date_group.name }}" data-date-filter-button aria-controls="category-posts" aria-pressed="false">
            <span class="category-filter-icon" aria-hidden="true">📅</span>
            <span class="category-filter-name">{{ date_group.name | date: "%Y.%m.%d" }}</span>
            <span class="category-filter-count">{{ date_group.items | size }}</span>
          </button>
        {% endfor %}
      </div>
    {% endif %}
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
      <div class="category-list-toolbar">
        <label for="posts-per-page">페이지당 글 수</label>
        <select id="posts-per-page" data-page-size>
          <option value="10" selected>10개</option>
          <option value="15">15개</option>
          <option value="20">20개</option>
          <option value="25">25개</option>
          <option value="30">30개</option>
        </select>
        <span class="category-page-range" data-page-range aria-live="polite"></span>
      </div>

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
            <article class="category-board-row" data-category-row data-category="{{ post.category | slugify }}" data-created-date="{{ post.date | date: '%Y-%m-%d' }}" role="row">
              <span class="category-board-number" data-post-number role="cell">{{ post_number }}</span>
              <div class="category-board-title-cell" role="cell">
                <a class="category-board-title" href="{{ post.url | relative_url }}">
                  <span class="category-board-title-line">
                    <span class="category-board-label">{{ post.category }}</span>
                    <strong>{{ post.title }}</strong>
                    <span class="category-board-arrow" aria-hidden="true">→</span>
                  </span>
                  <span class="category-board-description">{{ post.description }}</span>
                </a>
              </div>
              <time class="category-board-created" datetime="{{ post.date | date_to_xmlschema }}" role="cell">
                <button class="category-board-date-filter" type="button" data-date-filter="{{ post.date | date: '%Y-%m-%d' }}" aria-label="{{ post.date | date: '%Y년 %m월 %d일' }}에 작성한 게시글 보기">{{ post.date | date: "%Y.%m.%d" }}</button>
              </time>
              <time class="category-board-updated" datetime="{{ modified_date | date_to_xmlschema }}" role="cell">{{ modified_date | date: "%Y.%m.%d" }}</time>
            </article>
          {% endfor %}
        </div>
      </div>
      <nav class="category-pagination" data-category-pagination aria-label="게시글 페이지 선택"></nav>
      <p class="category-filter-empty" data-category-empty hidden>선택한 조건에 해당하는 글이 없습니다.</p>
    {% else %}
      <p class="category-filter-empty">아직 작성된 TIL이 없습니다.</p>
    {% endif %}
  </section>
</div>

{% assign asset_version = site.github.build_revision | default: '1' %}
<script src="{{ '/assets/js/categories.js' | relative_url }}?v={{ asset_version }}" defer></script>
