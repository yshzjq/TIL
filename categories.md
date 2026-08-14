---
title: 카테고리
permalink: /categories/
description: 분야별로 정리한 모든 TIL 기록입니다.
---

{% assign til_pages = site.pages | where: "til", true | sort: "date" | reverse %}

<header class="page-header">
  <p class="eyebrow">ALL NOTES</p>
  <h1>카테고리</h1>
  <p>관심 있는 분야를 골라 지금까지의 배움 기록을 살펴보세요.</p>
</header>

<div class="category-sections">
  {% for item in site.data.categories %}
    {% assign category_posts = til_pages | where: "category", item.name %}
    <section class="category-section" id="{{ item.slug }}">
      <div class="category-section-heading">
        <span class="category-icon" aria-hidden="true">{{ item.icon }}</span>
        <div>
          <h2>{{ item.name }}</h2>
          <p>{{ item.description }} · {{ category_posts | size }}개의 기록</p>
        </div>
      </div>

      {% if category_posts.size > 0 %}
        <ol class="archive-list">
          {% for post in category_posts %}
            <li>
              <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
              <a href="{{ post.url | relative_url }}">
                <strong>{{ post.title }}</strong>
                <span>{{ post.description }}</span>
              </a>
              <span class="archive-arrow" aria-hidden="true">→</span>
            </li>
          {% endfor %}
        </ol>
      {% else %}
        <p class="category-empty">아직 이 분야의 기록이 없습니다.</p>
      {% endif %}
    </section>
  {% endfor %}
</div>
