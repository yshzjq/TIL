---
layout: default
title: Today I Learned
---

# 📚 Today I Learned

공부하면서 새롭게 알게 된 내용을 기록합니다.

---

## 🗂 Categories

- 🐍 Python
- 🌿 Git
- 📊 Data
- 🤖 Machine Learning
- 🧠 LLM
- 📐 Math

---

## 📝 최근 TIL

{% assign til_pages = site.pages | where: "til", true | sort: "date" | reverse %}

{% for page in til_pages %}

### [{{ page.title }}]({{ page.url | relative_url }})

**{{ page.category }}** · {{ page.date | date: "%Y-%m-%d" }}

{{ page.description }}

---

{% endfor %}