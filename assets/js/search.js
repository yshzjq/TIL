(() => {
  const root = document.querySelector("[data-search]");
  if (!root) return;

  const form = root.querySelector(".search-form");
  const input = root.querySelector("#site-search");
  const clearButton = root.querySelector(".search-clear");
  const panel = root.querySelector(".search-results");
  const status = root.querySelector(".search-status");
  const resultList = root.querySelector(".search-result-list");

  let documents = null;
  let loading = null;
  let currentResults = [];
  let selectedIndex = -1;

  const normalize = (value) =>
    String(value || "")
      .normalize("NFKC")
      .toLocaleLowerCase("ko-KR")
      .replace(/\s+/g, " ")
      .trim();

  const loadDocuments = () => {
    if (documents) return Promise.resolve(documents);
    if (loading) return loading;

    loading = fetch(root.dataset.indexUrl, { credentials: "same-origin" })
      .then((response) => {
        if (!response.ok) throw new Error("검색 색인을 불러오지 못했습니다.");
        return response.json();
      })
      .then((items) => {
        documents = items.map((item) => ({
          ...item,
          searchText: normalize(
            `${item.title} ${item.description} ${item.category} ${item.content}`
          ),
          normalizedTitle: normalize(item.title),
          normalizedCategory: normalize(item.category),
          normalizedDescription: normalize(item.description),
        }));
        return documents;
      });

    return loading;
  };

  const closeResults = () => {
    panel.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    selectedIndex = -1;
  };

  const openResults = () => {
    panel.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };

  const updateSelection = (nextIndex) => {
    const links = [...resultList.querySelectorAll(".search-result")];
    if (!links.length) return;

    selectedIndex = (nextIndex + links.length) % links.length;
    links.forEach((link, index) => {
      const selected = index === selectedIndex;
      link.classList.toggle("selected", selected);
      link.setAttribute("aria-selected", String(selected));
    });

    const selectedLink = links[selectedIndex];
    input.setAttribute("aria-activedescendant", selectedLink.id);
    selectedLink.scrollIntoView({ block: "nearest" });
  };

  const makeResult = (item, index) => {
    const link = document.createElement("a");
    link.className = "search-result";
    link.id = `search-result-${index}`;
    link.href = item.url;
    link.setAttribute("role", "option");
    link.setAttribute("aria-selected", "false");

    const meta = document.createElement("span");
    meta.className = "search-result-meta";
    meta.textContent = `${item.category} · ${item.date}`;

    const title = document.createElement("strong");
    title.textContent = item.title;

    const description = document.createElement("span");
    description.className = "search-result-description";
    description.textContent = item.description;

    link.append(meta, title, description);
    return link;
  };

  const renderResults = (items) => {
    currentResults = items;
    selectedIndex = -1;
    input.removeAttribute("aria-activedescendant");
    resultList.replaceChildren();

    if (!items.length) {
      status.textContent = "검색 결과가 없습니다.";
      openResults();
      return;
    }

    status.textContent = `${items.length}개의 검색 결과`;
    items.forEach((item, index) => resultList.append(makeResult(item, index)));
    openResults();
  };

  const search = async () => {
    const query = normalize(input.value);
    clearButton.hidden = query.length === 0;

    if (!query) {
      currentResults = [];
      resultList.replaceChildren();
      closeResults();
      return;
    }

    status.textContent = "검색 중…";
    openResults();

    try {
      const items = await loadDocuments();
      const terms = query.split(" ");
      const matches = items
        .filter((item) => terms.every((term) => item.searchText.includes(term)))
        .map((item) => {
          let score = 0;
          if (item.normalizedTitle === query) score += 120;
          else if (item.normalizedTitle.startsWith(query)) score += 90;
          else if (item.normalizedTitle.includes(query)) score += 60;
          if (item.normalizedCategory.includes(query)) score += 35;
          if (item.normalizedDescription.includes(query)) score += 20;
          return { ...item, score };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, 6);

      renderResults(matches);
    } catch (error) {
      currentResults = [];
      resultList.replaceChildren();
      status.textContent = "검색을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.";
      openResults();
    }
  };

  input.addEventListener("input", search);
  input.addEventListener("focus", () => {
    if (input.value.trim()) search();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      updateSelection(selectedIndex + 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      updateSelection(selectedIndex - 1);
    } else if (event.key === "Escape") {
      closeResults();
      input.blur();
    }
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!currentResults.length) return;
    const targetIndex = selectedIndex >= 0 ? selectedIndex : 0;
    window.location.href = currentResults[targetIndex].url;
  });

  clearButton.addEventListener("click", () => {
    input.value = "";
    clearButton.hidden = true;
    closeResults();
    input.focus();
  });

  document.addEventListener("click", (event) => {
    if (!root.contains(event.target)) closeResults();
  });
})();
