(() => {
  const browser = document.querySelector("[data-category-browser]");
  if (!browser) return;

  const categoryButtons = [...browser.querySelectorAll("[data-category-filter]")];
  const dateButtons = [...browser.querySelectorAll("[data-date-filter-button]")];
  const rows = [...browser.querySelectorAll("[data-category-row]")];
  const board = browser.querySelector("[data-category-board]");
  const emptyState = browser.querySelector("[data-category-empty]");
  const title = browser.querySelector("[data-category-title]");
  const summary = browser.querySelector("[data-category-summary]");
  const visibleCount = browser.querySelector("[data-visible-count]");
  const pageSizeSelect = browser.querySelector("[data-page-size]");
  const pageRange = browser.querySelector("[data-page-range]");
  const pagination = browser.querySelector("[data-category-pagination]");

  let filteredRows = rows;
  let currentPage = 1;

  const getPageSize = () => Number(pageSizeSelect?.value) || 10;

  const renderPagination = (pageCount) => {
    if (!pagination) return;

    pagination.replaceChildren();
    pagination.hidden = filteredRows.length === 0 || pageCount <= 1;

    for (let page = 1; page <= pageCount; page += 1) {
      const button = document.createElement("button");
      const isCurrent = page === currentPage;

      button.type = "button";
      button.className = "category-page-button";
      button.dataset.page = String(page);
      button.textContent = String(page);
      button.setAttribute("aria-label", `${page}페이지 보기`);
      if (isCurrent) button.setAttribute("aria-current", "page");

      pagination.append(button);
    }
  };

  const renderRows = () => {
    const count = filteredRows.length;
    const pageSize = getPageSize();
    const pageCount = Math.max(1, Math.ceil(count / pageSize));
    currentPage = Math.min(currentPage, pageCount);

    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, count);
    const pageRows = filteredRows.slice(start, end);

    rows.forEach((row) => {
      row.hidden = !pageRows.includes(row);
      row.classList.remove("page-last");
    });
    pageRows.at(-1)?.classList.add("page-last");

    filteredRows.forEach((row, index) => {
      const number = row.querySelector("[data-post-number]");
      if (number) number.textContent = String(count - index);
    });

    if (pageRange) {
      pageRange.textContent = count === 0 ? "표시할 글이 없습니다" : `${start + 1}–${end} / ${count}개`;
    }
    if (board) board.hidden = count === 0;
    if (emptyState) emptyState.hidden = count !== 0;

    renderPagination(pageCount);
  };

  const showRows = (visibleRows, heading, description) => {
    const count = visibleRows.length;
    filteredRows = visibleRows;
    currentPage = 1;

    if (title) title.textContent = heading;
    if (summary) summary.textContent = `${description} · 총 ${count}개`;
    if (visibleCount) visibleCount.textContent = String(count);

    renderRows();
  };

  const activateCategory = (requestedCategory, updateUrl = false) => {
    const activeButton =
      categoryButtons.find((button) => button.dataset.categoryFilter === requestedCategory) ||
      categoryButtons.find((button) => button.dataset.categoryFilter === "all");

    if (!activeButton) return;

    const category = activeButton.dataset.categoryFilter;
    const visibleRows = rows.filter(
      (row) => category === "all" || row.dataset.category === category
    );

    categoryButtons.forEach((button) => {
      const isActive = button === activeButton;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
    dateButtons.forEach((button) => {
      button.classList.remove("active");
      button.setAttribute("aria-pressed", "false");
    });

    const label = activeButton.dataset.categoryLabel;
    const description = activeButton.dataset.categoryDescription;
    showRows(
      visibleRows,
      category === "all" ? "전체 게시글" : `${label} 게시글`,
      description
    );

    if (updateUrl) {
      const url = new URL(window.location.href);
      url.hash = category === "all" ? "" : category;
      window.history.replaceState(null, "", url);
    }
  };

  const activateDate = (date, updateUrl = false) => {
    const visibleRows = rows.filter((row) => row.dataset.createdDate === date);
    const formattedDate = date.replaceAll("-", ".");

    categoryButtons.forEach((button) => {
      button.classList.remove("active");
      button.setAttribute("aria-pressed", "false");
    });
    dateButtons.forEach((button) => {
      const isActive = button.dataset.dateFilter === date;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });

    showRows(visibleRows, `${formattedDate} 작성 게시글`, `${formattedDate}에 작성한 배움 기록`);

    if (updateUrl) {
      const url = new URL(window.location.href);
      url.hash = `date=${date}`;
      window.history.replaceState(null, "", url);
    }
  };

  const activateFromHash = () => {
    const hash = decodeURIComponent(window.location.hash.slice(1));
    if (hash.startsWith("date=")) {
      activateDate(hash.slice(5));
      return;
    }
    activateCategory(hash || "all");
  };

  categoryButtons.forEach((button) => {
    button.addEventListener("click", () => {
      activateCategory(button.dataset.categoryFilter, true);
    });
  });

  browser.addEventListener("click", (event) => {
    const dateButton = event.target.closest("[data-date-filter]");
    if (!dateButton) return;
    activateDate(dateButton.dataset.dateFilter, true);
  });

  pageSizeSelect?.addEventListener("change", () => {
    currentPage = 1;
    renderRows();
  });

  pagination?.addEventListener("click", (event) => {
    const pageButton = event.target.closest("[data-page]");
    if (!pageButton) return;

    const requestedPage = Number(pageButton.dataset.page);
    const pageCount = Math.ceil(filteredRows.length / getPageSize());
    if (!Number.isInteger(requestedPage) || requestedPage < 1 || requestedPage > pageCount) return;

    currentPage = requestedPage;
    renderRows();
    pagination.querySelector(`[data-page="${currentPage}"]`)?.focus();
    board?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  window.addEventListener("hashchange", activateFromHash);

  activateFromHash();
})();
