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

  const showRows = (visibleRows, heading, description) => {
    rows.forEach((row) => {
      row.hidden = !visibleRows.includes(row);
    });

    visibleRows.forEach((row, index) => {
      const number = row.querySelector("[data-post-number]");
      if (number) number.textContent = String(visibleRows.length - index);
    });

    const count = visibleRows.length;
    if (title) title.textContent = heading;
    if (summary) summary.textContent = `${description} · 총 ${count}개`;
    if (visibleCount) visibleCount.textContent = String(count);
    if (board) board.hidden = count === 0;
    if (emptyState) emptyState.hidden = count !== 0;
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

  window.addEventListener("hashchange", activateFromHash);

  activateFromHash();
})();
