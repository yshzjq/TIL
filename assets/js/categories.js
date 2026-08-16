(() => {
  const browser = document.querySelector("[data-category-browser]");
  if (!browser) return;

  const buttons = [...browser.querySelectorAll("[data-category-filter]")];
  const rows = [...browser.querySelectorAll("[data-category-row]")];
  const board = browser.querySelector("[data-category-board]");
  const emptyState = browser.querySelector("[data-category-empty]");
  const title = browser.querySelector("[data-category-title]");
  const summary = browser.querySelector("[data-category-summary]");
  const visibleCount = browser.querySelector("[data-visible-count]");

  const activateCategory = (requestedCategory, updateUrl = false) => {
    const activeButton =
      buttons.find((button) => button.dataset.categoryFilter === requestedCategory) ||
      buttons.find((button) => button.dataset.categoryFilter === "all");

    if (!activeButton) return;

    const category = activeButton.dataset.categoryFilter;
    const visibleRows = rows.filter(
      (row) => category === "all" || row.dataset.category === category
    );

    buttons.forEach((button) => {
      const isActive = button === activeButton;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });

    rows.forEach((row) => {
      row.hidden = !visibleRows.includes(row);
    });

    visibleRows.forEach((row, index) => {
      const number = row.querySelector("[data-post-number]");
      if (number) number.textContent = String(visibleRows.length - index);
    });

    const label = activeButton.dataset.categoryLabel;
    const description = activeButton.dataset.categoryDescription;
    const count = visibleRows.length;

    if (title) title.textContent = category === "all" ? "전체 게시글" : `${label} 게시글`;
    if (summary) summary.textContent = `${description} · 총 ${count}개`;
    if (visibleCount) visibleCount.textContent = String(count);
    if (board) board.hidden = count === 0;
    if (emptyState) emptyState.hidden = count !== 0;

    if (updateUrl) {
      const url = new URL(window.location.href);
      url.hash = category === "all" ? "" : category;
      window.history.replaceState(null, "", url);
    }
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      activateCategory(button.dataset.categoryFilter, true);
    });
  });

  window.addEventListener("hashchange", () => {
    activateCategory(window.location.hash.slice(1) || "all");
  });

  activateCategory(window.location.hash.slice(1) || "all");
})();
