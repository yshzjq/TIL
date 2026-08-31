(() => {
  const nearbyPosts = document.querySelector("[data-nearby-posts]");
  if (!nearbyPosts) return;

  const buttons = [...nearbyPosts.querySelectorAll("[data-nearby-scope-button]")];
  const panels = [...nearbyPosts.querySelectorAll("[data-nearby-scope-panel]")];
  const availableScopes = buttons.map((button) => button.dataset.nearbyScopeButton);

  const activateScope = (requestedScope, updateUrl = false) => {
    const scope = availableScopes.includes(requestedScope) ? requestedScope : "all";

    buttons.forEach((button) => {
      const isActive = button.dataset.nearbyScopeButton === scope;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.nearbyScopePanel !== scope;
    });

    if (updateUrl) {
      const url = new URL(window.location.href);
      if (scope === "all") url.searchParams.delete("scope");
      else url.searchParams.set("scope", scope);
      window.history.replaceState(null, "", url);
    }
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      activateScope(button.dataset.nearbyScopeButton, true);
    });
  });

  const requestedScope = new URL(window.location.href).searchParams.get("scope");
  activateScope(requestedScope || "all");
})();
