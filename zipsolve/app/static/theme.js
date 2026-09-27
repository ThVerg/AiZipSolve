// Shared theme toggle for every page (Play / Editor / Dashboard). Classic script, load it in <head>:
//   <script src="/static/theme.js"></script>
// Applies the stored theme before first paint, and wires any [data-theme-toggle] button.
// Stored per viewer in localStorage "zip-theme" ("light" | "dark"; absent = follow the OS).
// Pages can listen for the "zip-theme" event on window to re-read CSS variables.
(function () {
  var root = document.documentElement;
  try { var t = localStorage.getItem("zip-theme"); if (t === "light" || t === "dark") root.dataset.theme = t; } catch (e) { /* ignore */ }
  function isDark() {
    return root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  var SUN = '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var MOON = '<svg class="ic" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  function paint() {
    var btns = document.querySelectorAll("[data-theme-toggle]");
    for (var i = 0; i < btns.length; i++) {
      btns[i].innerHTML = isDark() ? SUN : MOON;
      btns[i].setAttribute("aria-label", isDark() ? "Switch to light theme" : "Switch to dark theme");
      btns[i].title = isDark() ? "Light theme" : "Dark theme";
    }
  }
  function set(t) {
    root.dataset.theme = t;
    try { localStorage.setItem("zip-theme", t); } catch (e) { /* ignore */ }
    paint();
    window.dispatchEvent(new CustomEvent("zip-theme", { detail: { theme: t } }));
  }
  window.zipTheme = { isDark: isDark, set: set, toggle: function () { set(isDark() ? "light" : "dark"); } };
  document.addEventListener("click", function (e) {
    var b = e.target.closest && e.target.closest("[data-theme-toggle]");
    if (b) window.zipTheme.toggle();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
    paint();
    if (!root.dataset.theme) window.dispatchEvent(new CustomEvent("zip-theme", { detail: { theme: isDark() ? "dark" : "light" } }));
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", paint); else paint();
})();
