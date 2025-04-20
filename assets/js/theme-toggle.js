// Theme toggle for dark mode
(function() {
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle') || document.querySelector('.dark-mode-toggle');
  if (!toggle) return;

  // Set initial theme
  const stored = localStorage.getItem('theme');
  if (stored === 'dark') {
    root.setAttribute('data-theme', 'dark');
  } else {
    root.setAttribute('data-theme', 'light');
  }

  toggle.addEventListener('click', function() {
    const current = root.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  });
})();
