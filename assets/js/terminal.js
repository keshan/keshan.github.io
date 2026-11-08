/**
 * terminal.js - Interactive behaviors for keshan.github.io
 * Search, Publication Filtering, Mobile Navigation, and Code Enhancements
 */

document.addEventListener('DOMContentLoaded', function () {
  'use strict';

  // ========================================
  // 1. INSTANT SEARCH (Cmd+K or /)
  // ========================================
  const searchModal = document.getElementById('search-modal');
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  const searchToggleBtn = document.getElementById('search-toggle-btn');
  const searchCloseBtn = document.getElementById('search-close-btn');
  let searchData = null;
  let selectedIndex = -1;

  async function loadSearchData() {
    if (searchData) return searchData;
    try {
      const response = await fetch('/search.json');
      if (!response.ok) throw new Error('Failed to load search index');
      searchData = await response.json();
      return searchData;
    } catch (e) {
      console.warn('Search index load error:', e);
      return [];
    }
  }

  function openSearch() {
    if (!searchModal) return;
    searchModal.classList.add('active');
    searchModal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    loadSearchData();
    setTimeout(() => {
      if (searchInput) {
        searchInput.value = '';
        searchInput.focus();
        renderResults([]);
      }
    }, 50);
  }

  function closeSearch() {
    if (!searchModal) return;
    searchModal.classList.remove('active');
    searchModal.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    selectedIndex = -1;
  }

  function renderResults(results) {
    if (!searchResults) return;
    selectedIndex = -1;

    if (!searchInput.value.trim()) {
      searchResults.innerHTML = '<div class="search-placeholder">Type to search articles by title, topic, or keyword...</div>';
      return;
    }

    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-empty"><span class="search-empty-icon">∅</span> No matching articles found.</div>';
      return;
    }

    const html = results.map((item, index) => {
      const tags = (item.tags || []).map(t => `<span class="search-result-tag">#${t}</span>`).join(' ');
      return `
        <a href="${item.url}" class="search-result-item" data-index="${index}">
          <div class="search-result-header">
            <span class="search-result-title">${escapeHtml(item.title)}</span>
            <span class="search-result-date">${item.date || ''}</span>
          </div>
          ${item.description ? `<p class="search-result-desc">${escapeHtml(item.description)}</p>` : ''}
          ${tags ? `<div class="search-result-tags">${tags}</div>` : ''}
        </a>
      `;
    }).join('');

    searchResults.innerHTML = html;
  }

  function performSearch(query) {
    if (!searchData) return;
    const cleanQuery = query.toLowerCase().trim();
    if (!cleanQuery) {
      renderResults([]);
      return;
    }

    const matches = searchData.filter(item => {
      const titleMatch = (item.title || '').toLowerCase().includes(cleanQuery);
      const descMatch = (item.description || '').toLowerCase().includes(cleanQuery);
      const tagMatch = (item.tags || []).some(t => String(t).toLowerCase().includes(cleanQuery));
      const catMatch = (item.categories || []).some(c => String(c).toLowerCase().includes(cleanQuery));
      return titleMatch || descMatch || tagMatch || catMatch;
    });

    renderResults(matches);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  if (searchToggleBtn) {
    searchToggleBtn.addEventListener('click', openSearch);
  }

  if (searchCloseBtn) {
    searchCloseBtn.addEventListener('click', closeSearch);
  }

  if (searchModal) {
    const backdrop = searchModal.querySelector('.search-modal-backdrop');
    if (backdrop) backdrop.addEventListener('click', closeSearch);
  }

  if (searchInput) {
    searchInput.addEventListener('input', function () {
      performSearch(this.value);
    });

    searchInput.addEventListener('keydown', function (e) {
      const items = searchResults ? searchResults.querySelectorAll('.search-result-item') : [];
      if (items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = (selectedIndex + 1) % items.length;
        updateActiveResult(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = (selectedIndex - 1 + items.length) % items.length;
        updateActiveResult(items);
      } else if (e.key === 'Enter') {
        if (selectedIndex >= 0 && items[selectedIndex]) {
          e.preventDefault();
          items[selectedIndex].click();
        }
      }
    });
  }

  function updateActiveResult(items) {
    items.forEach((item, idx) => {
      if (idx === selectedIndex) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('selected');
      }
    });
  }

  // Keyboard shortcut: Press / or Cmd+K / Ctrl+K
  document.addEventListener('keydown', function (e) {
    const isEditing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) || document.activeElement.isContentEditable;

    if ((e.key === '/' && !isEditing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
      e.preventDefault();
      if (searchModal && searchModal.classList.contains('active')) {
        closeSearch();
      } else {
        openSearch();
      }
    } else if (e.key === 'Escape' && searchModal && searchModal.classList.contains('active')) {
      closeSearch();
    }
  });

  // Pre-load search index in the background after idle
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(() => loadSearchData());
  } else {
    setTimeout(loadSearchData, 2000);
  }

  // ========================================
  // 2. PUBLICATION FILTERING
  // ========================================
  const filterButtons = document.querySelectorAll('.filter-btn');
  const pubItems = document.querySelectorAll('.publication-item');

  if (filterButtons.length > 0 && pubItems.length > 0) {
    filterButtons.forEach(btn => {
      btn.addEventListener('click', function () {
        filterButtons.forEach(b => b.classList.remove('active'));
        this.classList.add('active');

        const filter = this.getAttribute('data-filter') || 'all';

        pubItems.forEach(item => {
          const itemType = item.getAttribute('data-type');
          if (filter === 'all' || itemType === filter) {
            item.style.display = '';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
          } else {
            item.style.display = 'none';
          }
        });
      });
    });
  }

  // ========================================
  // 3. MOBILE NAVIGATION TOGGLE
  // ========================================
  const navToggle = document.getElementById('mobile-nav-toggle');
  const siteNav = document.getElementById('site-nav');

  if (navToggle && siteNav) {
    navToggle.addEventListener('click', function () {
      const expanded = this.getAttribute('aria-expanded') === 'true' || false;
      this.setAttribute('aria-expanded', !expanded);
      siteNav.classList.toggle('nav-open');
      this.classList.toggle('active');
    });

    // Close nav on click outside
    document.addEventListener('click', function (e) {
      if (!navToggle.contains(e.target) && !siteNav.contains(e.target) && siteNav.classList.contains('nav-open')) {
        siteNav.classList.remove('nav-open');
        navToggle.classList.remove('active');
        navToggle.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // ========================================
  // 4. CODE COPY BUTTON (Global Fallback)
  // ========================================
  document.querySelectorAll('div.highlighter-rouge, figure.highlight').forEach(block => {
    if (block.querySelector('.code-copy-btn')) return; // already added

    const copyBtn = document.createElement('button');
    copyBtn.className = 'code-copy-btn';
    copyBtn.innerHTML = '⎘ copy';
    copyBtn.title = 'Copy code to clipboard';

    copyBtn.addEventListener('click', function () {
      const codeEl = block.querySelector('code');
      if (!codeEl) return;
      const text = codeEl.textContent;

      navigator.clipboard.writeText(text).then(() => {
        copyBtn.innerHTML = '✓ copied';
        copyBtn.classList.add('copied');
        setTimeout(() => {
          copyBtn.innerHTML = '⎘ copy';
          copyBtn.classList.remove('copied');
        }, 2000);
      }).catch(() => {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        try {
          document.execCommand('copy');
          copyBtn.innerHTML = '✓ copied';
          copyBtn.classList.add('copied');
          setTimeout(() => {
            copyBtn.innerHTML = '⎘ copy';
            copyBtn.classList.remove('copied');
          }, 2000);
        } catch (err) { }
        document.body.removeChild(textarea);
      });
    });

    block.appendChild(copyBtn);
  });
});
