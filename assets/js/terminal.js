document.addEventListener('DOMContentLoaded', function() {
  const style = document.createElement('style');
  style.textContent = `
    .cursor-blink {
      display: inline-block;
      width: 10px;
      height: 1.2em;
      background: #00ff88;
      margin-left: 4px;
      animation: cursor-blink 1s step-end infinite;
      vertical-align: text-bottom;
    }
    @keyframes cursor-blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.3); opacity: 0.7; }
    }
  `;
  document.head.appendChild(style);

  const terminalTitle = document.querySelector('.terminal-title');
  if (terminalTitle && !terminalTitle.querySelector('.cursor-blink')) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor-blink';
    terminalTitle.appendChild(cursor);
  }

  document.querySelectorAll('h1, h2, h3, .post-card-title, .hero h1').forEach(el => {
    el.classList.add('glitch-hover');
  });

  const codeBlocks = document.querySelectorAll('div.highlighter-rouge, figure.highlight');
  codeBlocks.forEach(block => {
    const classList = block.className;
    const langMatch = classList.match(/language-(\w+)/);
    if (langMatch && langMatch[1] && !block.querySelector('.code-lang-label')) {
      const label = document.createElement('span');
      label.className = 'code-lang-label';
      label.textContent = langMatch[1];
      block.appendChild(label);
    }
    if (!block.querySelector('.code-copy-btn')) {
      const btn = document.createElement('button');
      btn.className = 'code-copy-btn';
      btn.textContent = 'copy';
      btn.onclick = function(e) {
        e.stopPropagation();
        const code = block.querySelector('code');
        if (code) {
          navigator.clipboard.writeText(code.textContent).then(() => {
            btn.textContent = 'copied!';
            btn.classList.add('copied');
            setTimeout(() => {
              btn.textContent = 'copy';
              btn.classList.remove('copied');
            }, 2000);
          });
        }
      };
      block.appendChild(btn);
    }
  });

  const searchOverlay = document.createElement('div');
  searchOverlay.className = 'search-overlay';
  searchOverlay.innerHTML = `
    <button class="search-close">&times;</button>
    <div class="search-container">
      <div class="search-input-wrapper">
        <div class="search-prompt"><span>keshan@blog:~$</span><span>grep</span></div>
        <input type="text" class="search-input" placeholder="Search posts..." autofocus>
      </div>
      <div class="search-results"></div>
    </div>
  `;
  document.body.appendChild(searchOverlay);

  const searchInput = searchOverlay.querySelector('.search-input');
  const searchResults = searchOverlay.querySelector('.search-results');

  const searchData = [
    { title: 'The Art of Controlled Randomness', url: '/The-Art-of-Controlled-Randomness/', excerpt: 'A Deep Dive into Sampling Techniques in LLMs' },
    { title: 'Video Analysis with Gemini', url: '/Video-analysis-with-Gemini/', excerpt: 'From Video to Structured Insight' },
    { title: 'Tensorflow Graphs and Sessions', url: '/Tensorflow-Graphs-and-Sessions/', excerpt: 'Introduction to graphs and sessions' },
    { title: 'Churn Prediction on Tensorflow', url: '/Churn-prediction-on-tensorflow/', excerpt: 'Building ML models with TensorFlow' },
    { title: 'Tensorflow Estimators', url: '/Tensorflow-Estimators/', excerpt: 'Using pre-made estimators' },
    { title: 'Listening to Hydrogen', url: '/Listening-to-Hydrogen/', excerpt: 'Speech technology exploration' },
    { title: 'Unraveling Gravitational Waves', url: '/Unraveling-Gravitational-waves/', excerpt: 'LIGO data analysis' },
  ];

  searchInput.addEventListener('input', function() {
    const query = this.value.toLowerCase();
    searchResults.innerHTML = '';
    if (query.length > 0) {
      const results = searchData.filter(post => 
        post.title.toLowerCase().includes(query) || post.excerpt.toLowerCase().includes(query)
      );
      results.forEach(post => {
        const item = document.createElement('a');
        item.href = post.url;
        item.className = 'search-result-item';
        item.innerHTML = `<h4>${post.title}</h4><p>${post.excerpt}</p>`;
        searchResults.appendChild(item);
      });
      if (results.length === 0) {
        searchResults.innerHTML = '<p style="color: #6c7086; text-align: center;">No results found</p>';
      }
    }
  });

  searchOverlay.querySelector('.search-close').addEventListener('click', () => {
    searchOverlay.classList.remove('active');
    document.body.style.overflow = '';
  });

  searchOverlay.addEventListener('click', (e) => {
    if (e.target === searchOverlay) {
      searchOverlay.classList.remove('active');
      document.body.style.overflow = '';
    }
  });

  const siteNav = document.querySelector('.site-nav');
  if (siteNav) {
    const searchBtn = document.createElement('button');
    searchBtn.className = 'search-trigger';
    searchBtn.innerHTML = 'search <kbd>/</kbd>';
    searchBtn.onclick = () => {
      searchOverlay.classList.add('active');
      document.body.style.overflow = 'hidden';
      searchInput.focus();
    };
    siteNav.appendChild(searchBtn);
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === '/' && !searchOverlay.classList.contains('active')) {
      e.preventDefault();
      searchOverlay.classList.add('active');
      document.body.style.overflow = 'hidden';
      searchInput.focus();
    }
    if (e.key === 'Escape' && searchOverlay.classList.contains('active')) {
      searchOverlay.classList.remove('active');
      document.body.style.overflow = '';
    }
  });

  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9997;';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  let particles = [];
  let mouseX = 0, mouseY = 0;

  function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    for (let i = 0; i < 2; i++) {
      particles.push({
        x: mouseX, y: mouseY,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        life: 1, size: Math.random() * 3 + 1
      });
    }
  });

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles = particles.filter(p => p.life > 0);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy; p.life -= 0.02;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 255, 136, ${p.life * 0.6})`;
      ctx.fill();
    });
    requestAnimationFrame(animate);
  }
  animate();

  const cards = document.querySelectorAll('.post-card, .publication-item, .skill-item');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  cards.forEach((card, index) => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = `opacity 0.5s ease ${index * 0.1}s, transform 0.5s ease ${index * 0.1}s`;
    observer.observe(card);
  });
});
