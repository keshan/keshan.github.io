document.addEventListener('DOMContentLoaded', function() {
  const style = document.createElement('style');
  style.textContent = `
    .cursor-blink {
      display: inline-block;
      width: 10px;
      height: 1.2em;
      background: var(--term-green, #00ff88);
      margin-left: 4px;
      animation: cursor-blink 1s step-end infinite;
      vertical-align: text-bottom;
    }
    
    @keyframes cursor-blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }
    
    .typing-effect {
      overflow: hidden;
      white-space: nowrap;
      animation: typing 2s steps(30, end);
    }
    
    @keyframes typing {
      from { width: 0; }
      to { width: 100%; }
    }
    
    .glitch-text {
      position: relative;
    }
    
    .glitch-text:hover {
      animation: glitch 0.3s ease-in-out infinite;
    }
    
    @keyframes glitch {
      0% { transform: translate(0); }
      20% { transform: translate(-2px, 2px); }
      40% { transform: translate(-2px, -2px); }
      60% { transform: translate(2px, 2px); }
      80% { transform: translate(2px, -2px); }
      100% { transform: translate(0); }
    }
    
    .fade-in-up {
      animation: fadeInUp 0.5s ease-out forwards;
      opacity: 0;
    }
    
    @keyframes fadeInUp {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .scanline {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 9999;
      background: repeating-linear-gradient(
        0deg,
        rgba(0, 0, 0, 0.1) 0px,
        rgba(0, 0, 0, 0.1) 1px,
        transparent 1px,
        transparent 2px
      );
      animation: scanline-move 8s linear infinite;
    }
    
    @keyframes scanline-move {
      0% { transform: translateY(0); }
      100% { transform: translateY(2px); }
    }
    
    .terminal-prompt {
      display: inline-flex;
      align-items: center;
    }
    
    .terminal-prompt::after {
      content: '_';
      color: var(--term-green, #00ff88);
      animation: cursor-blink 1s step-end infinite;
    }
    
    .pulse-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      background: var(--term-green, #00ff88);
      border-radius: 50%;
      animation: pulse 2s ease-in-out infinite;
      box-shadow: 0 0 10px var(--term-green, #00ff88);
    }
    
    @keyframes pulse {
      0%, 100% { 
        transform: scale(1);
        opacity: 1;
      }
      50% { 
        transform: scale(1.3);
        opacity: 0.7;
      }
    }
  `;
  document.head.appendChild(style);

  const terminalTitle = document.querySelector('.terminal-title');
  if (terminalTitle && !terminalTitle.querySelector('.cursor-blink')) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor-blink';
    terminalTitle.appendChild(cursor);
  }

  const headings = document.querySelectorAll('h1, h2, h3');
  headings.forEach((heading, index) => {
    if (!heading.querySelector('.cursor-blink') && index < 3) {
      const cursor = document.createElement('span');
      cursor.className = 'cursor-blink';
      cursor.style.fontSize = '0.6em';
      cursor.style.height = '0.8em';
      heading.appendChild(cursor);
    }
  });

  const links = document.querySelectorAll('.site-nav a');
  links.forEach(link => {
    link.classList.add('fade-in-up');
  });

  const cards = document.querySelectorAll('.post-card, .publication-item, .skill-item');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, index) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, index * 100);
      }
    });
  }, { threshold: 0.1 });

  cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(card);
  });
});
