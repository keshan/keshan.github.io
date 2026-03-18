document.addEventListener('DOMContentLoaded', function() {
  const cursor = document.createElement('span');
  cursor.className = 'terminal-cursor';
  cursor.innerHTML = '_';
  
  document.querySelectorAll('.author-name').forEach(el => {
    el.appendChild(cursor.cloneNode(true));
  });
  
  const headings = document.querySelectorAll('.post-card-title, .hero h1, .post-title');
  headings.forEach((heading, index) => {
    heading.style.animationDelay = `${index * 0.1}s`;
    heading.classList.add('fade-in');
  });
  
  const links = document.querySelectorAll('a');
  links.forEach(link => {
    if (!link.closest('.share-buttons') && !link.closest('.author-social')) {
      link.setAttribute('data-before', '→ ');
    }
  });
  
  const postCards = document.querySelectorAll('.post-card');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in');
      }
    });
  }, { threshold: 0.1 });
  
  postCards.forEach(card => observer.observe(card));
});
