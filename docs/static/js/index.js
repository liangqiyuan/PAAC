function copyBibTeX() {
  const code = document.getElementById('bibtex-code');
  const button = document.querySelector('.copy-bibtex-btn');
  const label = button.querySelector('.copy-text');
  navigator.clipboard.writeText(code.textContent).then(function () {
    button.classList.add('copied');
    label.textContent = 'Copied';
    setTimeout(function () {
      button.classList.remove('copied');
      label.textContent = 'Copy';
    }, 2000);
  });
}

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

window.addEventListener('scroll', function () {
  document.querySelector('.scroll-to-top').classList.toggle('visible', window.pageYOffset > 300);
});
