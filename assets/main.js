(() => {
  'use strict';
  const config = window.SITE_CONFIG;
  const authors = document.getElementById('authors');
  config.authors.forEach(author => {
    const entry = document.createElement('span');
    entry.textContent = author.name;
    const sup = document.createElement('sup');
    sup.textContent = author.affiliations;
    entry.append(sup);
    authors.append(entry);
  });
  config.affiliations.forEach((name, index) => {
    const entry = document.createElement('span');
    const sup = document.createElement('sup');
    sup.textContent = index + 1;
    entry.append(sup, document.createTextNode(name));
    document.getElementById('affiliations').append(entry);
  });
  if (config.paperUrl) {
    const link = document.getElementById('paper-link');
    link.href = config.paperUrl;
    link.textContent = 'Read the paper ↗';
  }
  if (config.codeUrl) {
    document.getElementById('code-link').href = config.codeUrl;
    document.getElementById('resource-code').href = config.codeUrl;
  }
  document.getElementById('citation').textContent = config.citation;

  const benchmarks = ['DynaMath', 'WeMath', 'MathVerse', 'MathVista GPS', 'OlympiadBench'];
  const models = {
    canvas: { name: 'Bagel-Canvas', baseline: [42.3,35.3,55.8,77.9,24.0], gated: [48.0,38.8,58.4,81.3,26.2], flops: [75,27,25,26,37], averageGain: '3.4', averageFlops: 32 },
    zebra: { name: 'Bagel-Zebra-CoT', baseline: [39.0,34.5,48.2,81.0,16.7], gated: [43.5,38.1,51.3,83.4,18.9], flops: [45,91,75,34,40], averageGain: '3.1', averageFlops: 57 }
  };
  function showResults(key) {
    const model = models[key];
    document.querySelectorAll('[data-model]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.model === key)));
    const tbody = document.getElementById('results-body');
    tbody.replaceChildren();
    benchmarks.forEach((benchmark, index) => {
      const row = document.createElement('tr');
      const values = [benchmark, model.baseline[index].toFixed(1), model.gated[index].toFixed(1), `+${(model.gated[index]-model.baseline[index]).toFixed(1)} pp`, `−${model.flops[index]}%`];
      values.forEach((value, column) => {
        const cell = document.createElement(column === 0 ? 'th' : 'td');
        if (column === 0) { cell.scope = 'row'; cell.style.cssText = 'font-size:12px;font-weight:500;color:var(--ink);border-top:0;'; }
        if (column === 3) cell.className = 'gain';
        if (column === 4) cell.className = 'compute';
        cell.textContent = value;
        row.append(cell);
      });
      tbody.append(row);
    });
    document.getElementById('results-caption').textContent = `${model.name} benchmark accuracy and image-generation FLOPs reduction`;
    document.getElementById('average-gain').textContent = `+${model.averageGain} pp`;
    document.getElementById('average-model').textContent = `on ${model.name}`;
    document.getElementById('average-flops').textContent = `${model.averageFlops}%`;
  }
  document.querySelectorAll('[data-model]').forEach(button => button.addEventListener('click', () => showResults(button.dataset.model)));
  showResults('canvas');

  const intent = document.getElementById('intent');
  const fidelity = document.getElementById('fidelity');
  function updateGate() {
    const keep = Number(intent.value) > 0.60 || Number(fidelity.value) > 0.01;
    document.getElementById('intent-value').textContent = Number(intent.value).toFixed(2);
    document.getElementById('fidelity-value').textContent = Number(fidelity.value).toFixed(3);
    document.getElementById('gate-decision').className = `gate-decision ${keep ? 'keep' : 'abort'}`;
    document.getElementById('decision-title').textContent = keep ? 'Continue visual generation' : 'Abort visual generation';
    document.getElementById('decision-copy').textContent = keep ? 'At least one signal is strong. Let the model finish drawing.' : 'Both signals are weak. Continue reasoning in text.';
  }
  intent.addEventListener('input', updateGate);
  fidelity.addEventListener('input', updateGate);
  updateGate();

  document.getElementById('copy-citation').addEventListener('click', async () => {
    const button = document.getElementById('copy-citation');
    try {
      await navigator.clipboard.writeText(config.citation);
      button.textContent = 'Copied ✓';
      document.getElementById('copy-status').textContent = 'BibTeX copied to clipboard.';
      setTimeout(() => { button.textContent = 'Copy BibTeX ↗'; }, 2200);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById('citation'));
      selection.removeAllRanges(); selection.addRange(range);
      button.textContent = 'Selected — press Ctrl/Cmd+C';
      document.getElementById('copy-status').textContent = 'Citation selected. Copy with your keyboard.';
    }
  });
  const dialog = document.getElementById('figure-dialog');
  document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    const source = button.querySelector('img');
    const image = document.getElementById('enlarged-figure');
    image.src = source.src; image.alt = source.alt;
    dialog.showModal();
  }));
  document.getElementById('close-figure').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  const sections = [...document.querySelectorAll('section[id]')];
  const navLinks = [...document.querySelectorAll('.section-nav a')];
  function updateSection() {
    const current = sections.filter(section => section.getBoundingClientRect().top <= 180).at(-1) || sections[0];
    navLinks.forEach(link => {
      const active = link.hash === `#${current.id}`;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  window.addEventListener('scroll', updateSection, { passive: true });
  updateSection();
})();
