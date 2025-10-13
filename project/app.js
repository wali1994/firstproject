let rawRows = [];

const els = {
  file: document.getElementById('fileInput'),
  btnSample: document.getElementById('btnUseSample'),
  loadStatus: document.getElementById('loadStatus'),
  btnScore: document.getElementById('btnScore'),
  scoreStatus: document.getElementById('scoreStatus'),
  wR: document.getElementById('wRecency'),
  wF: document.getElementById('wFrequency'),
  wM: document.getElementById('wMonetary'),
  wT: document.getElementById('wTime'),
  topK: document.getElementById('topK'),
  tableBody: document.querySelector('#resultTable tbody'),
  btnExport: document.getElementById('btnExport'),
};

els.file.addEventListener('change', handleFile);
els.btnSample.addEventListener('click', loadSample);
els.btnScore.addEventListener('click', scoreNow);
els.btnExport.addEventListener('click', exportRanked);

function handleFile() {
  const file = els.file.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const text = reader.result;
    const { header, rows } = parseCSV(text);
    validateHeader(header);
    rawRows = castNumeric(rows);
    els.loadStatus.textContent = `Loaded ${rawRows.length} rows.`;
  };
  reader.onerror = () => {
    els.loadStatus.textContent = 'Error reading file.';
  };
  reader.readAsText(file);
}

function loadSample() {
  const sample = `Recency,Frequency,Monetary,Time
2,50,12500,98
6,24,5000,40
12,7,1400,15
1,70,17500,120
4,15,3200,22`;
  const { header, rows } = parseCSV(sample);
  validateHeader(header);
  rawRows = castNumeric(rows);
  els.loadStatus.textContent = `Loaded sample (${rawRows.length} rows).`;
}

function validateHeader(header) {
  const need = ['Recency','Frequency','Monetary','Time'];
  const ok = need.every(h => header.includes(h));
  if (!ok) throw new Error(`CSV must include headers: ${need.join(', ')}`);
}

function castNumeric(rows) {
  return rows.map(r => ({
    Recency: +r.Recency,
    Frequency: +r.Frequency,
    Monetary: +r.Monetary,
    Time: +r.Time
  }));
}

function scoreNow() {
  if (!rawRows.length) {
    els.scoreStatus.textContent = 'Please load data first.';
    return;
  }
  const weights = {
    wR: +els.wR.value || 0.35,
    wF: +els.wF.value || 0.30,
    wM: +els.wM.value || 0.25,
    wT: +els.wT.value || 0.10,
  };

  // Optional: auto-normalize weights to sum=1
  const sum = weights.wR + weights.wF + weights.wM + weights.wT;
  for (const k of Object.keys(weights)) weights[k] = weights[k] / (sum || 1);

  const ranked = computeScores(rawRows, weights);
  renderTable(ranked);
  els.scoreStatus.textContent = `Scored ${ranked.length} donors.`;
}

function renderTable(ranked) {
  const k = Math.max(1, +els.topK.value || 20);
  const top = ranked.slice(0, k);
  els.tableBody.innerHTML = top.map((row, i) => `
    <tr>
      <td>${i+1}</td>
      <td>${row.Recency}</td>
      <td>${row.Frequency}</td>
      <td>${row.Monetary}</td>
      <td>${row.Time}</td>
      <td>${row._score}</td>
    </tr>
  `).join('');
}

function exportRanked() {
  if (!rawRows.length) {
    els.scoreStatus.textContent = 'No data to export.';
    return;
  }
  // Re-score with current weights to guarantee up-to-date export
  const weights = {
    wR: +els.wR.value || 0.35,
    wF: +els.wF.value || 0.30,
    wM: +els.wM.value || 0.25,
    wT: +els.wT.value || 0.10,
  };
  const sum = weights.wR + weights.wF + weights.wM + weights.wT;
  for (const k of Object.keys(weights)) weights[k] = weights[k] / (sum || 1);

  const ranked = computeScores(rawRows, weights);
  const header = ['Recency','Frequency','Monetary','Time','Score'];
  const lines = [header.join(',')].concat(
    ranked.map(r => [r.Recency, r.Frequency, r.Monetary, r.Time, r._score].join(','))
  );
  downloadText('ranked_donors.csv', lines.join('\n'));
}
