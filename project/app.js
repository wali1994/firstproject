let rawRows = [];
let lastHeader = [];

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
  // LR
  lr: document.getElementById('lr'),
  epochs: document.getElementById('epochs'),
  btnTrainLR: document.getElementById('btnTrainLR'),
  trainStatus: document.getElementById('trainStatus'),
  coeffBox: document.getElementById('coeffBox'),
  btnApplyLR: document.getElementById('btnApplyLR'),
  applyStatus: document.getElementById('applyStatus'),
};

els.file.addEventListener('change', handleFile);
els.btnSample.addEventListener('click', loadSample);
els.btnScore.addEventListener('click', scoreNow);
els.btnExport.addEventListener('click', exportRanked);
els.btnTrainLR.addEventListener('click', trainLR);
els.btnApplyLR.addEventListener('click', applyLR);

function handleFile() {
  const file = els.file.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result;
      const { header, rows } = parseCSV(text);
      validateHeader(header);
      lastHeader = header;
      rawRows = castNumeric(rows, header);
      els.loadStatus.textContent = `Loaded ${rawRows.length} rows.`;
      renderEDA(rawRows); // EDA refresh
      // Reset LR state
      LR.trained = false;
      els.coeffBox.textContent = '';
      els.trainStatus.textContent = '';
      els.applyStatus.textContent = '';
    } catch (e) {
      els.loadStatus.textContent = 'Error: ' + e.message;
    }
  };
  reader.onerror = () => {
    els.loadStatus.textContent = 'Error reading file.';
  };
  reader.readAsText(file);
}

function loadSample() {
  const sample = `Recency,Frequency,Monetary,Time,Class
2,50,12500,98,1
6,24,5000,40,0
12,7,1400,15,0
1,70,17500,120,1
4,15,3200,22,0`;
  const { header, rows } = parseCSV(sample);
  validateHeader(header);
  lastHeader = header;
  rawRows = castNumeric(rows, header);
  els.loadStatus.textContent = `Loaded sample (${rawRows.length} rows).`;
  renderEDA(rawRows);
  LR.trained = false;
  els.coeffBox.textContent = '';
  els.trainStatus.textContent = '';
  els.applyStatus.textContent = '';
}

function validateHeader(header) {
  const need = ['Recency','Frequency','Monetary','Time'];
  const ok = need.every(h => header.includes(h));
  if (!ok) throw new Error(`CSV must include headers: ${need.join(', ')}`);
}

function castNumeric(rows, header) {
  // Keep any extra columns (like target) as-is, but numeric if possible
  return rows.map(r => {
    const obj = {};
    for (const k of Object.keys(r)) {
      const v = r[k];
      obj[k] = isFinite(+v) && v !== '' ? +v : v;
    }
    return obj;
  });
}

function currentWeights() {
  const w = {
    wR: +els.wR.value || 0.35,
    wF: +els.wF.value || 0.30,
    wM: +els.wM.value || 0.25,
    wT: +els.wT.value || 0.10,
  };
  const sum = w.wR + w.wF + w.wM + w.wT;
  for (const k of Object.keys(w)) w[k] = w[k] / (sum || 1);
  return w;
}

function scoreNow() {
  if (!rawRows.length) { els.scoreStatus.textContent = 'Please load data first.'; return; }
  const ranked = computeScores(rawRows, currentWeights());
  renderTable(ranked);
  els.scoreStatus.textContent = `Scored ${ranked.length} donors (weighted linear).`;
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
  if (!rawRows.length) { els.scoreStatus.textContent = 'No data to export.'; return; }
  const ranked = computeScores(rawRows, currentWeights());
  const header = ['Recency','Frequency','Monetary','Time','Score'];
  const lines = [header.join(',')].concat(
    ranked.map(r => [r.Recency, r.Frequency, r.Monetary, r.Time, r._score].join(','))
  );
  downloadText('ranked_donors.csv', lines.join('\n'));
}

/** Logistic Regression */
function trainLR(){
  if (!rawRows.length) { els.trainStatus.textContent = 'Load data first.'; return; }
  const { X, y, yName, note } = prepareXY(rawRows, lastHeader);
  if (!X || !y) { els.trainStatus.textContent = note || 'No target column detected (need 0/1 labels).'; return; }

  els.trainStatus.textContent = `${note} Training...`;
  const opts = { lr: +els.lr.value || 0.05, epochs: +els.epochs.value || 200 };
  const { beta0, beta } = trainLogistic(X, y, opts);
  LR.beta0 = beta0; LR.beta = beta; LR.trained = true;
  els.trainStatus.textContent = `Trained Logistic Regression (epochs=${opts.epochs}, lr=${opts.lr}).`;
  els.coeffBox.textContent =
`Logistic Regression coefficients:
Intercept (beta0): ${beta0.toFixed(4)}
Recency (norm):    ${beta[0].toFixed(4)}  (lower raw Recency => higher normalized; note MVP inverted Recency)
Frequency (norm):  ${beta[1].toFixed(4)}
Monetary (norm):   ${beta[2].toFixed(4)}
Time (norm):       ${beta[3].toFixed(4)}`;
}

function applyLR(){
  if (!LR.trained) { els.applyStatus.textContent = 'Train the logistic model first.'; return; }
  if (!rawRows.length) { els.applyStatus.textContent = 'Load data first.'; return; }
  const probs = applyLogistic(rawRows, LR);
  // Attach calibrated score and re-render
  const enriched = rawRows.map((r,i) => ({...r, _score: +probs[i].toFixed(4)})).sort((a,b)=>b._score - a._score);
  renderTable(enriched);
  els.applyStatus.textContent = `Applied calibrated scores (logistic probability).`;
}
