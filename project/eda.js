let edaCharts = {};

/** Build histograms and correlation heatmap */
function renderEDA(rows) {
  if (!rows?.length) return;

  const features = ['Recency','Frequency','Monetary','Time'];
  const dataCols = {};
  features.forEach(f => dataCols[f] = rows.map(r => +r[f]));

  // Histograms
  drawHist('histRecency', dataCols['Recency'], 'Recency');
  drawHist('histFrequency', dataCols['Frequency'], 'Frequency');
  drawHist('histMonetary', dataCols['Monetary'], 'Monetary');
  drawHist('histTime',     dataCols['Time'],     'Time');

  // Correlation matrix
  const corr = corrMatrix(dataCols, features);
  drawHeatmap('corrHeatmap', corr, features);
  const note = document.getElementById('corrNote');
  note.textContent = 'Pearson correlation on raw features — darker = stronger correlation.';
}

function drawHist(canvasId, arr, label) {
  const {bins, counts} = histogram(arr, 10);
  const ctx = document.getElementById(canvasId);
  if (edaCharts[canvasId]) edaCharts[canvasId].destroy();
  edaCharts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bins.map(b => `${b.lo.toFixed(0)}–${b.hi.toFixed(0)}`),
      datasets: [{ label: label, data: counts }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { autoSkip: true, maxTicksLimit: 10 } } }
    }
  });
}

function histogram(arr, k=10) {
  const min = Math.min(...arr), max = Math.max(...arr);
  const width = (max - min) / k || 1;
  const bins = Array.from({length:k}, (_,i)=>({lo: min + i*width, hi: min + (i+1)*width}));
  const counts = new Array(k).fill(0);
  for (const v of arr) {
    let idx = Math.floor((v - min) / width);
    if (idx >= k) idx = k-1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  return {bins, counts};
}

function corrMatrix(cols, features) {
  const n = features.length;
  const mat = Array.from({length:n}, () => Array(n).fill(0));
  for (let i=0;i<n;i++){
    for (let j=0;j<n;j++){
      mat[i][j] = pearson(cols[features[i]], cols[features[j]]);
    }
  }
  return mat;
}

function pearson(x, y) {
  const n = x.length;
  const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
  const mx = mean(x), my = mean(y);
  let num=0, dx=0, dy=0;
  for (let i=0;i<n;i++){
    const ax = x[i]-mx, ay = y[i]-my;
    num += ax*ay; dx += ax*ax; dy += ay*ay;
  }
  const den = Math.sqrt(dx*dy) || 1;
  return num/den;
}

function drawHeatmap(canvasId, matrix, labels) {
  // Render as a bubble-like heatmap using Chart.js scatter + sizes based on |corr|
  const ctx = document.getElementById(canvasId);
  if (edaCharts[canvasId]) edaCharts[canvasId].destroy();

  // Flatten points
  const points = [];
  const n = labels.length;
  for (let i=0;i<n;i++){
    for (let j=0;j<n;j++){
      const v = matrix[i][j];
      points.push({x:j+1, y:n-i, v});
    }
  }
  edaCharts[canvasId] = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [{
        label: 'Correlation',
        data: points.map(p => ({x:p.x, y:p.y, r: 8 + Math.abs(p.v)*14}))
      }]
    },
    options: {
      plugins: { legend: { display:false }, tooltip: {
        callbacks: { label: ctx => ` ${labels[n-ctx.parsed.y]} vs ${labels[ctx.parsed.x-1]}: ${points[(ctx.dataIndex)].v.toFixed(2)}` }
      }},
      scales: {
        x: { min: 0.5, max: n+0.5, ticks:{ callback: (_,i)=> labels[i] } },
        y: { min: 0.5, max: n+0.5, ticks:{ callback: (_,i)=> labels[n-i] } }
      }
    }
  });
}
