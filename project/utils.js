// Minimal CSV parser (expects a header row). Handles simple, comma-separated values.
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines[0].split(',').map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    const cols = lines[i].split(',').map(c => c.trim());
    const obj = {};
    header.forEach((h, idx) => obj[h] = cols[idx]);
    rows.push(obj);
  }
  return { header, rows };
}

function downloadText(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function minMax(arr) {
  const min = Math.min(...arr), max = Math.max(...arr);
  return { min, max, range: (max - min) || 1 };
}
