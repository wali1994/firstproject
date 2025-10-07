/* app.js — Controller for Two-Tower Movie Recommender (shallow + deep)
 * Option B: robust loader tries several directories for u.item / u.data
 * Requires: two-tower.js
 */

function $id(id){ return document.getElementById(id); }
const loadBtn = $id('loadBtn');
const trainBtn = $id('trainBtn');
const testBtn  = $id('testBtn');
const lossCanvas = $id('lossChart');
const embCanvas  = $id('embeddingChart');
const resultsDiv = $id('results');

function setStatus(msg) {
  const el = $id('status-line');
  if (el) el.textContent = msg; else console.log('[STATUS]', msg);
}

/* --------------------- Data holders --------------------- */
let movies = [];        // [{rawId, idx, title, genres:[19]}]
let ratings = [];       // [{uRaw, iRaw, uIdx, iIdx, rating, ts}]
let numUsers = 0, numItems = 0;
const numGenres = 19;   // MovieLens 100K genres

let userIdToIdx = new Map();
let itemIdToIdx = new Map();
let idxToMovie  = [];
let userHist    = new Map();

let allItemGenres = null;     // Float32Array [numItems * 19]
let filteredUserIdx = [];     // users with ≥ MIN_USER_RATINGS
const MIN_USER_RATINGS = 20;

/* --------------------- Models --------------------- */
let shallow = null; // TwoTowerModel useDeep=false
let deep    = null; // TwoTowerModel useDeep=true

/* --------------------- Load data (Option B) --------------------- */
async function loadData() {
  setStatus('Loading dataset…');
  trainBtn.disabled = true;
  testBtn.disabled  = true;

  // 👉 Add / edit directories to match your repo layout
  const CANDIDATE_DIRS = [
    '.',         // same folder as index.html
    './data',
    './week4',
    './ml-100k'
  ];

  async function tryFetch(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${path} → ${res.status} ${res.statusText}`);
    const txt = await res.text();
    if (!txt.trim()) throw new Error(`${path} is empty`);
    return txt;
  }

  let itemText = null, dataText = null, usedDir = null, errors = [];
  for (const dir of CANDIDATE_DIRS) {
    try {
      const [it, dt] = await Promise.all([
        tryFetch(`${dir}/u.item`),
        tryFetch(`${dir}/u.data`)
      ]);
      itemText = it; dataText = dt; usedDir = dir;
      break;
    } catch (e) {
      errors.push(e.message);
    }
  }

  if (!itemText || !dataText) {
    setStatus(
      'Load error: Could not find u.item / u.data.\n' +
      'Tried:\n' + errors.map(x => '• ' + x).join('\n')
    );
    console.error('Tried paths:\n' + errors.join('\n'));
    return;
  }

  parseItems(itemText);
  parseRatings(dataText);
  buildUserHistories();

  filteredUserIdx = Array.from(userHist.keys())
    .filter(u => userHist.get(u).length >= MIN_USER_RATINGS);

  setStatus(
    `Loaded from "${usedDir}" — ${numItems} movies, ${numUsers} users, ` +
    `${ratings.length} ratings. Users with ≥${MIN_USER_RATINGS}: ${filteredUserIdx.length}`
  );

  if (numItems > 0 && numUsers > 0 && ratings.length > 0) {
    trainBtn.disabled = false; // ✅ enable Train
  } else {
    setStatus('Data parsed but zero rows found — check file formats.');
  }
}

/* --------------------- Parsing helpers --------------------- */
function parseItems(text) {
  movies = []; itemIdToIdx.clear(); idxToMovie = [];
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 2 + numGenres) continue;
    const rawId = parseInt(parts[0], 10);
    const title = parts[1];
    const genreFlags = parts.slice(parts.length - numGenres)
      .map(v => parseInt(v, 10) || 0);
    const idx = movies.length;
    const m = { rawId, idx, title, genres: genreFlags };
    movies.push(m);
    itemIdToIdx.set(rawId, idx);
    idxToMovie[idx] = m;
  }
  numItems = movies.length;

  allItemGenres = new Float32Array(numItems * numGenres);
  for (let i = 0; i < numItems; i++) {
    for (let g = 0; g < numGenres; g++) {
      allItemGenres[i * numGenres + g] = movies[i].genres[g];
    }
  }
}

function parseRatings(text) {
  ratings = []; userIdToIdx.clear();
  const userSet = new Set();
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    const [uRawS, iRawS, rS, tsS] = line.trim().split(/\s+/);
    const uRaw = parseInt(uRawS, 10);
    const iRaw = parseInt(iRawS, 10);
    const r    = parseFloat(rS);
    const ts   = parseInt(tsS, 10) || 0;
    if (!itemIdToIdx.has(iRaw)) continue; // skip unknown movie ids
    if (!userIdToIdx.has(uRaw)) userIdToIdx.set(uRaw, userIdToIdx.size);
    const uIdx = userIdToIdx.get(uRaw);
    const iIdx = itemIdToIdx.get(iRaw);
    ratings.push({ uRaw, iRaw, uIdx, iIdx, rating: r, ts });
    userSet.add(uIdx);
  }
  numUsers = userSet.size;
}

function buildUserHistories() {
  userHist = new Map();
  ratings.forEach(({uIdx, iIdx, rating, ts}) => {
    if (!userHist.has(uIdx)) userHist.set(uIdx, []);
    userHist.get(uIdx).push({ iIdx, rating, ts });
  });
  for (const u of userHist.keys()) {
    userHist.get(u).sort((a,b)=> b.ts - a.ts); // most recent first
  }
}

/* --------------------- Training --------------------- */
async function train() {
  // Clear TF variables to avoid name collisions
  if (tf.disposeVariables) tf.disposeVariables();
  else {
    for (const n in tf.engine().registeredVariables) {
      tf.dispose(tf.engine().registeredVariables[n]);
    }
    tf.engine().registeredVariables = {};
    tf.engine().state.registeredVariables = {};
  }

  if (!ratings.length) { setStatus('Load data first.'); return; }
  setStatus('Initializing models…');
  trainBtn.disabled = true; testBtn.disabled = true;

  shallow?.dispose?.(); deep?.dispose?.();

  shallow = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32, useDeep: false, numGenres, lr: 1e-3
  });

  deep = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32, useDeep: true,
    userHidden: [64, 32], itemHidden: [64, 32],
    numGenres, userFeatDim: 0, lr: 1e-3
  });

  const trainPairs = ratings.filter(r => (userHist.get(r.uIdx)?.length || 0) >= MIN_USER_RATINGS);
  const epochs = 5, batchSize = 512;
  const lossesShallow = [], lossesDeep = [];
  const ctx = lossCanvas?.getContext('2d'); clearCanvas(lossCanvas);

  function shuffle(a){ for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]];} return a; }

  for (let ep=0; ep<epochs; ep++) {
    shuffle(trainPairs);
    let step = 0;
    for (let start=0; start<trainPairs.length; start+=batchSize) {
      const end = Math.min(start+batchSize, trainPairs.length);
      const batch = trainPairs.slice(start, end);

      const uIdx = new Array(batch.length);
      const iIdx = new Array(batch.length);
      const itemGenresBatch = new Array(batch.length);

      for (let b=0; b<batch.length; b++) {
        const r = batch[b];
        uIdx[b] = r.uIdx;
        iIdx[b] = r.iIdx;
        const row = new Array(numGenres);
        for (let g=0; g<numGenres; g++) row[g] = movies[r.iIdx].genres[g];
        itemGenresBatch[b] = row;
      }

      const l1 = await shallow.trainStep(uIdx, iIdx);
      const l2 = await deep.trainStep(uIdx, iIdx, { itemGenres: itemGenresBatch });

      lossesShallow.push(l1); lossesDeep.push(l2);
      if (ctx && (step % 10 === 0)) drawLoss(ctx, lossesShallow, lossesDeep);
      step++;
    }
    setStatus(`Epoch ${ep+1}/${epochs} — shallow: ${lossesShallow.at(-1).toFixed(4)} | deep: ${lossesDeep.at(-1).toFixed(4)}`);
  }

  // Build item indices for recommendation
  shallow.buildItemIndex();
  const allItemGenres2D = tf.tensor2d(allItemGenres, [numItems, numGenres]);
  deep.buildItemIndex(allItemGenres2D);
  allItemGenres2D.dispose();

  await drawEmbeddingPCA(deep.getItemVectors());

  setStatus('Training finished. Click Test to see recommendations.');
  testBtn.disabled = false;
  trainBtn.disabled = false;
}

/* --------------------- Testing / Results --------------------- */
async function test() {
  if (!shallow || !deep) { setStatus('Train the models first.'); return; }
  const uIdx = filteredUserIdx.length ? filteredUserIdx[Math.floor(Math.random()*filteredUserIdx.length)] : 0;

  const topRated = topNUserRated(uIdx, 10);
  const ratedSet = new Set(userHist.get(uIdx)?.map(x => x.iIdx) || []);

  const { topIdx: tS } = await shallow.scoreAllForUser(uIdx);
  const recShallow = tS.filter(i => !ratedSet.has(i)).slice(0, 10);

  const { topIdx: tD } = await deep.scoreAllForUser(uIdx);
  const recDeep = tD.filter(i => !ratedSet.has(i)).slice(0, 10);

  renderTables(uIdx, topRated, recShallow, recDeep);
}

function topNUserRated(uIdx, N=10) {
  const hist = userHist.get(uIdx) || [];
  const sorted = hist.slice().sort((a,b)=> (b.rating - a.rating) || (b.ts - a.ts));
  return sorted.slice(0, N).map(x => x.iIdx);
}

function idxToUserRaw(uIdx) {
  if (!idxToUserRaw._cache) {
    idxToUserRaw._cache = new Map();
    for (const [raw, idx] of userIdToIdx.entries()) idxToUserRaw._cache.set(idx, raw);
  }
  return idxToUserRaw._cache.get(uIdx) ?? uIdx;
}

function renderTables(uIdx, topRatedIdx, topRecIdx, topDeepIdx) {
  resultsDiv.innerHTML = '';
  const title = document.createElement('h3');
  title.textContent = `Recommendations for User ${idxToUserRaw(uIdx)}`;
  resultsDiv.appendChild(title);

  const grid = document.createElement('div');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = '1fr 1fr 1fr';
  grid.style.gap = '12px';

  grid.appendChild(buildTable('Top 10 User’s Rated Movies', topRatedIdx));
  grid.appendChild(buildTable('Top 10 Recommended (Shallow)', topRecIdx));
  grid.appendChild(buildTable('Top 10 Recommended (Deep Learning)', topDeepIdx));

  resultsDiv.appendChild(grid);
}

function buildTable(title, itemIdxList) {
  const card = document.createElement('div');
  card.style.border = '1px solid #e2e8f0';
  card.style.borderRadius = '10px';
  card.style.padding = '10px';
  const h = document.createElement('h4'); h.textContent = title; card.appendChild(h);
  const table = document.createElement('table');
  table.innerHTML = `<thead><tr><th>#</th><th>Movie</th></tr></thead>`;
  const tb = document.createElement('tbody');
  itemIdxList.forEach((iIdx, k) => {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td'); td1.textContent = String(k+1);
    const td2 = document.createElement('td'); td2.textContent = idxToMovie[iIdx]?.title || `Movie ${iIdx}`;
    tr.appendChild(td1); tr.appendChild(td2); tb.appendChild(tr);
  });
  table.appendChild(tb); card.appendChild(table); return card;
}

/* --------------------- Charts --------------------- */
function clearCanvas(cnv) {
  if (!cnv) return;
  const ctx = cnv.getContext('2d');
  ctx.clearRect(0, 0, cnv.width, cnv.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, cnv.width, cnv.height);
}

function drawLoss(ctx, s1, s2) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#fff'; ctx.fillRect(0,0,W,H);
  function series(ary, color){
    if (ary.length < 2) return;
    const mx = Math.max(...ary), mn = Math.min(...ary);
    const padX=10, padY=10;
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=2;
    for (let i=0;i<ary.length;i++){
      const x = padX + (W-2*padX)*i/(ary.length-1);
      const y = H - padY - (H-2*padY)*(ary[i]-mn)/(mx-mn+1e-8);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }
  series(s1, '#1f77b4'); // shallow
  series(s2, '#d62728'); // deep
  ctx.fillStyle='#333'; ctx.fillText('Training Loss (blue=shallow, red=deep)', 10, 14);
}

async function drawEmbeddingPCA(itemVectors) {
  if (!embCanvas) return;
  const X = itemVectors; const [N, D] = X.shape;
  const mean = X.mean(0); const Xc = X.sub(mean);
  const cov = tf.matMul(Xc.transpose(), Xc).div(N - 1);
  const { u: eigVecs } = tf.linalg.svd(cov);
  const W = eigVecs.slice([0,0],[D,2]);
  const proj = tf.matMul(Xc, W);
  const pts = await proj.array();

  const ctx = embCanvas.getContext('2d');
  clearCanvas(embCanvas);
  const w=embCanvas.width, h=embCanvas.height, pad=20;
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  ctx.fillStyle='#0a84ff33';
  for (const [x0,y0] of pts) {
    const x = pad + (w-2*pad)*(x0-minX)/(maxX-minX+1e-8);
    const y = h - pad - (h-2*pad)*(y0-minY)/(maxY-minY+1e-8);
    ctx.fillRect(x-1, y-1, 2, 2);
  }
  ctx.fillStyle='#333'; ctx.fillText('Item Embeddings (PCA)', 10, 14);

  mean.dispose(); Xc.dispose(); cov.dispose(); eigVecs.dispose(); W.dispose(); proj.dispose();
}

/* --------------------- Buttons --------------------- */
loadBtn?.addEventListener('click', async () => {
  try { await loadData(); }
  catch(e){ setStatus('Load error: '+e.message); console.error(e); }
});
trainBtn?.addEventListener('click', async () => {
  try { await train(); }
  catch(e){ setStatus('Train error: '+e.message); console.error(e); trainBtn.disabled = false; }
});
testBtn?.addEventListener('click', async () => {
  try { await test(); }
  catch(e){ setStatus('Test error: '+e.message); console.error(e); }
});

// Optional: auto-load
// document.addEventListener('DOMContentLoaded', () => loadBtn?.click());
