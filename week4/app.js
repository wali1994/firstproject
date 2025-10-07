/* app.js — Controller for Two-Tower Movie Recommender
 * Works with two-tower.js provided earlier (supports shallow & deep towers).
 *
 * UI:
 *  - Buttons (any of these ids will be detected):
 *      Load:  loadBtn | loadDataBtn | loadData
 *      Train: trainBtn | train
 *      Test:  testBtn  | test
 *  - Canvases: #lossChart, #embeddingChart
 *  - Container: #results
 */

///////////////////////
// DOM helpers
///////////////////////
function byIdAny(...ids) { return ids.map(id => document.getElementById(id)).find(Boolean); }
const loadBtn  = byIdAny('loadBtn', 'loadDataBtn', 'loadData');
const trainBtn = byIdAny('trainBtn', 'train');
const testBtn  = byIdAny('testBtn', 'test');
const lossCanvas = document.getElementById('lossChart');
const embCanvas  = document.getElementById('embeddingChart');
const resultsDiv = document.getElementById('results');

function setStatus(msg) {
  if (resultsDiv) {
    const id = 'status-line';
    let el = document.getElementById(id);
    if (!el) { el = document.createElement('div'); el.id = id; resultsDiv.prepend(el); }
    el.textContent = msg;
  } else {
    console.log('[STATUS]', msg);
  }
}

///////////////////////
// Data structures
///////////////////////
let movies = [];        // [{rawId, idx, title, genres:[19]}]
let ratings = [];       // [{uRaw, iRaw, uIdx, iIdx, rating, ts}]
let numUsers = 0;
let numItems = 0;
const numGenres = 19;   // ML-100K: last 19 flags in u.item

let userIdToIdx = new Map(); // raw userId -> 0-based
let itemIdToIdx = new Map(); // raw movieId -> 0-based
let idxToMovie = [];         // idx -> movie object
let userHist = new Map();    // uIdx -> [{iIdx, rating, ts}, ...]

let allItemGenres = null;    // [numItems, 19] Float32Array
let filteredUserIdx = [];    // users with >= minRatings
const MIN_USER_RATINGS = 20;

///////////////////////
// Models
///////////////////////
let shallow = null; // TwoTowerModel (embed-dot)
let deep    = null; // TwoTowerModel (MLP towers)

///////////////////////
// Load Dataset
///////////////////////
async function loadData() {
  setStatus('Loading dataset (u.item & u.data)...');
  const itemRes = await fetch('u.item');
  const dataRes = await fetch('u.data');
  if (!itemRes.ok) throw new Error(`Failed to fetch u.item (${itemRes.status})`);
  if (!dataRes.ok) throw new Error(`Failed to fetch u.data (${dataRes.status})`);
  const itemText = await itemRes.text();
  const dataText = await dataRes.text();

  parseItems(itemText);
  parseRatings(dataText);
  buildUserHistories();

  filteredUserIdx = Array.from(userHist.keys()).filter(u => userHist.get(u).length >= MIN_USER_RATINGS);

  setStatus(`Loaded ${numItems} movies, ${numUsers} users, ${ratings.length} ratings. Users with ≥${MIN_USER_RATINGS} ratings: ${filteredUserIdx.length}`);
  if (trainBtn) trainBtn.disabled = false;
  if (testBtn) testBtn.disabled = true;
}

function parseItems(text) {
  movies = [];
  itemIdToIdx.clear();
  idxToMovie = [];
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 2 + numGenres) continue;
    const rawId = parseInt(parts[0], 10);
    const title = parts[1];
    const genreFlags = parts.slice(parts.length - numGenres).map(v => parseInt(v, 10) || 0);
    const idx = movies.length;
    const m = { rawId, idx, title, genres: genreFlags };
    movies.push(m);
    itemIdToIdx.set(rawId, idx);
    idxToMovie[idx] = m;
  }
  numItems = movies.length;

  // Build allItemGenres matrix
  allItemGenres = new Float32Array(numItems * numGenres);
  for (let i = 0; i < numItems; i++) {
    for (let g = 0; g < numGenres; g++) {
      allItemGenres[i * numGenres + g] = movies[i].genres[g];
    }
  }
}

function parseRatings(text) {
  ratings = [];
  userIdToIdx.clear();
  const userSet = new Set();

  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    const [uRawS, iRawS, rS, tsS] = line.trim().split(/\s+/);
    const uRaw = parseInt(uRawS, 10);
    const iRaw = parseInt(iRawS, 10);
    const r    = parseFloat(rS);
    const ts   = parseInt(tsS, 10) || 0;
    if (!itemIdToIdx.has(iRaw)) continue; // guard
    // map to 0-based indices
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
  // sort each by timestamp desc for recency
  for (const u of userHist.keys()) {
    userHist.get(u).sort((a,b)=> b.ts - a.ts);
  }
}

///////////////////////
// Training
///////////////////////
async function train() {
  // reset TensorFlow.js variable registry
  for (const name in tf.engine().registeredVariables) {
    tf.dispose(tf.engine().registeredVariables[name]);
  }
  tf.engine().registeredVariables = {};
  tf.engine().state.registeredVariables = {};

  if (!ratings.length) throw new Error('Load data first.');
  setStatus('Initializing models...');
  ...
}
  if (!ratings.length) throw new Error('Load data first.');
  setStatus('Initializing models...');

  shallow?.dispose?.();
  deep?.dispose?.();

  // init models
  shallow = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32,
    useDeep: false,
    numGenres,  // not used in shallow tower, but ok to pass
    lr: 1e-3
  });

  deep = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32,
    useDeep: true,
    userHidden: [64, 32],
    itemHidden: [64, 32],
    numGenres,      // use genres in item tower
    userFeatDim: 0, // set >0 if you later add user features
    lr: 1e-3
  });

  // create train list only from users with enough ratings
  const trainPairs = ratings.filter(r => userHist.get(r.uIdx)?.length >= MIN_USER_RATINGS);

  // simple shuffle
  function shuffle(a) {
    for (let i=a.length-1; i>0; i--) { const j = Math.floor(Math.random()*(i+1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  const epochs = 5;          // adjust as you like
  const batchSize = 512;     // pairs per step
  const lossesShallow = [];
  const lossesDeep = [];

  const ctx = lossCanvas?.getContext('2d');
  clearCanvas(lossCanvas);

  for (let ep = 0; ep < epochs; ep++) {
    shuffle(trainPairs);
    let step = 0;
    for (let start = 0; start < trainPairs.length; start += batchSize) {
      const end = Math.min(start + batchSize, trainPairs.length);
      const batch = trainPairs.slice(start, end);

      const uIdx = new Array(batch.length);
      const iIdx = new Array(batch.length);
      const itemGenresBatch = new Array(batch.length);

      for (let b = 0; b < batch.length; b++) {
        const r = batch[b];
        uIdx[b] = r.uIdx;
        iIdx[b] = r.iIdx;
        // pick genres row for item
        const row = new Array(numGenres);
        for (let g=0; g<numGenres; g++) {
          row[g] = movies[r.iIdx].genres[g];
        }
        itemGenresBatch[b] = row;
      }

      const l1 = await shallow.trainStep(uIdx, iIdx);
      const l2 = await deep.trainStep(uIdx, iIdx, { itemGenres: itemGenresBatch });

      lossesShallow.push(l1);
      lossesDeep.push(l2);

      if ((step % 10 === 0) && ctx) drawLoss(ctx, lossesShallow, lossesDeep);
      step++;
    }
    setStatus(`Epoch ${ep+1}/${epochs} — shallow loss: ${lossesShallow.at(-1).toFixed(4)} | deep loss: ${lossesDeep.at(-1).toFixed(4)}`);
  }

  // Build item indices for recommendation
  shallow.buildItemIndex(); // shallow needs no genres
  const allItemGenres2D = tf.tensor2d(allItemGenres, [numItems, numGenres]);
  deep.buildItemIndex(allItemGenres2D);
  allItemGenres2D.dispose();

  // Draw item embedding PCA (using deep vectors)
  await drawEmbeddingPCA(deep.getItemVectors());

  if (testBtn) testBtn.disabled = false;
}

///////////////////////
// Testing / Display
///////////////////////
async function test() {
  if (!shallow || !deep) { setStatus('Train the models first.'); return; }

  // choose a user with many ratings; you can fix e.g., 60 if you like
  const uIdx = filteredUserIdx.length ? filteredUserIdx[Math.floor(Math.random()*filteredUserIdx.length)] : 0;

  const topRated = topNUserRated(uIdx, 10);
  const ratedSet = new Set(userHist.get(uIdx)?.map(x => x.iIdx) || []);

  const { topIdx: topShallow } = await shallow.scoreAllForUser(uIdx);
  const recShallow = topShallow.filter(i => !ratedSet.has(i)).slice(0, 10);

  const { topIdx: topDeep } = await deep.scoreAllForUser(uIdx);
  const recDeep = topDeep.filter(i => !ratedSet.has(i)).slice(0, 10);

  renderTables(uIdx, topRated, recShallow, recDeep);
  setStatus(`Shown recommendations for User ${idxToUserRaw(uIdx)} (internal idx ${uIdx}).`);
}

function topNUserRated(uIdx, N=10) {
  const hist = userHist.get(uIdx) || [];
  // sort by rating desc, then recency
  const sorted = hist.slice().sort((a,b) => (b.rating - a.rating) || (b.ts - a.ts));
  return sorted.slice(0, N).map(x => x.iIdx);
}

function idxToUserRaw(uIdx) {
  // find raw id from map (reverse lookup once)
  if (!idxToUserRaw._cache) {
    idxToUserRaw._cache = new Map();
    for (const [raw, idx] of userIdToIdx.entries()) idxToUserRaw._cache.set(idx, raw);
  }
  return idxToUserRaw._cache.get(uIdx) ?? uIdx;
}

function renderTables(uIdx, topRatedIdx, topRecIdx, topDeepIdx) {
  if (!resultsDiv) return;
  resultsDiv.innerHTML = '';

  const title = document.createElement('h3');
  title.textContent = `Recommendations for User ${idxToUserRaw(uIdx)}`;
  resultsDiv.appendChild(title);

  const wrapper = document.createElement('div');
  wrapper.style.display = 'grid';
  wrapper.style.gridTemplateColumns = '1fr 1fr 1fr';
  wrapper.style.gap = '12px';

  const t1 = buildTable('Top 10 User’s Rated Movies', topRatedIdx);
  const t2 = buildTable('Top 10 Recommended (Shallow)', topRecIdx);
  const t3 = buildTable('Top 10 Recommended (Deep Learning)', topDeepIdx);

  wrapper.appendChild(t1);
  wrapper.appendChild(t2);
  wrapper.appendChild(t3);
  resultsDiv.appendChild(wrapper);
}

function buildTable(title, itemIdxList) {
  const card = document.createElement('div');
  card.style.border = '1px solid #ddd';
  card.style.borderRadius = '10px';
  card.style.padding = '10px';
  const h = document.createElement('h4'); h.textContent = title; card.appendChild(h);
  const table = document.createElement('table');
  table.style.width = '100%';
  table.innerHTML = `<thead><tr><th>#</th><th>Movie</th></tr></thead>`;
  const tb = document.createElement('tbody');
  itemIdxList.forEach((iIdx, k) => {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td'); td1.textContent = (k+1).toString();
    const td2 = document.createElement('td'); td2.textContent = idxToMovie[iIdx]?.title || `Movie ${iIdx}`;
    tr.appendChild(td1); tr.appendChild(td2);
    tb.appendChild(tr);
  });
  table.appendChild(tb);
  card.appendChild(table);
  return card;
}

///////////////////////
// Charts
///////////////////////
function clearCanvas(cnv) {
  if (!cnv) return;
  const ctx = cnv.getContext('2d');
  ctx.clearRect(0, 0, cnv.width, cnv.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, cnv.width, cnv.height);
}

function drawLoss(ctx, lossesShallow, lossesDeep) {
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#fff'; ctx.fillRect(0,0,W,H);

  function drawSeries(series, color) {
    if (series.length < 2) return;
    const max = Math.max(...series);
    const min = Math.min(...series);
    const padX = 10, padY = 10;
    ctx.beginPath();
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    for (let i = 0; i < series.length; i++) {
      const x = padX + (W - 2*padX) * i / (series.length - 1);
      const y = H - padY - (H - 2*padY) * (series[i] - min) / (max - min + 1e-8);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawSeries(lossesShallow, '#1f77b4'); // blue
  drawSeries(lossesDeep,    '#d62728'); // red

  ctx.fillStyle = '#333';
  ctx.fillText('Training Loss (blue=shallow, red=deep)', 10, 14);
}

async function drawEmbeddingPCA(itemVectors) {
  if (!embCanvas) return;
  // itemVectors: tf.Tensor [N, D]
  const X = itemVectors; // use deep vectors
  const [N, D] = X.shape;
  // center
  const mean = X.mean(0);
  const Xc = X.sub(mean);
  // PCA via SVD of covariance (Xc^T Xc)
  const cov = tf.matMul(Xc.transpose(), Xc).div(N - 1);
  const { u: eigVecs } = tf.linalg.svd(cov); // columns are eigenvectors
  const W = eigVecs.slice([0,0],[D,2]); // take first 2 PCs
  const proj = tf.matMul(Xc, W);        // [N,2]
  const pts = await proj.array();

  // draw
  const ctx = embCanvas.getContext('2d');
  clearCanvas(embCanvas);
  const w = embCanvas.width, h = embCanvas.height;
  // scale to canvas
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 20;
  ctx.fillStyle = '#0a84ff33';
  for (const [x0,y0] of pts) {
    const x = pad + (w - 2*pad) * (x0 - minX) / (maxX - minX + 1e-8);
    const y = h - pad - (h - 2*pad) * (y0 - minY) / (maxY - minY + 1e-8);
    ctx.fillRect(x-1, y-1, 2, 2);
  }
  ctx.fillStyle = '#333';
  ctx.fillText('Item Embeddings (PCA)', 10, 14);

  // cleanup
  mean.dispose(); Xc.dispose(); cov.dispose(); eigVecs.dispose(); W.dispose(); proj.dispose();
}

///////////////////////
// Wire up buttons
///////////////////////
if (loadBtn)  loadBtn.addEventListener('click', async () => { try {
  if (trainBtn) trainBtn.disabled = true; if (testBtn) testBtn.disabled = true;
  await loadData(); setStatus('Dataset ready. You can Train now.');
} catch (e) { setStatus('Load error: ' + e.message); console.error(e); }});

if (trainBtn) trainBtn.addEventListener('click', async () => { try {
  if (!ratings.length) await loadData();
  if (testBtn) testBtn.disabled = true;
  await train(); setStatus('Training finished. You can Test now.');
} catch (e) { setStatus('Train error: ' + e.message); console.error(e); }});

if (testBtn)  testBtn.addEventListener('click', async () => { try {
  await test();
} catch (e) { setStatus('Test error: ' + e.message); console.error(e); }});

// Optional: auto-load on page open
document.addEventListener('DOMContentLoaded', async () => {
  // comment the next two lines if you prefer manual
  // try { await loadData(); setStatus('Dataset ready.'); if (trainBtn) trainBtn.disabled = false; } catch(e){}
});
