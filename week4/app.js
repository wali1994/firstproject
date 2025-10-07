/* app.js — Controller for Two-Tower Movie Recommender (shallow + deep)
 * Uses dataset at ./data/u.item and ./data/u.data
 */

const $ = (id) => document.getElementById(id);
const loadBtn = $('loadBtn');
const trainBtn = $('trainBtn');
const testBtn  = $('testBtn');
const lossCanvas = $('lossChart');
const embCanvas  = $('embeddingChart');
const resultsDiv = $('results');

function setStatus(msg) {
  const el = $('status-line');
  if (el) el.textContent = msg; else console.log('[STATUS]', msg);
}

/* -------- Data holders -------- */
let movies = [];        // [{rawId, idx, title, genres:[19]}]
let ratings = [];       // [{uRaw, iRaw, uIdx, iIdx, rating, ts}]
let numUsers = 0, numItems = 0;
const NUM_GENRES = 19;

let userIdToIdx = new Map();
let itemIdToIdx = new Map();
let idxToMovie  = [];
let userHist    = new Map();

let allItemGenres = null;     // Float32Array [numItems * 19]
let filteredUserIdx = [];
const MIN_USER_RATINGS = 20;

/* -------- Models -------- */
let shallow = null; // useDeep=false
let deep    = null; // useDeep=true

/* ========================= Load data ========================= */
async function loadData() {
  setStatus('Loading dataset from ./data …');
  trainBtn.disabled = true; testBtn.disabled = true;

  try {
    const itemRes = await fetch('./data/u.item');
    const dataRes = await fetch('./data/u.data');
    if (!itemRes.ok) throw new Error(`u.item fetch failed: ${itemRes.status}`);
    if (!dataRes.ok) throw new Error(`u.data fetch failed: ${dataRes.status}`);

    const itemText = await itemRes.text();
    const dataText = await dataRes.text();
    if (!itemText.trim()) throw new Error('u.item is empty');
    if (!dataText.trim()) throw new Error('u.data is empty');

    parseItems(itemText);
    parseRatings(dataText);
    buildUserHistories();

    filteredUserIdx = Array.from(userHist.keys())
      .filter(u => userHist.get(u).length >= MIN_USER_RATINGS);

    setStatus(
      `Loaded ${numItems} movies, ${numUsers} users, ${ratings.length} ratings. ` +
      `Users with ≥${MIN_USER_RATINGS}: ${filteredUserIdx.length}`
    );
    if (numItems && numUsers && ratings.length) trainBtn.disabled = false;
  } catch (e) {
    console.error(e);
    setStatus('Load error: ' + e.message);
  }
}

function parseItems(text) {
  movies = []; itemIdToIdx.clear(); idxToMovie = [];
  const lines = text.split('\n').filter(Boolean);
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 2 + NUM_GENRES) continue;
    const rawId = parseInt(parts[0], 10);
    const title = parts[1];
    const genreFlags = parts.slice(parts.length - NUM_GENRES).map(v => parseInt(v, 10) || 0);
    const idx = movies.length;
    const m = { rawId, idx, title, genres: genreFlags };
    movies.push(m);
    itemIdToIdx.set(rawId, idx);
    idxToMovie[idx] = m;
  }
  numItems = movies.length;

  allItemGenres = new Float32Array(numItems * NUM_GENRES);
  for (let i = 0; i < numItems; i++) {
    for (let g = 0; g < NUM_GENRES; g++) {
      allItemGenres[i * NUM_GENRES + g] = movies[i].genres[g];
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
    if (!itemIdToIdx.has(iRaw)) continue;
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
  ratings.forEach(({uIdx,iIdx,rating,ts}) => {
    if (!userHist.has(uIdx)) userHist.set(uIdx, []);
    userHist.get(uIdx).push({ iIdx, rating, ts });
  });
  for (const u of userHist.keys()) userHist.get(u).sort((a,b)=> b.ts - a.ts);
}

/* ========================= Train ========================= */
async function train() {
  // clear TF variables so retraining works cleanly
  if (tf.disposeVariables) tf.disposeVariables();
  else {
    for (const n in tf.engine().registeredVariables) tf.dispose(tf.engine().registeredVariables[n]);
    tf.engine().registeredVariables = {};
    tf.engine().state.registeredVariables = {};
  }

  if (!ratings.length) { setStatus('Load data first.'); return; }
  setStatus('Initializing models…');
  trainBtn.disabled = true; testBtn.disabled = true;

  shallow?.dispose?.(); deep?.dispose?.();

  shallow = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32, useDeep: false, lr: 1e-3
  });

  deep = new TwoTowerModel(numUsers, numItems, {
    embeddingDim: 32, useDeep: true,
    userHidden: [64, 32], itemHidden: [64, 32],
    numGenres: NUM_GENRES, userFeatDim: 0, lr: 1e-3
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
        const row = new Array(NUM_GENRES);
        for (let g=0; g<NUM_GENRES; g++) row[g] = movies[r.iIdx].genres[g];
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

  // Build indices for recommendation
  shallow.buildItemIndex();
  const allItemGenres2D = tf.tensor2d(allItemGenres, [numItems, NUM_GENRES]);
  deep.buildItemIndex(allItemGenres2D);
  allItemGenres2D.dispose();

  await drawEmbeddingPCA(deep.getItemVectors());

  setStatus('Training finished. Click Test to see recommendations.');
  testBtn.disabled = false;
  trainBtn.disabled = false;
}

/* ========================= Test / Results ========================= */
async function test() {
  if (!shallow || !deep) { setStatus('Train the models first.'); return; }
  const users = Array.from(userHist.keys()).filter(u => userHist.get(u).length >= MIN_USER_RATINGS);
  const uIdx = users.length ? users[Math.floor(Math.random()*users.length)] : 0;

  const topRated = topNUserRated(uIdx, 10);
  const ratedSet = new Set(userHist.get(uIdx)?.map(x => x.iIdx) || []);

  const { topIdx: tS } = await shallow.scoreAllForUser(uIdx);
  const { topIdx: tD } = await deep.scoreAllForUser(uIdx);

  const recShallow = tS.filter(i => !ratedSet.has(i)).slice(0, 10);
  const recDeep    = tD.filter(i => !ratedSet.has(i)).slice(0, 10);

  renderTables(uIdx, topRated, recShallow, recDeep);
}

function topNUserRated(uIdx, N=10) {
  const hist = userHist.get(uIdx) || [];
  const sorted = hist.slice().sort((a,b)=> (b.rating - a.rating) || (b.ts - a.ts));
  return sorted.slice(0, N).map(x => x.iIdx);
}

/* ========================= UI / Charts ========================= */
function renderTables(uIdx, topRatedIdx, topRecIdx, topDeepIdx) {
  resultsDiv.innerHTML = '';
  const title = document.createElement('h3');
  const rawUser = (() => {
    if (!renderTables._cache) {
      renderTables._cache = new Map();
      for (const [raw, idx] of userIdToIdx.entries()) renderTables._cache.set(idx, raw);
    }
    return renderTables._cache.get(uIdx) ?? uIdx;
  })();
  title.textContent = `Recommendations for User ${rawUser}`;
  resultsDiv.appendChild(title);

  const grid = document.createElement('div');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = '1fr 1fr 1fr';
  grid.style.gap = '12px';

  grid.appendChild(buildTable('Top 10 Rated Movies (Historical)', topRatedIdx));
  grid.appendChild(buildTable('Top 10 Recommended (Shallow)', topRecIdx));
  grid.appendChild(buildTable('Top 10 Recommended (Deep Learning)', topDeepIdx));

  resultsDiv.appendChild(grid);
}

function buildTable(title, itemIdxList) {
  const card = document.createElement('div');
  card.style.border = '1px solid #e2e8f0';
  card.style.borderRadius = '12px';
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

function clearCanvas(c){ if (!c) return; const x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height); x.fillStyle='#fff'; x.fillRect(0,0,c.width,c.height); }
function drawLoss(ctx, s1, s2) {
  const W=ctx.canvas.width,H=ctx.canvas.height; clearCanvas(ctx.canvas);
  function series(a,color){ if(a.length<2)return; const mx=Math.max(...a),mn=Math.min(...a); const px=10,py=10;
    ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=2;
    for(let i=0;i<a.length;i++){ const x=px+(W-2*px)*i/(a.length-1); const y=H-py-(H-2*py)*(a[i]-mn)/(mx-mn+1e-8); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y); }
    ctx.stroke();
  }
  series(s1,'#1f77b4'); series(s2,'#d62728');
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
  const pts = proj.arraySync(); // sync read avoids disposal timing
  proj.dispose(); mean.dispose(); Xc.dispose(); cov.dispose(); eigVecs.dispose(); W.dispose();

  const ctx = embCanvas.getContext('2d'); clearCanvas(embCanvas);
  const w=embCanvas.width,h=embCanvas.height,p=20;
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  ctx.fillStyle='#0a84ff33';
  for (const [x0,y0] of pts) {
    const x = p + (w-2*p)*(x0-minX)/(maxX-minX+1e-8);
    const y = h - p - (h-2*p)*(y0-minY)/(maxY-minY+1e-8);
    ctx.fillRect(x-1, y-1, 2, 2);
  }
  ctx.fillStyle='#333'; ctx.fillText('Item Embeddings (PCA)', 10, 14);
}

/* ========================= Buttons ========================= */
loadBtn?.addEventListener('click', async () => {
  try { await loadData(); } catch(e){ setStatus('Load error: '+e.message); console.error(e); }
});
trainBtn?.addEventListener('click', async () => {
  try { await train(); } catch(e){ setStatus('Train error: '+e.message); console.error(e); trainBtn.disabled=false; }
});
testBtn?.addEventListener('click', async () => {
  try { await test(); } catch(e){ setStatus('Test error: '+e.message); console.error(e); }
});

// Optional auto-load
// document.addEventListener('DOMContentLoaded', () => loadBtn?.click());
