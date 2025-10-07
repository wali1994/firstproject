// app.js
/* In-browser Two-Tower Retrieval on MovieLens 100K (TF.js)
   - Loads /data/u.data and /data/u.item
   - Trains a TwoTowerModel (two-tower.js) with in-batch softmax or BPR loss
   - Renders live loss, PCA projection of item embeddings, and a test table
*/

(() => {
  // ---------- State ----------
  const state = {
    interactions: [],      // { userId, itemId, rating, ts }
    items: new Map(),      // itemId -> { title, year }
    userToItems: new Map(),// userId -> [{itemId, rating, ts}]
    userIds: [],           // distinct userIds (string)
    itemIds: [],           // distinct itemIds (string)
    userId2idx: new Map(), // raw userId -> 0-based
    itemId2idx: new Map(), // raw itemId -> 0-based
    idx2userId: [],        // 0-based -> raw userId
    idx2itemId: [],        // 0-based -> raw itemId
    model: null,           // TwoTowerModel
    itemEmbMatrix: null,   // tf.Variable or tensor of item embeddings
    trained: false,
  };

  // ---------- DOM ----------
  const el = {
    btnLoad: document.getElementById('btnLoad'),
    btnTrain: document.getElementById('btnTrain'),
    btnTest: document.getElementById('btnTest'),
    status: document.getElementById('status'),
    dataBadge: document.getElementById('dataBadge'),
    modelBadge: document.getElementById('modelBadge'),
    lossCanvas: document.getElementById('lossCanvas'),
    embedCanvas: document.getElementById('embedCanvas'),
    tableArea: document.getElementById('tableArea'),
    // controls
    maxInteractions: document.getElementById('maxInteractions'),
    embDim: document.getElementById('embDim'),
    batchSize: document.getElementById('batchSize'),
    epochs: document.getElementById('epochs'),
    lr: document.getElementById('lr'),
    lossKind: document.getElementById('lossKind'),
  };

  const setBadge = (node, text, kind) => {
    node.textContent = text;
    node.className = `badge ${kind || ''}`;
  };

  const log = (s) => {
    console.log(s);
    el.status.textContent = s + '\n' + el.status.textContent;
  };

  // ---------- Utils ----------
  function parseUItemLine(line) {
    // u.item: item_id|title|release_date|...
    const parts = line.split('|');
    const itemId = parts[0];
    let title = parts[1] || '';
    let year = null;
    // Try to parse year from title like "Toy Story (1995)"
    const m = title.match(/\((\d{4})\)\s*$/);
    if (m) {
      year = parseInt(m[1], 10);
      title = title.replace(/\(\d{4}\)\s*$/, '').trim();
    }
    return { itemId, title, year };
  }

  function parseUDataLine(line) {
    // u.data: user_id \t item_id \t rating \t timestamp
    const [u, i, r, ts] = line.split('\t');
    return {
      userId: u, itemId: i,
      rating: parseInt(r, 10),
      ts: parseInt(ts, 10),
    };
  }

  function buildIndexers() {
    const userSet = new Set();
    const itemSet = new Set();
    for (const it of state.interactions) {
      userSet.add(it.userId);
      itemSet.add(it.itemId);
    }
    state.userIds = Array.from(userSet).sort((a,b)=>Number(a)-Number(b));
    state.itemIds = Array.from(itemSet).sort((a,b)=>Number(a)-Number(b));
    state.userId2idx.clear(); state.itemId2idx.clear();
    state.idx2userId = []; state.idx2itemId = [];
    state.userIds.forEach((uid, idx) => {
      state.userId2idx.set(uid, idx);
      state.idx2userId[idx] = uid;
    });
    state.itemIds.forEach((iid, idx) => {
      state.itemId2idx.set(iid, idx);
      state.idx2itemId[idx] = iid;
    });
  }

  function buildUserMaps() {
    state.userToItems.clear();
    for (const it of state.interactions) {
      if (!state.userToItems.has(it.userId)) state.userToItems.set(it.userId, []);
      state.userToItems.get(it.userId).push({ itemId: it.itemId, rating: it.rating, ts: it.ts });
    }
    // sort per user for quick "top rated then recency"
    for (const [u, arr] of state.userToItems) {
      arr.sort((a, b) => {
        if (b.rating !== a.rating) return b.rating - a.rating;
        return b.ts - a.ts;
      });
    }
  }

  function sampleQualifiedUser(minCount = 20) {
    const candidates = [];
    for (const [u, arr] of state.userToItems) {
      if (arr.length >= minCount) candidates.push(u);
    }
    if (!candidates.length) return null;
    return candidates[Math.floor(Math.random() * candidates.length)];
  }

  function topKFromScores(scores, K, excludeSet) {
    // scores: Float32Array or array; return array of indices (desc)
    const pairs = [];
    const n = scores.length;
    for (let i = 0; i < n; i++) {
      if (excludeSet && excludeSet.has(i)) continue;
      pairs.push([scores[i], i]);
    }
    pairs.sort((a, b) => b[0] - a[0]);
    return pairs.slice(0, K).map(p => p[1]);
  }

  // ---------- Simple Canvas Line Chart ----------
  class MiniChart {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.values = [];
    }
    push(v) {
      this.values.push(v);
      if (this.values.length > 1000) this.values.shift();
      this.draw();
    }
    reset() { this.values = []; this.draw(); }
    draw() {
      const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
      ctx.clearRect(0,0,W,H);
      // bg grid
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let y = 0; y <= 4; y++) {
        const yy = (H/4)*y + 0.5; ctx.moveTo(0, yy); ctx.lineTo(W, yy);
      }
      ctx.stroke();
      if (!this.values.length) return;
      const min = Math.min(...this.values);
      const max = Math.max(...this.values);
      const pad = 8;
      ctx.beginPath();
      for (let i = 0; i < this.values.length; i++) {
        const x = pad + (W - 2*pad) * (i / Math.max(1,(this.values.length - 1)));
        const y = H - pad - (H - 2*pad) * ((this.values[i] - min) / Math.max(1e-9,(max - min)));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      // pretty gradient stroke
      const grad = ctx.createLinearGradient(0,0,W,0);
      grad.addColorStop(0, '#22d3ee'); grad.addColorStop(1, '#8b5cf6');
      ctx.strokeStyle = grad; ctx.lineWidth = 2; ctx.stroke();
      // labels
      ctx.fillStyle = '#cbd5e1'; ctx.font = '12px ui-monospace, monospace';
      ctx.fillText(`min ${min.toFixed(4)}  max ${max.toFixed(4)}  last ${this.values[this.values.length-1].toFixed(4)}`, 10, 18);
    }
  }

  const lossChart = new MiniChart(el.lossCanvas);

  // ---------- PCA (Power Iteration) ----------
  // Returns { xs: Float32Array, ys: Float32Array, idxs: number[] } for sampled item indices
  async function projectItemsPCA(getItemEmbFn, sampleCount = 1000) {
    // Build a matrix X [m x d]: m sampled items, d emb dim
    const numItems = state.itemIds.length;
    const sel = [];
    for (let i = 0; i < numItems; i++) sel.push(i);
    // shuffle & slice
    for (let i = sel.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [sel[i], sel[j]] = [sel[j], sel[i]];
    }
    const idxs = sel.slice(0, Math.min(sampleCount, numItems));
    // gather embeddings
    const d = Number(el.embDim.value);
    const X = tf.tidy(() => {
      const idxT = tf.tensor1d(idxs, 'int32');
      return getItemEmbFn(idxT).clone(); // [m,d]
    });

    // center
    let Xc = tf.tidy(() => {
      const mean = tf.mean(X, 0); // [d]
      return tf.sub(X, mean);
    });

    // power iteration for first PC
    const m = idxs.length;
    let v1 = tf.randomNormal([d], 0, 1);
    for (let it = 0; it < 20; it++) {
      const t = tf.matMul(Xc, v1.reshape([d, 1]));   // [m,1] = Xc * v1
      const t2 = tf.matMul(Xc.transpose(), t);       // [d,1] = Xc^T * t
      v1.dispose(); v1 = tf.div(t2.reshape([d]), tf.norm(t2));
    }
    // deflate
    const proj1 = tf.matMul(Xc, v1.reshape([d,1]));  // [m,1]
    const v1v1T = tf.outerProduct(v1, v1);           // [d,d]
    const Xc_def = tf.tidy(() => Xc.sub(tf.matMul(proj1, v1.reshape([1,d])))); // Xc - (Xc v1) v1^T

    // second PC
    let v2 = tf.randomNormal([d], 0, 1);
    for (let it = 0; it < 20; it++) {
      const t = tf.matMul(Xc_def, v2.reshape([d, 1]));
      const t2 = tf.matMul(Xc_def.transpose(), t);
      v2.dispose(); v2 = tf.div(t2.reshape([d]), tf.norm(t2));
    }

    const proj2 = tf.matMul(Xc, v2.reshape([d,1])); // [m,1]
    const xs = (await proj1.data()).slice(); // Float32Array
    const ys = (await proj2.data()).slice();

    X.dispose(); Xc.dispose(); proj1.dispose(); proj2.dispose(); v1.dispose(); v2.dispose(); Xc_def.dispose(); v1v1T.dispose();
    return { xs, ys, idxs };
  }

  function drawEmbeddingScatter(xs, ys, idxs) {
    const ctx = el.embedCanvas.getContext('2d');
    const W = el.embedCanvas.width, H = el.embedCanvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);

    if (!xs.length) return;

    // scale
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 16;
    const scaleX = x => pad + (W - 2*pad) * ((x - minX) / Math.max(1e-9,(maxX - minX)));
    const scaleY = y => H - pad - (H - 2*pad) * ((y - minY) / Math.max(1e-9,(maxY - minY)));

    // store points for simple hover
    const points = [];
    ctx.fillStyle = '#94a3b8';
    for (let i = 0; i < xs.length; i++) {
      const x = scaleX(xs[i]), y = scaleY(ys[i]);
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI*2);
      ctx.fill();
      points.push({ x, y, idx: idxs[i] });
    }

    // hover behavior (titles)
    const onMove = (e) => {
      const rect = el.embedCanvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left) * (el.embedCanvas.width / rect.width);
      const my = (e.clientY - rect.top) * (el.embedCanvas.height / rect.height);

      // find nearest within radius
      let best = null; let bestD2 = 12*12;
      for (const p of points) {
        const dx = p.x - mx, dy = p.y - my;
        const d2 = dx*dx + dy*dy;
        if (d2 < bestD2) { best = p; bestD2 = d2; }
      }
      // redraw
      ctx.clearRect(0,0,W,H);
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
      ctx.fillStyle = '#94a3b8';
      for (const p of points) {
        ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI*2); ctx.fill();
      }
      if (best) {
        // highlight
        ctx.beginPath(); ctx.arc(best.x, best.y, 4, 0, Math.PI*2);
        const grad = ctx.createLinearGradient(0,0,W,0);
        grad.addColorStop(0, '#22d3ee'); grad.addColorStop(1, '#8b5cf6');
        ctx.fillStyle = grad; ctx.fill();
        // label
        const itemId = state.idx2itemId[best.idx];
        const meta = state.items.get(itemId);
        const title = meta ? (meta.title + (meta.year ? ` (${meta.year})` : '')) : itemId;
        const label = title.length > 60 ? title.slice(0,57)+'…' : title;
        ctx.font = '12px ui-sans-serif, system-ui';
        const tw = ctx.measureText(label).width + 10;
        const lx = Math.min(Math.max(best.x + 8, 4), W - tw - 4);
        const ly = Math.max(best.y - 8, 16);
        ctx.fillStyle = 'rgba(17,24,39,0.9)';
        ctx.fillRect(lx, ly - 14, tw, 18);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.strokeRect(lx, ly - 14, tw, 18);
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(label, lx + 5, ly);
      }
    };
    el.embedCanvas.onmousemove = onMove;
  }

  // ---------- Data Loading ----------
  async function loadData() {
    setBadge(el.dataBadge, 'Data: loading…', 'warn');
    el.btnLoad.disabled = true;

    const respItems = await fetch('./data/u.item');
    const textItems = await respItems.text();
    const linesItems = textItems.split('\n').filter(Boolean);
    for (const line of linesItems) {
      const { itemId, title, year } = parseUItemLine(line);
      if (!itemId) continue;
      state.items.set(itemId, { title, year });
    }

    const respData = await fetch('./data/u.data');
    const textData = await respData.text();
    const linesData = textData.split('\n').filter(Boolean);
    const maxN = parseInt(el.maxInteractions.value, 10) || 80000;
    state.interactions = [];
    for (let i = 0; i < Math.min(maxN, linesData.length); i++) {
      state.interactions.push(parseUDataLine(linesData[i]));
    }

    buildIndexers();
    buildUserMaps();

    setBadge(el.dataBadge, `Data: ${state.interactions.length} interactions, ${state.userIds.length} users, ${state.itemIds.length} items`, 'ok');
    log('Loaded data.\n- Interactions: ' + state.interactions.length + '\n- Users: ' + state.userIds.length + '\n- Items: ' + state.itemIds.length);
    el.btnTrain.disabled = false;
  }

  // ---------- Training ----------
  async function train() {
    if (!state.interactions.length) return;

    el.btnTrain.disabled = true; el.btnTest.disabled = true;
    setBadge(el.modelBadge, 'Model: initializing…', 'warn');
    lossChart.reset(); el.tableArea.innerHTML = '';

    const embDim = parseInt(el.embDim.value, 10) || 32;
    const batchSize = parseInt(el.batchSize.value, 10) || 512;
    const epochs = parseInt(el.epochs.value, 10) || 5;
    const lr = parseFloat(el.lr.value) || 0.01;
    const lossKind = el.lossKind.value;

    // Build training pairs (uIdx, iIdx)
    const pairs = [];
    for (const it of state.interactions) {
      const uIdx = state.userId2idx.get(it.userId);
      const iIdx = state.itemId2idx.get(it.itemId);
      if (uIdx == null || iIdx == null) continue;
      pairs.push([uIdx, iIdx]);
    }
    // shuffle
    for (let i = pairs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
    }

    // Initialize model
    state.model = new TwoTowerModel(state.userIds.length, state.itemIds.length, embDim, lossKind);
    state.model.setOptimizer(tf.train.adam(lr));
    setBadge(el.modelBadge, `Model: training (${lossKind})…`, 'warn');

    // Batching helpers
    function* batchIter(arr, B) {
      for (let i = 0; i < arr.length; i += B) {
        yield arr.slice(i, i + B);
      }
    }

    // Train loop
    let step = 0;
    for (let ep = 0; ep < epochs; ep++) {
      let epLossSum = 0, epCount = 0;
      for (const batch of batchIter(pairs, batchSize)) {
        const uIdxs = batch.map(p => p[0]);
        const iIdxs = batch.map(p => p[1]);
        const uT = tf.tensor1d(uIdxs, 'int32');
        const iT = tf.tensor1d(iIdxs, 'int32');
        const loss = await state.model.trainStep(uT, iT);
        uT.dispose(); iT.dispose();
        epLossSum += loss; epCount++;
        step++;
        lossChart.push(loss);
        if (step % 10 === 0) {
          await tf.nextFrame(); // keep UI responsive
        }
      }
      log(`Epoch ${ep+1}/${epochs} — avg loss: ${(epLossSum/Math.max(1,epCount)).toFixed(5)}`);
    }

    // Cache item emb matrix for fast scoring
    state.itemEmbMatrix = state.model.getItemEmbeddingTable(); // tf.Tensor [numItems, embDim]
    state.trained = true;
    setBadge(el.modelBadge, 'Model: trained', 'ok');

    // Projection
    setBadge(el.modelBadge, 'Projecting embeddings…', 'warn');
    const { xs, ys, idxs } = await projectItemsPCA((idxT) => state.model.itemForward(idxT), 1000);
    drawEmbeddingScatter(xs, ys, idxs);
    setBadge(el.modelBadge, 'Model: ready', 'ok');
    el.btnTest.disabled = false;
  }

  // ---------- Testing (Top-Rated vs Top-Recommended) ----------
  async function testOnce() {
    if (!state.trained || !state.itemEmbMatrix) {
      log('Please train first.');
      return;
    }
    el.btnTest.disabled = true;

    const userId = sampleQualifiedUser(20);
    if (!userId) {
      log('No user with ≥20 ratings found in the loaded slice.');
      el.btnTest.disabled = false;
      return;
    }
    const rated = state.userToItems.get(userId);
    const topRated = rated.slice(0, 10)
      .map(x => {
        const meta = state.items.get(x.itemId);
        return meta ? (meta.title + (meta.year ? ` (${meta.year})` : '')) : x.itemId;
      });

    // Scores for all items
    const uIdx = state.userId2idx.get(userId);
    const uT = tf.tensor1d([uIdx], 'int32');
    const uEmb = state.model.getUserEmbedding(uT).squeeze(); // [embDim]
    uT.dispose();

    // score = uEmb dot itemEmbMatrix
    let scoresT = tf.matMul(state.itemEmbMatrix, uEmb.reshape([uEmb.shape[0], 1])); // [numItems,1] = Item(Num x D) * u(D x 1)
    scoresT = scoresT.reshape([state.itemIds.length]);

    const scores = await scoresT.data();
    scoresT.dispose(); uEmb.dispose();

    const excludeSet = new Set(rated.map(x => state.itemId2idx.get(x.itemId)));
    const topIdx = topKFromScores(scores, 10, excludeSet);
    const recs = topIdx.map(idx => {
      const iid = state.idx2itemId[idx];
      const meta = state.items.get(iid);
      return meta ? (meta.title + (meta.year ? ` (${meta.year})` : '')) : iid;
    });

    renderSideBySide(topRated, recs, userId);
    el.btnTest.disabled = false;
  }

  function renderSideBySide(leftArr, rightArr, userId) {
    const safe = s => (s || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    const rows = [];
    const n = Math.max(leftArr.length, rightArr.length);
    for (let i = 0; i < n; i++) {
      rows.push(`<tr><td>${i+1}. ${leftArr[i]?safe(leftArr[i]):''}</td><td>${i+1}. ${rightArr[i]?safe(rightArr[i]):''}</td></tr>`);
    }
    el.tableArea.innerHTML = `
      <table class="two-col-table">
        <thead><tr><th>Top-10 Historically Rated (User ${safe(userId)})</th><th>Top-10 Recommended (Unseen)</th></tr></thead>
        <tbody>${rows.join('')}</tbody>
      </table>
    `;
  }

  // ---------- Wire up ----------
  el.btnLoad.addEventListener('click', loadData);
  el.btnTrain.addEventListener('click', train);
  el.btnTest.addEventListener('click', testOnce);

  // Initial badges
  setBadge(el.dataBadge, 'Data: not loaded', '');
  setBadge(el.modelBadge, 'Model: idle', '');
})();
