let LR = {
  beta0: 0,
  beta: [0,0,0,0], // Recency, Frequency, Monetary, Time (normalized)
  trained: false
};

const TARGET_CANDIDATES = ['Class','Donated','Target','Label','Made Donation in March 2007','March_2007'];

function findTargetHeader(header){
  const low = header.map(h => h.trim().toLowerCase());
  for (const cand of TARGET_CANDIDATES){
    const idx = low.indexOf(cand.trim().toLowerCase());
    if (idx !== -1) return header[idx];
  }
  return null;
}

function prepareXY(rows, header) {
  const yName = findTargetHeader(header);
  if (!yName) return { X:null, y:null, yName:null, note:'No target column found.' };

  const Xraw = rows.map(r => [ +r.Recency, +r.Frequency, +r.Monetary, +r.Time ]);
  const y = rows.map(r => +r[yName]);

  // Normalize columns 0..3 (min-max)
  const cols = [[],[],[],[]];
  Xraw.forEach(row => row.forEach((v,i)=> cols[i].push(v)));
  const mm = cols.map(c => minMax(c));
  const X = Xraw.map(row => row.map((v,i)=> (v - mm[i].min) / (mm[i].range || 1)));

  return { X, y, yName, note: `Target detected: "${yName}" (0/1).` };
}

function trainLogistic(X, y, opts={lr:0.05, epochs:200}) {
  const n = X.length, d = X[0].length;
  let beta0 = 0;
  let beta = new Array(d).fill(0);

  const lr = opts.lr, epochs = opts.epochs;

  for (let ep=0; ep<epochs; ep++){
    let g0 = 0;
    const g = new Array(d).fill(0);
    for (let i=0;i<n;i++){
      const z = beta0 + dot(beta, X[i]);
      const p = 1/(1+Math.exp(-z));
      const err = p - y[i];
      g0 += err;
      for (let j=0;j<d;j++){
        g[j] += err * X[i][j];
      }
    }
    beta0 -= lr * (g0 / n);
    for (let j=0;j<d;j++) beta[j] -= lr * (g[j] / n);
  }
  return {beta0, beta};
}

function applyLogistic(rows, model) {
  // recompute normalized X (same transform implicitly — we use fresh min-max)
  const Xraw = rows.map(r => [ +r.Recency, +r.Frequency, +r.Monetary, +r.Time ]);
  const cols = [[],[],[],[]];
  Xraw.forEach(row => row.forEach((v,i)=> cols[i].push(v)));
  const mm = cols.map(c => minMax(c));
  const X = Xraw.map(row => row.map((v,i)=> (v - mm[i].min) / (mm[i].range || 1)));

  return X.map(x => 1/(1+Math.exp(-(model.beta0 + dot(model.beta, x)))));
}

function dot(a,b){ let s=0; for (let i=0;i<a.length;i++) s += a[i]*b[i]; return s; }
