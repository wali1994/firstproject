// Feature engineering + simple scoring (no external libs)
// Intuition:
// - Lower Recency (more recent) → better
// - Higher Frequency → better
// - Higher Monetary (total volume) → better
// - Longer Time (tenure) can be mildly positive

function computeScores(data, weights) {
  // Extract numeric arrays (coerce to number safely)
  const R = data.map(d => +d.Recency);
  const F = data.map(d => +d.Frequency);
  const M = data.map(d => +d.Monetary);
  const T = data.map(d => +d.Time);

  // Normalize via min-max to [0,1]
  const nR = normalizeInvert(R); // invert so lower Recency → higher
  const nF = normalize(F);
  const nM = normalize(M);
  const nT = normalize(T);

  // Weighted linear + sigmoid to get [0,1]
  const { wR, wF, wM, wT } = weights;
  const scored = data.map((row, i) => {
    const z = (wR * nR[i]) + (wF * nF[i]) + (wM * nM[i]) + (wT * nT[i]);
    const score = sigmoid(z * 3 - 1.5); // stretch & center — tweakable
    return { ...row, _score: +score.toFixed(4) };
  });

  // Sort descending by score
  scored.sort((a, b) => b._score - a._score);
  return scored;
}

function normalize(arr) {
  const {min, range} = minMax(arr);
  return arr.map(v => (v - min) / range);
}
function normalizeInvert(arr) {
  const n = normalize(arr);
  return n.map(x => 1 - x);
}
