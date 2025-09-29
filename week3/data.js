// data.js — load and parse MovieLens 100K (u.item, u.data)
// Exposes globals consumed by script.js
let movies = [];     // [{id, title, year}]
let ratings = [];    // [{userId, movieId, rating}]
let numUsers = 0;    // unique user count / max id
let numMovies = 0;   // unique movie count / max id

// Load raw files. Place u.item and u.data in the same folder as index.html
async function loadData() {
  const movieRes = await fetch('u.item');
  const ratingRes = await fetch('u.data');
  if (!movieRes.ok) throw new Error(`Failed to fetch u.item (${movieRes.status})`);
  if (!ratingRes.ok) throw new Error(`Failed to fetch u.data (${ratingRes.status})`);

  const movieText = await movieRes.text();
  const ratingText = await ratingRes.text();

  movies = parseItemData(movieText);
  ratings = parseRatingData(ratingText);

  // Derive contiguous maxima for embedding inputDim (ids usually start at 1 in MovieLens 100K)
  const userIds = new Set(ratings.map(r => r.userId));
  const movieIds = new Set([...ratings.map(r => r.movieId), ...movies.map(m => m.id)]);
  numUsers = Math.max(...userIds);
  numMovies = Math.max(...movieIds);
}

// Parse u.item lines: movieId|movieTitle|releaseDate|...|genre flags
function parseItemData(text) {
  const lines = text.split('\n').filter(Boolean);
  const out = [];
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 2) continue;
    const id = parseInt(parts[0], 10);
    const title = parts[1];
    // Try extracting year from title e.g., "Toy Story (1995)"
    let year = null;
    const m = title.match(/\((\d{4})\)/);
    if (m) year = parseInt(m[1], 10);
    out.push({ id, title, year });
  }
  return out;
}

// Parse u.data lines: userId\tmovieId\trating\ttimestamp
function parseRatingData(text) {
  const lines = text.split('\n').filter(Boolean);
  const out = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/); // split on whitespace
    if (parts.length < 3) continue;
    const userId = parseInt(parts[0], 10);
    const movieId = parseInt(parts[1], 10);
    const rating = parseFloat(parts[2]);
    out.push({ userId, movieId, rating });
  }
  return out;
}
