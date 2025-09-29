// script.js — Matrix Factorization with Embeddings in TensorFlow.js
// -----------------------------------------------------------------------------
// This file implements the homework requirements WITHOUT changing the README.
// Key ideas:
// 1) Embedding Layers: learn user/movie latent vectors from IDs
// 2) Latent Vectors: flatten embeddings to dense vectors
// 3) Prediction: dot product of user/movie vectors + bias terms → rating
// -----------------------------------------------------------------------------

let model;
let isTraining = false;

window.onload = async function () {
  try {
    updateStatus('Loading MovieLens data...');
    await loadData();               // reads u.item + u.data → globals: movies, ratings, numUsers, numMovies
    populateUserDropdown();
    populateMovieDropdown();
    updateStatus('Data loaded. Training model...');
    await trainModel();             // trains MF model in-browser
  } catch (err) {
    console.error('Initialization error:', err);
    updateStatus('Error initializing: ' + err.message, true);
  }
};

function populateUserDropdown() {
  const userSelect = document.getElementById('user-select');
  userSelect.innerHTML = '';
  // MovieLens user IDs are typically contiguous starting at 1
  for (let i = 1; i <= numUsers; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `User ${i}`;
    userSelect.appendChild(opt);
  }
}

function populateMovieDropdown() {
  const movieSelect = document.getElementById('movie-select');
  movieSelect.innerHTML = '';
  movies.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    movieSelect.appendChild(opt);
  });
}

// ============================== MODEL ========================================
// Embeddings → Latent vectors → Dot product (+ biases) → Predicted rating
// -----------------------------------------------------------------------------
function createModel(numUsers, numMovies, latentDim = 10) {
  // Inputs are integer IDs; each sample is shape [1]
  const userInput = tf.input({ shape: [1], name: 'userInput' });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });

  // --- Embedding Layers ---
  // Learnable lookup tables mapping IDs → dense vectors of length latentDim
  const userEmbedding = tf.layers.embedding({
    inputDim: numUsers + 1, // +1 to be safe if IDs start at 1
    outputDim: latentDim,
    name: 'userEmbedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: latentDim,
    name: 'movieEmbedding'
  }).apply(movieInput);

  // --- Latent Vectors ---
  // Flatten embedding outputs (shape [1, latentDim]) → [latentDim]
  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  // --- Bias terms (optional but helpful) ---
  // Per-user and per-movie scalar biases capture global rating tendencies
  const userBias = tf.layers.embedding({
    inputDim: numUsers + 1,
    outputDim: 1,
    name: 'userBias'
  }).apply(userInput);
  const movieBias = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: 1,
    name: 'movieBias'
  }).apply(movieInput);
  const userBiasFlat = tf.layers.flatten().apply(userBias);
  const movieBiasFlat = tf.layers.flatten().apply(movieBias);

  // --- Prediction ---
  // Dot product between user & movie latent vectors gives interaction strength
  const dot = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]); // shape [1]
  // Add biases: rating ≈ u·v + b_user + b_movie
  const sum = tf.layers.add().apply([dot, userBiasFlat, movieBiasFlat]);
  // Ensure consistent shape [1] per example
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(sum);

  // Assemble model
  return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}

// Train end-to-end on the rating triples (userId, movieId, rating)
async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;

    model = createModel(numUsers, numMovies, 10);
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError'   // explicit rating regression (1–5 scale)
    });

    // Prepare tensors from rating tuples
    const userIds = ratings.map(r => r.userId);
    const movieIds = ratings.map(r => r.movieId);
    const ratingVals = ratings.map(r => r.rating);

    const userTensor = tf.tensor2d(userIds, [userIds.length, 1]);
    const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1]);
    const ratingTensor = tf.tensor2d(ratingVals, [ratingVals.length, 1]);

    updateStatus('Training model (please wait)...');

    await model.fit([userTensor, movieTensor], ratingTensor, {
      epochs: 10,
      batchSize: 64,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          updateStatus(`Epoch ${epoch + 1}/10 — loss: ${logs.loss.toFixed(4)}`);
        }
      }
    });

    tf.dispose([userTensor, movieTensor, ratingTensor]);

    updateStatus('Model training completed successfully!');
    document.getElementById('predict-btn').disabled = false;
    isTraining = false;
  } catch (err) {
    console.error('Training error:', err);
    updateStatus('Error training model: ' + err.message, true);
    isTraining = false;
  }
}

// Predict a rating for the selected user & movie
async function predictRating() {
  if (isTraining) {
    updateResult('Model is still training. Please wait...', 'medium');
    return;
  }
  const userId = parseInt(document.getElementById('user-select').value);
  const movieId = parseInt(document.getElementById('movie-select').value);
  if (!userId || !movieId) {
    updateResult('Please select both a user and a movie.', 'medium');
    return;
  }
  try {
    const userTensor  = tf.tensor2d([[userId]]);
    const movieTensor = tf.tensor2d([[movieId]]);
    const pred        = model.predict([userTensor, movieTensor]);
    const val         = await pred.data(); // flat Float32Array
    const predicted   = val[0];

    tf.dispose([userTensor, movieTensor, pred]);

    const movie = movies.find(m => m.id === movieId);
    const title = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;

    // Clamp to [1,5] since MovieLens ratings are 1–5
    const clamped = Math.min(5, Math.max(1, predicted));

    let cls = 'medium';
    if (clamped >= 4) cls = 'high';
    else if (clamped <= 2) cls = 'low';

    updateResult(
      `Predicted rating for User ${userId} on "${title}": <strong>${clamped.toFixed(2)}/5</strong>`,
      cls
    );
  } catch (err) {
    console.error('Prediction error:', err);
    updateResult('Error making prediction: ' + err.message, 'low');
  }
}

// ============================== UI HELPERS ===================================
function updateStatus(message, isError = false) {
  const el = document.getElementById('status');
  el.textContent = message;
  el.style.borderLeftColor = isError ? '#e74c3c' : '#2a7de1';
  el.style.background = isError ? '#fdedec' : '#f0f5ff';
}

function updateResult(message, className = '') {
  const el = document.getElementById('result');
  el.innerHTML = message;
  el.className = `result ${className}`;
}
