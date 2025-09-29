// script.js — Matrix Factorization with Embeddings in TensorFlow.js
let model;
let isTraining = false;

window.onload = async function () {
  try {
    updateStatus('Loading MovieLens data...');
    await loadData();
    populateUserDropdown();
    populateMovieDropdown();
    updateStatus('Data loaded. Training model...');
    await trainModel();
  } catch (err) {
    console.error('Initialization error:', err);
    updateStatus('Error initializing: ' + err.message, true);
  }
};

function populateUserDropdown() {
  const userSelect = document.getElementById('user-select');
  userSelect.innerHTML = '';
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
// Model: Embeddings -> Dot Product (+ Biases) -> Predicted Rating
function createModel(numUsers, numMovies, latentDim = 10) {
  const userInput = tf.input({ shape: [1], name: 'userInput' });
  const movieInput = tf.input({ shape: [1], name: 'movieInput' });
  const userEmbedding = tf.layers.embedding({ inputDim: numUsers + 1, outputDim: latentDim, name: 'userEmbedding' }).apply(userInput);
  const movieEmbedding = tf.layers.embedding({ inputDim: numMovies + 1, outputDim: latentDim, name: 'movieEmbedding' }).apply(movieInput);
  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);
  const userBias = tf.layers.embedding({ inputDim: numUsers + 1, outputDim: 1, name: 'userBias' }).apply(userInput);
  const movieBias = tf.layers.embedding({ inputDim: numMovies + 1, outputDim: 1, name: 'movieBias' }).apply(movieInput);
  const userBiasFlat = tf.layers.flatten().apply(userBias);
  const movieBiasFlat = tf.layers.flatten().apply(movieBias);
  const dot = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
  const sum = tf.layers.add().apply([dot, userBiasFlat, movieBiasFlat]);
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(sum);
  return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}
async function trainModel() {
  try {
    isTraining = true;
    document.getElementById('predict-btn').disabled = true;
    model = createModel(numUsers, numMovies, 10);
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
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
        onEpochEnd: (epoch, logs) => { updateStatus(`Epoch ${epoch + 1}/10 — loss: ${logs.loss.toFixed(4)}`); }
      }
    });
    tf.dispose([userTensor, movieTensor, ratingTensor]);
    updateStatus('Model training completed! You can now predict.');
    document.getElementById('predict-btn').disabled = false;
    isTraining = false;
  } catch (err) {
    console.error('Training error:', err);
    updateStatus('Error training: ' + err.message, true);
    isTraining = false;
  }
}
async function predictRating() {
  if (isTraining) { updateResult('Model is still training. Please wait...', 'medium'); return; }
  const userId = parseInt(document.getElementById('user-select').value);
  const movieId = parseInt(document.getElementById('movie-select').value);
  if (!userId || !movieId) { updateResult('Please select both a user and a movie.', 'medium'); return; }
  try {
    const userTensor = tf.tensor2d([[userId]]);
    const movieTensor = tf.tensor2d([[movieId]]);
    const pred = model.predict([userTensor, movieTensor]);
    const val = await pred.data();
    const predicted = val[0];
    tf.dispose([userTensor, movieTensor, pred]);
    const movie = movies.find(m => m.id === movieId);
    const title = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${movieId}`;
    const clamped = Math.min(5, Math.max(1, predicted));
    let cls = 'medium'; if (clamped >= 4) cls = 'high'; else if (clamped <= 2) cls = 'low';
    updateResult(`Predicted rating for User ${userId} on "${title}": <strong>${clamped.toFixed(2)}/5</strong>`, cls);
  } catch (err) {
    console.error('Prediction error:', err);
    updateResult('Error making prediction: ' + err.message, 'low');
  }
}
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
