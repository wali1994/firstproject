# Matrix Factorization Movie Recommender (Homework Build)

This project trains a collaborative filtering model **in the browser** using **TensorFlow.js** and the MovieLens 100K dataset (`u.item`, `u.data`).

> Per the homework instruction, the original README placeholders are unchanged. This repo includes detailed **inline comments** inside `script.js` that explain the **Embedding Layers → Latent Vectors → Prediction** pipeline.

## Files
- `index.html` — UI + script includes
- `style.css` — styling
- `data.js` — parses `u.item` and `u.data`
- `script.js` — TF.js model with embeddings, latent vectors, dot product (+ biases), training & prediction
- `PROMPT.md` — the exact prompt submitted for homework

## How to run
1. Put `index.html`, `style.css`, `data.js`, `script.js` in a folder **together with** `u.item` and `u.data`.
2. Open `index.html` in a modern browser (or host with GitHub Pages).
3. Wait for training to finish; then select a user & movie and click **Predict Rating**.

## Notes
- Embeddings use `inputDim = maxID + 1` to be safe when IDs start at 1.
- Loss: **MSE** is appropriate for explicit ratings (1–5).
- You can increase `epochs` or `latentDim` for better accuracy (slower).
