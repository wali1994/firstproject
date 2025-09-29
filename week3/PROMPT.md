# Prompt (for Homework Submission)

**Task**: Build a Matrix Factorization movie recommender with TensorFlow.js that runs entirely in the browser. Use MovieLens 100K files `u.item` and `u.data`. Implement:
- Embedding Layers for users and movies
- Latent Vectors (flatten embeddings)
- Prediction via dot product (+ optional biases)
- UI to select a user and a movie, and display a predicted rating

**Requirements**
1. Parse `u.item` and `u.data` client‑side (no server).
2. Create TF.js model: user/movie embeddings → dot product (+ biases) → predicted rating.
3. Train with MSE loss on observed ratings; use Adam optimizer.
4. After training, allow selecting a user+movie to predict rating.
5. Keep the app simple: one HTML page with linked JS/CSS.

**Deliverables**
- `index.html`, `style.css`, `data.js`, `script.js`
- Do *not* edit README placeholders.

**Important code (with comments) must explain:**
- What an embedding layer does and why we use it
- How latent vectors are formed (flatten)
- How dot product + biases forms the final prediction
- Why we use MSE for explicit ratings
