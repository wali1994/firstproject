# Matrix Factorization Movie Recommender

This project implements a **Matrix Factorization based recommender system** using **TensorFlow.js**.  
The model is trained directly in the browser on the **MovieLens dataset** (`u.item`, `u.data`) and predicts user ratings for movies.

---

## Model Architecture

- **Embedding Layers:**
  - `tf.layers.embedding({ inputDim: numUsers + 1, outputDim: latentDim, name: 'userEmbedding' })`
  - `tf.layers.embedding({ inputDim: numMovies + 1, outputDim: latentDim, name: 'movieEmbedding' })`

- **Latent Vectors:**
  - `userVector = tf.layers.flatten().apply(userEmbedding)`
  - `movieVector = tf.layers.flatten().apply(movieEmbedding)`

- **Prediction:**
  - `dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector])`
  - `prediction = tf.layers.reshape({ targetShape: [1] }).apply(dotProduct)`

---

## Training

- Loss function: **Mean Squared Error (MSE)**  
- Optimizer: **Adam** (learning rate = 0.001)  
- Trained for **10 epochs** with **batch size 64**  
- Validation split: **10%** of the data

---

## Usage

1. Place `index.html`, `style.css`, `data.js`, `script.js`, `u.item`, and `u.data` in the same directory.  
2. Open `index.html` in a modern browser (or deploy via GitHub Pages).  
3. The model will load and start training automatically.  
4. After training, select a **user** and a **movie**, then click **Predict Rating**.  
5. The predicted rating (1–5 scale) will be displayed.

---

## Example Output

```
Epoch 5/10 — loss: 0.8123
Model training completed successfully!
Predicted rating for User 1 on "Toy Story (1995)": 4.12/5
```
