# Movie Recommendation System – GitHub Pages Demo


This repository contains a **content-based movie recommender system** using JavaScript and HTML/CSS, designed to run directly in the browser (no backend needed). It is also ready to be deployed with **GitHub Pages**.


---


## Features
- Loads the **MovieLens dataset** (`u.item` and `u.data`).
- Parses movie information and genres dynamically.
- Provides an interactive dropdown for selecting a movie.
- Computes **Cosine Similarity** on genre vectors to recommend similar movies.
- Displays the top recommended movies with a clean UI.


---


## File Structure
- **index.html** → Main web page UI.
- **style.css** → Styling and layout.
- **data.js** → Loads and parses the MovieLens dataset files.
- **script.js** → Core logic for recommendations (uses **Cosine Similarity**).
- **u.item / u.data** → Movie and rating datasets from MovieLens.


---


## How It Works
1. User selects a movie from the dropdown.
2. The system creates a **binary vector** for its genres.
3. For each other movie, it builds another genre vector.
4. **Cosine similarity** is calculated:
```
similarity = (A · B) / (||A|| * ||B||)
```
5. Top 2 most similar movies are displayed as recommendations.


---


## How to Run
1. Clone the repo:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
2. Open `index.html` in your browser.
3. Or deploy with **GitHub Pages**:
- Go to **Settings → Pages**
- Set source to `main` branch, `/ (root)`
- Save and get your live site URL.


---


## Example Extension Ideas
- Add a toggle between **Jaccard** and **Cosine** similarity.
- Include ratings (from `u.data`) to weight recommendations.
- Visualize similarities with charts.
- Extend UI for top-N recommendations (not just top 2).


---


## License
This project is for **educational/demo purposes**.
