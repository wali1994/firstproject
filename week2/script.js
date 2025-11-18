// Initialize the application when the window loads
window.onload = async function () {
    const resultElement = document.getElementById('result');

    try {
        // Inform the user that data is loading
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load MovieLens data (u.item + u.data)
        await loadData();

        // Populate dropdown once data is ready
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // loadData already set a friendly error message on the page
    }
};

/**
 * Fill the dropdown with movie titles, sorted alphabetically.
 */
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');

    // Clear existing options except the placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    // Add movies as <option> elements
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;   // movieId from MovieLens
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

/* ------------------------------------------------------------------
   Cosine similarity on genre vectors
   ------------------------------------------------------------------
   We represent each movie as a binary vector over the 18 genres:
   example: [Action, Adventure, ..., Western]
   Then we compute cosine similarity between the liked movie and
   every other movie:

       cos(a, b) = (a · b) / (||a|| * ||b||)

   This is exactly what the homework asks for in slide 3.
-------------------------------------------------------------------*/

/**
 * Build a binary genre vector for a movie.
 * For each genre name, put 1 if the movie has that genre, else 0.
 */
function genreVectorFor(movie) {
    return genreNames.map(genre =>
        movie.genres.includes(genre) ? 1 : 0
    );
}

/**
 * Compute cosine similarity between two numeric vectors.
 */
function cosineSimilarity(vecA, vecB) {
    let dot = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < vecA.length; i++) {
        const a = vecA[i];
        const b = vecB[i];
        dot += a * b;
        magA += a * a;
        magB += b * b;
    }

    const denom = Math.sqrt(magA) * Math.sqrt(magB);
    return denom ? dot / denom : 0;
}

/**
 * Main recommendation function.
 * 1. Read the selected movie.
 * 2. Build its genre vector.
 * 3. Compute cosine similarity with every other movie.
 * 4. Show the top 2 recommended movies.
 */
function getRecommendations() {
    const resultElement = document.getElementById('result');

    try {
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value, 10);

        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }

        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }

        resultElement.textContent = "Calculating recommendations using cosine similarity...";
        resultElement.className = 'loading';

        // Small timeout so the loading text can paint before heavy work
        setTimeout(() => {
            try {
                const likedVector = genreVectorFor(likedMovie);

                // Consider all other movies as candidates
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

                // Compute cosine similarity score for each candidate
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateVector = genreVectorFor(candidate);
                    const score = cosineSimilarity(likedVector, candidateVector);
                    return { ...candidate, score };
                });

                // Sort by similarity score (highest first)
                scoredMovies.sort((a, b) => b.score - a.score);

                // Take top N recommendations (here N = 2, same as slide example)
                const topRecommendations = scoredMovies.slice(0, 2);

                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent =
                        `Because you liked "${likedMovie.title}", ` +
                        `we recommend (cosine similarity): ${recommendationTitles.join(', ')}.`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent =
                        `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent =
                    "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
