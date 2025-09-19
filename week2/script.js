// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// --- Cosine Similarity helpers ---
function genreVectorFor(movie) {
    return genreNames.map(genre => (movie.genres.includes(genre) ? 1 : 0));
}

function cosineSimilarity(vecA, vecB) {
    let dot = 0, magA = 0, magB = 0;
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

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        resultElement.textContent = "Calculating recommendations (cosine)...";
        resultElement.className = 'loading';
        
        setTimeout(() => {
            try {
                const likedVector = genreVectorFor(likedMovie);
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateVector = genreVectorFor(candidate);
                    const score = cosineSimilarity(likedVector, candidateVector);
                    return { ...candidate, score };
                });
                
                scoredMovies.sort((a, b) => b.score - a.score);
                const topRecommendations = scoredMovies.slice(0, 2);
                
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend (cosine): ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
