// Initialize the application when the window loads
window.onload = async function () {
    const resultElement = document.getElementById('result');

    try {
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        // Load data from u.item and u.data
        await loadData();

        // Fill dropdown
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // data.js already shows a friendly error in #result
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

// -------------------- Option 1: Select a movie --------------------

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
            resultElement.textContent = "Error: selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }

        resultElement.textContent = "Calculating recommendations (cosine on genres)...";
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
                    resultElement.textContent =
                        `Because you liked "${likedMovie.title}", we recommend (genre cosine): ` +
                        recommendationTitles.join(', ');
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

// -------------------- Option 2: Overview + theme --------------------

function extractKeywordsFromOverview(description) {
    const keywords = [];
    const text = description.toLowerCase();

    // Simple keyword extraction rules based on movie description
    if (text.includes("space") || text.includes("alien") || text.includes("galaxy")) {
        keywords.push("sci-fi");
    }
    if (text.includes("magic") || text.includes("dragon") || text.includes("wizard")) {
        keywords.push("fantasy");
    }
    if (text.includes("detective") || text.includes("police") || text.includes("murder")) {
        keywords.push("crime");
    }
    if (text.includes("love") || text.includes("romance")) {
        keywords.push("romantic");
    }
    if (text.includes("war") || text.includes("soldier") || text.includes("battle")) {
        keywords.push("war");
    }
    if (text.includes("horror") || text.includes("ghost") || text.includes("monster")) {
        keywords.push("horror");
    }
    if (text.includes("family") || text.includes("kids") || text.includes("children")) {
        keywords.push("family");
    }
    if (text.includes("comedy")) {
        keywords.push("comedy");
    }

    return keywords;
}

function getRecommendationsFromOverview() {
    const resultElement = document.getElementById('result-overview');
    const descriptionInput = document.getElementById('overview-text');

    const description = descriptionInput.value.trim();

    if (!description) {
        resultElement.textContent = "Please provide a movie description.";
        resultElement.className = 'error';
        return;
    }

    resultElement.textContent = "Analyzing your description and calculating recommendations...";
    resultElement.className = 'loading';

    setTimeout(() => {
        try {
            const keywords = extractKeywordsFromOverview(description);
            if (keywords.length === 0) {
                resultElement.textContent = "No keywords found in your description.";
                resultElement.className = 'error';
                return;
            }

            // Convert the extracted keywords into a vector
            const queryVector = genreNames.map(genre => (keywords.includes(genre.toLowerCase()) ? 1 : 0));

            const scoredMovies = movies.map(movie => {
                const movieVector = genreVectorFor(movie);
                const score = cosineSimilarity(queryVector, movieVector);
                return { ...movie, score };
            });

            scoredMovies.sort((a, b) => b.score - a.score);

            const topRecommendations = scoredMovies.slice(0, 3);

            if (topRecommendations.length > 0) {
                const titles = topRecommendations.map(m => m.title);
                resultElement.textContent =
                    "Based on your description, we recommend: " +
                    titles.join(', ');
                resultElement.className = 'success';
            } else {
                resultElement.textContent =
                    "No recommendations found for this description.";
                resultElement.className = 'error';
            }
        } catch (error) {
            console.error('Error in description-based recommendation:', error);
            resultElement.textContent =
                "An error occurred while calculating recommendations.";
            resultElement.className = 'error';
        }
    }, 100);
}
