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

// --- Basic helpers (shared by both options) ---

// Create a binary genre vector for a movie
function genreVectorFor(movie) {
    return genreNames.map(genre => (movie.genres.includes(genre) ? 1 : 0));
}

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
//
// Here we follow the slide idea:
// 1. Read raw overview text.
// 2. Extract simple features (keywords).
// 3. Map them to canonical genres (our master list is genreNames).
// 4. Build a one-hot vector and use the SAME cosine similarity.
//

// Simple rules to map description keywords and theme to MovieLens genres
function descriptionToGenreVector(description, themeLabel) {
    const text = (description + ' ' + (themeLabel || '')).toLowerCase();

    const activeGenres = new Set();

    // Sci-Fi / space style
    if (text.includes('space') ||
        text.includes('galaxy') ||
        text.includes('alien') ||
        text.includes('future') ||
        text.includes('robot')) {
        activeGenres.add('Sci-Fi');
        activeGenres.add('Action');
    }

    // Fantasy
    if (text.includes('magic') ||
        text.includes('wizard') ||
        text.includes('dragon') ||
        text.includes('kingdom')) {
        activeGenres.add('Fantasy');
        activeGenres.add("Children's");
    }

    // Crime / thriller / mystery
    if (text.includes('detective') ||
        text.includes('police') ||
        text.includes('murder') ||
        text.includes('case') ||
        text.includes('criminal')) {
        activeGenres.add('Crime');
        activeGenres.add('Thriller');
        activeGenres.add('Mystery');
    }

    // Romance / love
    if (text.includes('love') ||
        text.includes('romance') ||
        text.includes('wedding') ||
        text.includes('relationship')) {
        activeGenres.add('Romance');
        activeGenres.add('Comedy');
    }

    // War / conflict
    if (text.includes('war') ||
        text.includes('battle') ||
        text.includes('soldier') ||
        text.includes('army')) {
        activeGenres.add('War');
        activeGenres.add('Action');
    }

    // Horror
    if (text.includes('ghost') ||
        text.includes('monster') ||
        text.includes('haunted') ||
        text.includes('demon') ||
        text.includes('zombie') ||
        text.includes('killer')) {
        activeGenres.add('Horror');
        activeGenres.add('Thriller');
    }

    // Family / kids / friendship
    if (text.includes('family') ||
        text.includes('kids') ||
        text.includes('child') ||
        text.includes('friends') ||
        text.includes('school')) {
        activeGenres.add("Children's");
        activeGenres.add('Comedy');
        activeGenres.add('Drama');
    }

    // Use explicit theme dropdown as extra hint
    if (themeLabel) {
        const t = themeLabel.toLowerCase();
        if (t.includes('good vs evil') || t.includes('war')) {
            activeGenres.add('Action');
            activeGenres.add('Sci-Fi');
            activeGenres.add('War');
        } else if (t.includes('coming of age')) {
            activeGenres.add('Drama');
        } else if (t.includes('romance') || t.includes('love')) {
            activeGenres.add('Romance');
            activeGenres.add('Comedy');
        } else if (t.includes('friendship')) {
            activeGenres.add('Drama');
            activeGenres.add("Children's");
        } else if (t.includes('crime')) {
            activeGenres.add('Crime');
            activeGenres.add('Thriller');
        } else if (t.includes('horror') || t.includes('fear')) {
            activeGenres.add('Horror');
            activeGenres.add('Thriller');
        } else if (t.includes('comedy') || t.includes('fun')) {
            activeGenres.add('Comedy');
        }
    }

    // Build final one-hot vector in MovieLens genre order
    const vector = genreNames.map(name => (activeGenres.has(name) ? 1 : 0));
    return vector;
}

function getRecommendationsFromOverview() {
    const resultElement = document.getElementById('result-overview');
    const descriptionInput = document.getElementById('overview-text');
    const themeSelect = document.getElementById('theme-select');

    const description = descriptionInput.value.trim();
    const themeLabel = themeSelect.value;

    if (!description) {
        resultElement.textContent = "Please write a short overview first.";
        resultElement.className = 'error';
        return;
    }

    resultElement.textContent =
        "Analyzing your description and calculating recommendations (cosine)...";
    resultElement.className = 'loading';

    setTimeout(() => {
        try {
            const queryVector = descriptionToGenreVector(description, themeLabel);

            // If vector is all zeros, we have no signal
            const hasSignal = queryVector.some(v => v !== 0);
            if (!hasSignal) {
                resultElement.textContent =
                    "Could not detect any genres from your text. Try adding more details or choosing a theme.";
                resultElement.className = 'error';
                return;
            }

            // Compute cosine similarity between query and every movie
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
                    "Based on your description we recommend: " +
                    titles.join(', ');
                resultElement.className = 'success';
            } else {
                resultElement.textContent =
                    "No recommendations found for this description.";
                resultElement.className = 'error';
            }
        } catch (error) {
            console.error('Error in description based recommendation:', error);
            resultElement.textContent =
                "An error occurred while calculating recommendations.";
            resultElement.className = 'error';
        }
    }, 100);
}
