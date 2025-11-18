// ---------------------- Stage 2–4: LLM-style pipeline ----------------------
//
// We simulate the slide pipeline using rule-based code instead of a real LLM.
// For each movie.description we do:
//
// 1) Extract Features     -> subGenres, themes
// 2) Consolidate Keywords -> map synonyms to master lists
// 3) Finalize & Encode    -> one-hot vectors for subGenres and themes
//
// Then we concatenate
// [genre vector | sub-genre vector | theme vector]
// and use cosine similarity on this final vector.
//

// Master lists used in Stage 3 / Stage 4 of the slide
const masterSubGenres = [
    "Science Fiction",
    "Fantasy",
    "Action Thriller",
    "Romantic Comedy",
    "Crime Drama",
    "Family",
    "Animation"
];

const masterThemes = [
    "Good vs Evil",
    "Coming of Age",
    "Redemption",
    "Hope",
    "Friendship",
    "Love",
    "Revenge"
];

// Cache for encoded feature vectors per movie
const movieFeatureCache = new Map();

// Stage 2: Extract raw features from description
function extractFeatures(description) {
    const text = description.toLowerCase();

    const rawSubGenres = [];
    const rawThemes = [];

    // Very simple keyword rules to mimic LLM behavior
    if (text.includes("space") || text.includes("alien") || text.includes("galaxy")) {
        rawSubGenres.push("space opera", "sci-fi");
    }
    if (text.includes("magic") || text.includes("wizard") || text.includes("dragon")) {
        rawSubGenres.push("fantasy");
    }
    if (text.includes("police") || text.includes("detective") || text.includes("murder")) {
        rawSubGenres.push("crime", "thriller");
    }
    if (text.includes("love") || text.includes("romance")) {
        rawSubGenres.push("romantic comedy");
        rawThemes.push("love");
    }
    if (text.includes("family") || text.includes("kids") || text.includes("children")) {
        rawSubGenres.push("family");
        rawThemes.push("friendship");
    }
    if (text.includes("war") || text.includes("battle")) {
        rawSubGenres.push("war");
        rawThemes.push("good vs evil");
    }
    if (text.includes("school") || text.includes("teen") || text.includes("growing up")) {
        rawThemes.push("coming of age");
    }
    if (text.includes("hope") || text.includes("dream")) {
        rawThemes.push("hope");
    }
    if (text.includes("revenge")) {
        rawThemes.push("revenge");
    }

    return {
        subGenres: [...new Set(rawSubGenres)],
        themes: [...new Set(rawThemes)]
    };
}

// Stage 3: Consolidate keywords to master lists
function consolidateKeywords(rawFeatures) {
    const normalizedSubGenres = new Set();
    const normalizedThemes = new Set();

    rawFeatures.subGenres.forEach(sg => {
        const t = sg.toLowerCase();
        if (t.includes("sci") || t.includes("space")) {
            normalizedSubGenres.add("Science Fiction");
        } else if (t.includes("fantasy") || t.includes("dragon") || t.includes("magic")) {
            normalizedSubGenres.add("Fantasy");
        } else if (t.includes("thriller") || t.includes("crime")) {
            normalizedSubGenres.add("Action Thriller");
            normalizedSubGenres.add("Crime Drama");
        } else if (t.includes("romantic")) {
            normalizedSubGenres.add("Romantic Comedy");
        } else if (t.includes("family") || t.includes("children")) {
            normalizedSubGenres.add("Family");
        } else if (t.includes("animation") || t.includes("cartoon")) {
            normalizedSubGenres.add("Animation");
        }
    });

    rawFeatures.themes.forEach(th => {
        const t = th.toLowerCase();
        if (t.includes("good") || t.includes("evil") || t.includes("war")) {
            normalizedThemes.add("Good vs Evil");
        } else if (t.includes("coming of age") || t.includes("grow")) {
            normalizedThemes.add("Coming of Age");
        } else if (t.includes("redeem")) {
            normalizedThemes.add("Redemption");
        } else if (t.includes("hope") || t.includes("dream")) {
            normalizedThemes.add("Hope");
        } else if (t.includes("friend")) {
            normalizedThemes.add("Friendship");
        } else if (t.includes("love") || t.includes("romance")) {
            normalizedThemes.add("Love");
        } else if (t.includes("revenge")) {
            normalizedThemes.add("Revenge");
        }
    });

    return {
        subGenres: Array.from(normalizedSubGenres),
        themes: Array.from(normalizedThemes)
    };
}

// Stage 4: Finalize and encode as one-hot vectors
function encodeFinalFeatures(consolidatedFeatures) {
    const subGenreVector = masterSubGenres.map(name =>
        consolidatedFeatures.subGenres.includes(name) ? 1 : 0
    );

    const themeVector = masterThemes.map(name =>
        consolidatedFeatures.themes.includes(name) ? 1 : 0
    );

    return { subGenreVector, themeVector };
}

// Build the full feature vector for a movie
function fullFeatureVector(movie) {
    if (movieFeatureCache.has(movie.id)) {
        return movieFeatureCache.get(movie.id);
    }

    // Genre vector (from MovieLens genres)
    const genreVector = genreNames.map(genre =>
        movie.genres.includes(genre) ? 1 : 0
    );

    // LLM-style pipeline using description / overview
    const raw = extractFeatures(movie.description || "");
    const consolidated = consolidateKeywords(raw);
    const encoded = encodeFinalFeatures(consolidated);

    // Final vector = [genres | sub-genre one-hot | theme one-hot]
    const vector = [
        ...genreVector,
        ...encoded.subGenreVector,
        ...encoded.themeVector
    ];

    movieFeatureCache.set(movie.id, vector);
    return vector;
}

// ---------------------- Cosine similarity helpers ----------------------

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

// ---------------------- UI init ----------------------

window.onload = async function () {
    const resultElement = document.getElementById('result');

    try {
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';

        await loadData();

        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
    }
};

function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');

    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// ---------------------- Recommendation logic ----------------------

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

        resultElement.textContent =
            "Running feature pipeline and calculating cosine similarity...";
        resultElement.className = 'loading';

        setTimeout(() => {
            try {
                const likedVector = fullFeatureVector(likedMovie);

                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);

                const scoredMovies = candidateMovies.map(candidate => {
                    const candidateVector = fullFeatureVector(candidate);
                    const score = cosineSimilarity(likedVector, candidateVector);
                    return { ...candidate, score };
                });

                scoredMovies.sort((a, b) => b.score - a.score);

                const topRecommendations = scoredMovies.slice(0, 2);

                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(m => m.title);
                    resultElement.textContent =
                        `Because you liked "${likedMovie.title}", ` +
                        `we recommend (cosine on genres + overview features): ` +
                        recommendationTitles.join(', ') + `.`;
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
