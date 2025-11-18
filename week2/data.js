// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in the MovieLens 100K u.item file
// We ignore the "unknown" genre and keep the 18 real genres.
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

/**
 * Primary function to load data from MovieLens u.item and u.data.
 * This is a pure browser fetch, so it works on GitHub Pages as long
 * as the files u.item and u.data are in the same folder.
 */
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load and parse rating data (not used in this homework, but ready)
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent =
                `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
            resultElement.className = 'error';
        }
        // Re-throw so script.js can handle the failure state if needed
        throw error;
    }
}

/**
 * Parse movie metadata from u.item format.
 * Each line: movieId | title | releaseDate | videoRelease | IMDbURL | 19 genre flags
 * We convert the 0/1 genre flags into a list of genre names.
 */
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Extract genres (last 19 fields); we map them to the 18 genreNames above
        const genreValues = fields.slice(5, 24).map(value => parseInt(value, 10));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        movies.push({ id, title, genres });
    }
}

/**
 * Parse rating data from u.data format.
 * Each line: userId \t itemId \t rating \t timestamp
 * We only load it for completeness; the current homework uses only genres.
 */
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue;

        const userId = parseInt(fields[0], 10);
        const itemId = parseInt(fields[1], 10);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3], 10);

        ratings.push({ userId, itemId, rating, timestamp });
    }
}
