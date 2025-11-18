// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Genre names as defined in the MovieLens 100K u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

/**
 * Primary function to load data from MovieLens u.item and u.data.
 * This step matches Stage 1 from the slide: read raw data.
 */
async function loadData() {
    try {
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

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
        throw error;
    }
}

/**
 * Parse movie metadata from u.item format.
 * If your custom dataset has an "overview" or "description" column
 * you can plug it here instead of using the title as description.
 */
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue;

        const id = parseInt(fields[0]);
        const title = fields[1];

        // TODO for your own dataset:
        // const description = fields[<index_of_overview_column>];
        // For MovieLens we do not have overview so we reuse the title
        const description = title;

        // Extract genres (last 19 fields) and map to names
        const genreValues = fields.slice(5, 24).map(value => parseInt(value, 10));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        movies.push({ id, title, description, genres });
    }
}

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
