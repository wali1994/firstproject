// app.js

// Sample emotion-based content (in a real app, this would be dynamically loaded from a data source)
const contentDatabase = [
    { emotion: 0, recommendation: "Take a deep breath and relax with some meditation." },
    { emotion: 2, recommendation: "Try a mindfulness exercise to improve focus." },
    { emotion: 5, recommendation: "Take a walk in nature or listen to calming music." },
    { emotion: 8, recommendation: "You might enjoy a motivational podcast or journaling." },
    { emotion: 10, recommendation: "Engage in some physical activity to release endorphins." }
];

// Function to match mood with content
function getRecommendation(moodValue) {
    // Find the closest match for the mood value
    let closestMatch = contentDatabase.reduce((prev, curr) => 
        Math.abs(curr.emotion - moodValue) < Math.abs(prev.emotion - moodValue) ? curr : prev
    );
    return closestMatch.recommendation;
}

export { getRecommendation };
