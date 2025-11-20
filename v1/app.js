// app.js

// Sample function to get recommendations based on text input
function getTextRecommendation(text) {
    const recommendations = [
        { mood: "happy", recommendation: "Go for a walk, enjoy the sunshine!" },
        { mood: "sad", recommendation: "Consider meditating or journaling to process your feelings." },
        { mood: "anxious", recommendation: "Deep breathing exercises can help you relax." }
    ];

    for (const rec of recommendations) {
        if (text.toLowerCase().includes(rec.mood)) {
            return rec.recommendation;
        }
    }
    return "Sorry, no recommendation found based on your mood.";
}

export { getTextRecommendation };
