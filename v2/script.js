// script.js

import { getRecommendation } from './app.js';

// Get DOM elements
const moodSlider = document.getElementById("moodSlider");
const moodValueSpan = document.getElementById("moodValue");
const recommendationList = document.getElementById("recommendationList");

// Update the displayed mood value when the slider changes
moodSlider.addEventListener("input", function() {
    moodValueSpan.textContent = moodSlider.value;
    updateRecommendations(moodSlider.value);
});

// Function to update recommendations
function updateRecommendations(moodValue) {
    // Clear previous recommendations
    recommendationList.innerHTML = "";

    // Get the recommendation based on mood
    const recommendation = getRecommendation(moodValue);
    const listItem = document.createElement("li");
    listItem.textContent = recommendation;
    recommendationList.appendChild(listItem);
}

// Feedback buttons to capture user interaction
document.getElementById("positiveFeedback").addEventListener("click", () => {
    alert("Thank you for your feedback!");
});

document.getElementById("negativeFeedback").addEventListener("click", () => {
    alert("Sorry that the recommendation wasn't helpful.");
});

// Call the initial function to display recommendations for default mood value
updateRecommendations(moodSlider.value);

// Handle Voice Input Button
const voiceInputBtn = document.getElementById("voiceInputBtn");

voiceInputBtn.addEventListener("click", function() {
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-US';
    
    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        alert("You said: " + transcript);
        // Here, you could implement a sentiment analysis function to map speech to a mood score
    }

    recognition.onerror = function(event) {
        alert("Speech recognition error: " + event.error);
    }
});
