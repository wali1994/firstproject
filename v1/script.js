// script.js

import { getTextRecommendation } from './app.js';

// DOM elements
const textOptionBtn = document.getElementById("textOption");
const audioOptionBtn = document.getElementById("audioOption");

const textRecommendationSection = document.getElementById("textRecommendation");
const audioRecommendationSection = document.getElementById("audioRecommendation");

const textInput = document.getElementById("textInput");
const textOutput = document.getElementById("textOutput");

const audioFileInput = document.getElementById("audioFile");
const audioOutput = document.getElementById("audioOutput");

// Switch to text-based recommendation section
textOptionBtn.addEventListener("click", function() {
    textRecommendationSection.classList.add("active");
    audioRecommendationSection.classList.remove("active");
});

// Switch to audio-based recommendation section
audioOptionBtn.addEventListener("click", function() {
    audioRecommendationSection.classList.add("active");
    textRecommendationSection.classList.remove("active");
});

// Handle text-based recommendation
document.getElementById("textSubmit").addEventListener("click", function() {
    const moodText = textInput.value.trim();
    const recommendation = getTextRecommendation(moodText);
    textOutput.textContent = recommendation;
});

// Handle audio-based recommendation (for now, only simulating it)
document.getElementById("audioSubmit").addEventListener("click", function() {
    const file = audioFileInput.files[0];
    if (!file) {
        audioOutput.textContent = "Please upload an audio file.";
        return;
    }

    // Simulate audio-based recommendation (replace this with actual processing later)
    audioOutput.textContent = "Processing your audio file... (Simulated)";
});
