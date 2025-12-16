// Mood Data
const MOOD_DATA = {
    moods: {
        amused: { emoji: "😄", color: "#fbbf24", recommendation: "You're feeling amused! Keep enjoying this light-hearted moment. Share laughter with others and engage in activities that bring you joy." },
        angry: { emoji: "😠", color: "#ef4444", recommendation: "You're feeling angry. Try taking deep breaths, going for a walk, or engaging in physical activity to channel this energy." },
        disgusted: { emoji: "🤢", color: "#8b5cf6", recommendation: "You're feeling disgusted. Take a step back from the situation. Engage in activities that refresh your mind and spirit." },
        neutral: { emoji: "😐", color: "#9ca3af", recommendation: "You're feeling neutral. This is a balanced state perfect for reflection and planning. Use this time to assess your goals." },
        sleepy: { emoji: "😴", color: "#60a5fa", recommendation: "You're feeling sleepy. Your body might need rest. Consider taking a nap or engaging in a relaxing activity." },
        sad: { emoji: "😢", color: "#6366f1", recommendation: "You're feeling sad. It's okay to take a break and process your emotions. Consider talking to a friend or journaling your thoughts." },
        excited: { emoji: "🤩", color: "#10b981", recommendation: "You're feeling excited! This is a great time to pursue new opportunities or try something adventurous." }
    },
};

// Function to analyze mood based on text
async function analyzeMood(type) {
    let mood = 'neutral';  // Default mood is neutral
    
    if (type === 'text') {
        const text = document.getElementById("text-input").value.toLowerCase();

        // Improved mood detection based on multiple keywords
        if (text.includes("happy") || text.includes("joy") || text.includes("excited") || text.includes("amazing") || text.includes("fun")) {
            mood = 'amused';
        } else if (text.includes("angry") || text.includes("mad") || text.includes("furious") || text.includes("rage") || text.includes("annoyed")) {
            mood = 'angry';
        } else if (text.includes("disgust") || text.includes("gross") || text.includes("yuck") || text.includes("nasty")) {
            mood = 'disgusted';
        } else if (text.includes("sad") || text.includes("down") || text.includes("depressed") || text.includes("blue") || text.includes("gloomy")) {
            mood = 'sad';
        } else if (text.includes("sleepy") || text.includes("tired") || text.includes("exhausted") || text.includes("fatigued") || text.includes("lazy")) {
            mood = 'sleepy';
        } else if (text.includes("excited") || text.includes("pumped") || text.includes("energized") || text.includes("thrilled") || text.includes("elated")) {
            mood = 'excited';
        }
    } else if (type === 'voice') {
        // Placeholder for voice analysis
        mood = 'amused'; // For now, it's static; we'll add voice analysis logic later
    }

    displayMood(mood);
}

// Function to display the mood and recommendation
function displayMood(mood) {
    const moodInfo = MOOD_DATA.moods[mood];
    document.getElementById("mood-result").innerHTML = `
        <div class="alert" style="background-color: ${moodInfo.color}; color: white; font-size: 20px;">
            <span>${moodInfo.emoji} You're feeling ${mood}!</span>
            <p><strong>Recommendation:</strong> ${moodInfo.recommendation}</p>
        </div>
    `;
}

// Variables for voice recording
// Variables for voice recording
let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let recordingInterval;

// Start recording (using MediaRecorder API)
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }) // Request microphone access
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream); // Create MediaRecorder instance
            audioChunks = []; // Clear previous audio chunks

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data); // Push recorded audio data
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById("recorded-audio").src = audioUrl; // Display recorded audio
                document.getElementById("analyze-recording-btn").disabled = false; // Enable the analyze button
            };

            mediaRecorder.start(); // Start recording

            // Show recording UI changes
            document.getElementById("start-record-btn").style.display = "none"; // Hide Start button
            document.getElementById("stop-record-btn").style.display = "inline-block"; // Show Stop button
            document.getElementById("recording-timer").style.display = "block"; // Show recording timer

            // Start timer
            recordingStartTime = Date.now();
            recordingInterval = setInterval(updateRecordingTime, 1000); // Update the timer every second
        })
        .catch(error => {
            console.error("Error accessing the microphone:", error);
            alert("Microphone access denied. Please enable microphone permissions.");
        });
}

// Stop recording (voice)
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop(); // Stop recording
        mediaRecorder.stream.getTracks().forEach(track => track.stop()); // Stop the stream to release the microphone

        // Hide stop button and show start button again
        document.getElementById("stop-record-btn").style.display = "none";
        document.getElementById("start-record-btn").style.display = "inline-block";
        document.getElementById("recording-timer").style.display = "none"; // Hide timer
    }

    // Stop the timer
    clearInterval(recordingInterval);
}

// Update the recording timer
function updateRecordingTime() {
    const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000); // Calculate elapsed time in seconds
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    document.getElementById("timer").textContent =
        `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`; // Format time
}

// Clear recording (reset audio)
function clearRecording() {
    document.getElementById("recorded-audio").src = ""; // Clear the audio playback
    document.getElementById("analyze-recording-btn").disabled = true; // Disable analyze button
    audioChunks = []; // Clear audio chunks
}
