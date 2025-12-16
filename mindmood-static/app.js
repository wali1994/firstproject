/**
 * MindMood - Emotion-Aware Mental Wellness Recommender
 * Main Application Logic (Client-Side Mock)
 * 
 * This module handles:
 * - MOCK Emotion detection (simulated)
 * - Hybrid recommendation system (CBF + CF)
 * - User feedback and learning
 */

// ============================================
// Emotion Classes & Emoji Mapping (EmovDB Dataset)
// ============================================
const EMOTIONS = {
    neutral: { emoji: '😐', color: '#94a3b8', label: 'Neutral' },
    amused: { emoji: '😄', color: '#fbbf24', label: 'Amused' },
    angry: { emoji: '😠', color: '#ef4444', label: 'Angry' },
    disgusted: { emoji: '🤢', color: '#10b981', label: 'Disgusted' },
    sleepy: { emoji: '😴', color: '#8b5cf6', label: 'Sleepy' }
};

const EMOTION_LABELS = Object.keys(EMOTIONS);

// ============================================
// Content Database (Updated for EmovDB Emotions)
// ============================================
const contentDatabase = {
    podcasts: [
        {
            id: 'pod_001',
            title: 'Mindful Mornings',
            description: 'Start your day with guided mindfulness and positive affirmations',
            category: 'meditation',
            emoji: '🎙',
            emotionalMatch: { neutral: 0.7, amused: 0.5, angry: 0.3, disgusted: 0.4, sleepy: 0.8 },
            intensity: 0.4,
            duration: 15,
            tags: ['meditation', 'morning', 'breathing']
        },
        {
            id: 'pod_002',
            title: 'Laugh Out Loud Comedy',
            description: 'Hilarious comedy stories and jokes to brighten your day',
            category: 'entertainment',
            emoji: '🎙',
            emotionalMatch: { neutral: 0.6, amused: 0.95, angry: 0.5, disgusted: 0.3, sleepy: 0.2 },
            intensity: 0.8,
            duration: 45,
            tags: ['comedy', 'humor', 'entertainment']
        },
        {
            id: 'pod_003',
            title: 'Anger Management Strategies',
            description: 'Practical techniques for managing anger and frustration',
            category: 'mental-health',
            emoji: '🎙',
            emotionalMatch: { neutral: 0.8, amused: 0.3, angry: 0.9, disgusted: 0.6, sleepy: 0.2 },
            intensity: 0.6,
            duration: 30,
            tags: ['anger', 'mental-health', 'coping']
        },
        {
            id: 'pod_004',
            title: 'Motivational Talks Daily',
            description: 'Daily motivational stories to inspire and energize you',
            category: 'motivation',
            emoji: '🎙',
            emotionalMatch: { neutral: 0.5, amused: 0.8, angry: 0.4, disgusted: 0.5, sleepy: 0.1 },
            intensity: 0.8,
            duration: 20,
            tags: ['motivation', 'inspiration', 'growth']
        },
        {
            id: 'pod_005',
            title: 'Sleep Stories & Relaxation',
            description: 'Soothing stories and sounds to help you sleep peacefully',
            category: 'sleep',
            emoji: '🎙',
            emotionalMatch: { neutral: 0.8, amused: 0.2, angry: 0.1, disgusted: 0.2, sleepy: 0.95 },
            intensity: 0.2,
            duration: 60,
            tags: ['sleep', 'relaxation', 'bedtime']
        }
    ],
    music: [
        {
            id: 'mus_001',
            title: 'Ambient Relaxation',
            description: 'Peaceful ambient music for deep relaxation and meditation',
            category: 'ambient',
            emoji: '🎵',
            emotionalMatch: { neutral: 0.8, amused: 0.2, angry: 0.1, disgusted: 0.2, sleepy: 0.95 },
            intensity: 0.2,
            duration: 60,
            tags: ['ambient', 'relaxation', 'sleep']
        },
        {
            id: 'mus_002',
            title: 'Uplifting Pop Hits',
            description: 'Feel-good pop songs to boost your mood and energy',
            category: 'pop',
            emoji: '🎵',
            emotionalMatch: { neutral: 0.5, amused: 0.95, angry: 0.4, disgusted: 0.3, sleepy: 0.1 },
            intensity: 0.9,
            duration: 45,
            tags: ['pop', 'uplifting', 'energy']
        },
        {
            id: 'mus_003',
            title: 'Heavy Metal & Rock',
            description: 'Intense rock music for releasing tension and anger',
            category: 'rock',
            emoji: '🎵',
            emotionalMatch: { neutral: 0.4, amused: 0.3, angry: 0.9, disgusted: 0.5, sleepy: 0.1 },
            intensity: 0.95,
            duration: 50,
            tags: ['rock', 'metal', 'intense']
        },
        {
            id: 'mus_004',
            title: 'Energetic Workout Mix',
            description: 'High-energy tracks to power through your workout',
            category: 'electronic',
            emoji: '🎵',
            emotionalMatch: { neutral: 0.5, amused: 0.7, angry: 0.6, disgusted: 0.2, sleepy: 0.05 },
            intensity: 0.95,
            duration: 60,
            tags: ['workout', 'energy', 'electronic']
        },
        {
            id: 'mus_005',
            title: 'Chill Lo-Fi Beats',
            description: 'Relaxing lo-fi hip-hop for studying and unwinding',
            category: 'lofi',
            emoji: '🎵',
            emotionalMatch: { neutral: 0.9, amused: 0.4, angry: 0.2, disgusted: 0.3, sleepy: 0.8 },
            intensity: 0.3,
            duration: 90,
            tags: ['lofi', 'chill', 'study']
        }
    ],
    meditations: [
        {
            id: 'med_001',
            title: 'Body Scan Meditation',
            description: 'Progressive relaxation through body awareness',
            category: 'body-scan',
            emoji: '🧘',
            emotionalMatch: { neutral: 0.8, amused: 0.3, angry: 0.2, disgusted: 0.3, sleepy: 0.9 },
            intensity: 0.3,
            duration: 20,
            tags: ['body-scan', 'relaxation', 'awareness']
        },
        {
            id: 'med_002',
            title: 'Loving-Kindness Meditation',
            description: 'Cultivate compassion and kindness towards yourself and others',
            category: 'loving-kindness',
            emoji: '🧘',
            emotionalMatch: { neutral: 0.7, amused: 0.7, angry: 0.4, disgusted: 0.5, sleepy: 0.5 },
            intensity: 0.5,
            duration: 15,
            tags: ['loving-kindness', 'compassion', 'mindfulness']
        },
        {
            id: 'med_003',
            title: 'Breathing Exercises',
            description: 'Simple breathing techniques to calm your nervous system',
            category: 'breathing',
            emoji: '🧘',
            emotionalMatch: { neutral: 0.8, amused: 0.4, angry: 0.6, disgusted: 0.4, sleepy: 0.7 },
            intensity: 0.4,
            duration: 10,
            tags: ['breathing', 'anxiety', 'calm']
        },
        {
            id: 'med_004',
            title: 'Guided Visualization',
            description: 'Journey to peaceful places through guided imagery',
            category: 'visualization',
            emoji: '🧘',
            emotionalMatch: { neutral: 0.7, amused: 0.5, angry: 0.3, disgusted: 0.4, sleepy: 0.8 },
            intensity: 0.5,
            duration: 25,
            tags: ['visualization', 'imagery', 'relaxation']
        },
        {
            id: 'med_005',
            title: 'Mindfulness of Thoughts',
            description: 'Learn to observe thoughts without judgment',
            category: 'mindfulness',
            emoji: '🧘',
            emotionalMatch: { neutral: 0.9, amused: 0.4, angry: 0.3, disgusted: 0.4, sleepy: 0.6 },
            intensity: 0.6,
            duration: 18,
            tags: ['mindfulness', 'thoughts', 'awareness']
        }
    ],
    books: [
        {
            id: 'book_001',
            title: 'The Courage to Be Disliked',
            description: 'A dialogue exploring Adlerian psychology and personal freedom',
            category: 'psychology',
            emoji: '📚',
            emotionalMatch: { neutral: 0.8, amused: 0.5, angry: 0.4, disgusted: 0.5, sleepy: 0.4 },
            intensity: 0.6,
            duration: 320,
            tags: ['psychology', 'philosophy', 'growth']
        },
        {
            id: 'book_002',
            title: 'The Midnight Library',
            description: 'A magical tale about second chances and life choices',
            category: 'fiction',
            emoji: '📚',
            emotionalMatch: { neutral: 0.6, amused: 0.7, angry: 0.3, disgusted: 0.3, sleepy: 0.5 },
            intensity: 0.7,
            duration: 288,
            tags: ['fiction', 'fantasy', 'hope']
        },
        {
            id: 'book_003',
            title: 'Atomic Habits',
            description: 'Build good habits and break bad ones with practical strategies',
            category: 'self-help',
            emoji: '📚',
            emotionalMatch: { neutral: 0.9, amused: 0.4, angry: 0.3, disgusted: 0.4, sleepy: 0.2 },
            intensity: 0.7,
            duration: 320,
            tags: ['habits', 'self-improvement', 'productivity']
        },
        {
            id: 'book_004',
            title: 'Milk and Honey',
            description: 'Poetry collection exploring pain, love, and healing',
            category: 'poetry',
            emoji: '📚',
            emotionalMatch: { neutral: 0.7, amused: 0.3, angry: 0.4, disgusted: 0.5, sleepy: 0.6 },
            intensity: 0.5,
            duration: 256,
            tags: ['poetry', 'emotion', 'healing']
        },
        {
            id: 'book_005',
            title: 'Why Buddhism is True',
            description: 'Explore Buddhist philosophy and its relevance to modern life',
            category: 'spirituality',
            emoji: '📚',
            emotionalMatch: { neutral: 0.85, amused: 0.4, angry: 0.2, disgusted: 0.3, sleepy: 0.5 },
            intensity: 0.6,
            duration: 400,
            tags: ['buddhism', 'spirituality', 'mindfulness']
        }
    ]
};

// ============================================
// MOCK Emotion Detection Class (Client-Side)
// Simulates a trained model's output for demonstration purposes.
// ============================================
class EmotionDetector {
    constructor() {
        this.keywordMap = {
            amused: ['happy', 'joy', 'excited', 'wonderful', 'great', 'love', 'amazing', 'fantastic', 'laugh', 'fun', 'funny', 'tuyệt vời', 'vui', 'hạnh phúc', 'cười', 'thích'],
            angry: ['angry', 'furious', 'mad', 'frustrated', 'annoyed', 'irritated', 'rage', 'upset', 'hate', 'giận', 'bực', 'tức', 'ghét', 'khó chịu'],
            disgusted: ['disgusted', 'repulsed', 'gross', 'yuck', 'awful', 'horrible', 'disgusting', 'kinh tởm', 'ghê', 'buồn nôn', 'tởm'],
            sleepy: ['tired', 'sleepy', 'exhausted', 'drowsy', 'fatigue', 'worn out', 'drained', 'mệt', 'buồn ngủ', 'kiệt sức', 'uể oải'],
            neutral: ['okay', 'fine', 'normal', 'alright', 'meh', 'so-so', 'bình thường', 'ổn', 'tạm được', 'không sao']
        };
    }

    /**
     * Simulates voice analysis. Since we cannot analyze raw audio in the browser,
     * this returns a random, but plausible, result.
     * @param {Blob} audioBlob - The recorded audio data (ignored).
     * @returns {Promise<object>} - Simulated detection result.
     */
    async detectFromVoice(audioBlob) {
        // Simulate a slight delay for realism
        await new Promise(resolve => setTimeout(resolve, 500));

        // Randomly select a primary emotion
        const primaryEmotion = EMOTION_LABELS[Math.floor(Math.random() * EMOTION_LABELS.length)];
        
        let emotionScores = {};
        let maxScore = 0.7 + Math.random() * 0.2; // 70% to 90% confidence
        let remainingScore = 1.0 - maxScore;

        // Assign max score to primary emotion
        emotionScores[primaryEmotion] = maxScore;

        // Distribute remaining score to others
        const otherEmotions = EMOTION_LABELS.filter(e => e !== primaryEmotion);
        let scoreSum = 0;
        otherEmotions.forEach(e => {
            emotionScores[e] = Math.random();
            scoreSum += emotionScores[e];
        });

        // Normalize the remaining scores
        otherEmotions.forEach(e => {
            emotionScores[e] = (emotionScores[e] / scoreSum) * remainingScore;
        });

        // Final normalization check (should be close to 1)
        const finalSum = Object.values(emotionScores).reduce((a, b) => a + b, 0);
        
        return {
            emotion_scores: emotionScores,
            primary_emotion: primaryEmotion,
            confidence: emotionScores[primaryEmotion] / finalSum // Recalculate confidence based on final sum
        };
    }

    /**
     * Analyzes emotion from text input using a keyword-based approach.
     * @param {string} text - The text input.
     * @returns {Promise<object>} - Detection result.
     */
    async detectFromText(text) {
        await new Promise(resolve => setTimeout(resolve, 300));

        const textLower = text.toLowerCase();
        let emotionScores = {};

        // Initialize all emotions with a small base weight
        const baseWeight = 0.1;
        EMOTION_LABELS.forEach(emotion => {
            emotionScores[emotion] = baseWeight;
        });

        // Count keyword matches
        for (const [emotion, keywords] of Object.entries(this.keywordMap)) {
            for (const keyword of keywords) {
                if (textLower.includes(keyword)) {
                    emotionScores[emotion] += 0.2; // Boost score for each keyword match
                }
            }
        }

        // Normalize scores to sum to 1
        let sumScores = Object.values(emotionScores).reduce((a, b) => a + b, 0);
        if (sumScores === 0) {
            // Default to neutral if no keywords found
            emotionScores['neutral'] = 1.0;
            sumScores = 1.0;
        } else {
            for (const emotion in emotionScores) {
                emotionScores[emotion] /= sumScores;
            }
        }
        
        // Determine primary emotion and confidence
        const primaryEmotion = EMOTION_LABELS.reduce((a, b) => emotionScores[a] > emotionScores[b] ? a : b);
        const confidence = emotionScores[primaryEmotion];

        return {
            emotion_scores: emotionScores,
            primary_emotion: primaryEmotion,
            confidence: confidence
        };
    }
}

// ============================================
// Hybrid Recommender System
// (No changes needed here, as it's client-side logic)
// ============================================
class HybridRecommender {
    constructor() {
        this.userHistory = [];
        this.userFeedback = {};
        this.contentFeatures = this._initializeContentFeatures();
    }

    /**
     * Initialize content feature vectors based on emotional match
     */
    _initializeContentFeatures() {
        const features = {};

        Object.entries(contentDatabase).forEach(([category, items]) => {
            features[category] = {};
            items.forEach(item => {
                features[category][item.id] = item.emotionalMatch;
            });
        });

        return features;
    }

    /**
     * Content-Based Filtering: Find similar content
     */
    _contentBasedFiltering(emotionScores, category) {
        const recommendations = [];
        const items = contentDatabase[category];

        items.forEach(item => {
            // Calculate cosine similarity between emotion vector and content features
            const similarity = this._cosineSimilarity(emotionScores, item.emotionalMatch);
            
            // Adjust by intensity appropriateness
            const intensityMatch = 1 - Math.abs(item.intensity - 0.5);
            
            const score = similarity * 0.8 + intensityMatch * 0.2;

            recommendations.push({
                ...item,
                cbfScore: score,
                similarity: similarity
            });
        });

        return recommendations.sort((a, b) => b.cbfScore - a.cbfScore);
    }

    /**
     * Collaborative Filtering: Boost based on user history
     */
    _collaborativeFiltering(emotionScores, category, recommendations) {
        // Find similar emotions in user history
        const similarHistoryItems = this.userHistory.filter(item => {
            const historySimilarity = this._cosineSimilarity(
                emotionScores,
                this._emotionVectorFromHistory(item)
            );
            return historySimilarity > 0.6;
        });

        // Boost recommendations that were liked in similar emotional states
        recommendations.forEach(rec => {
            let cfBoost = 0;
            similarHistoryItems.forEach(histItem => {
                if (this.userFeedback[histItem.contentId] === 'helpful') {
                    cfBoost += 0.1;
                }
            });
            rec.cfScore = Math.min(cfBoost, 0.3);
        });

        return recommendations;
    }

    /**
     * Hybrid fusion: Combine CBF and CF scores
     */
    _hybridFusion(recommendations) {
        const alpha = 0.6;  // Content-based weight
        const beta = 0.3;   // Collaborative weight
        const gamma = 0.1;  // Temporal boost

        recommendations.forEach(rec => {
            const cfScore = rec.cfScore || 0;
            const hybridScore = (alpha * rec.cbfScore) + (beta * cfScore) + (gamma * 0.1);
            rec.hybridScore = hybridScore;
        });

        return recommendations.sort((a, b) => b.hybridScore - a.hybridScore);
    }

    /**
     * Generate recommendations for all categories
     */
    generateRecommendations(emotionScores) {
        const recommendations = {};

        Object.keys(contentDatabase).forEach(category => {
            // Step 1: Content-Based Filtering
            let recs = this._contentBasedFiltering(emotionScores, category);

            // Step 2: Collaborative Filtering
            recs = this._collaborativeFiltering(emotionScores, category, recs);

            // Step 3: Hybrid Fusion
            recs = this._hybridFusion(recs);

            // Step 4: Diversity filter (avoid repetition)
            recs = this._applyDiversityFilter(recs);

            recommendations[category] = recs.slice(0, 3); // Top 3 per category
        });

        return recommendations;
    }

    /**
     * Apply diversity filter to avoid recommending same content
     */
    _applyDiversityFilter(recommendations) {
        const filtered = [];
        const seenTags = new Set();

        recommendations.forEach(rec => {
            let isDiverse = true;
            rec.tags.forEach(tag => {
                if (seenTags.has(tag)) {
                    isDiverse = false;
                }
            });

            if (isDiverse) {
                filtered.push(rec);
                rec.tags.forEach(tag => seenTags.add(tag));
            }
        });

        return filtered.length > 0 ? filtered : recommendations.slice(0, 3);
    }

    /**
     * Cosine similarity between two vectors
     */
    _cosineSimilarity(vec1, vec2) {
        let dotProduct = 0;
        let magnitude1 = 0;
        let magnitude2 = 0;

        Object.keys(vec1).forEach(key => {
            dotProduct += vec1[key] * (vec2[key] || 0);
            magnitude1 += vec1[key] * vec1[key];
        });

        Object.keys(vec2).forEach(key => {
            magnitude2 += vec2[key] * vec2[key];
        });

        magnitude1 = Math.sqrt(magnitude1);
        magnitude2 = Math.sqrt(magnitude2);

        if (magnitude1 === 0 || magnitude2 === 0) return 0;
        return dotProduct / (magnitude1 * magnitude2);
    }

    /**
     * Convert history item to emotion vector
     */
    _emotionVectorFromHistory(item) {
        const vector = {};
        Object.keys(EMOTIONS).forEach(emotion => {
            vector[emotion] = 0.2; // Base weight
        });
        vector[item.emotion] = 1;
        // Normalize
        const sum = Object.values(vector).reduce((a, b) => a + b, 0);
        Object.keys(vector).forEach(emotion => {
            vector[emotion] = vector[emotion] / sum;
        });
        return vector;
    }

    /**
     * Record user feedback
     */
    recordFeedback(contentId, feedback) {
        this.userFeedback[contentId] = feedback;
    }

    /**
     * Record detection in history
     */
    recordDetection(emotion, confidence, emotionScores) {
        this.userHistory.push({
            emotion: emotion,
            confidence: confidence,
            scores: emotionScores,
            timestamp: new Date(),
            contentId: null
        });
    }
}

// ============================================
// Global Instances - Initialize on page load
// ============================================
let emotionDetector;
let recommender;

// Initialize when DOM is ready
function initializeApp() {
    // The emotionDetector is now the client-side class instance
    emotionDetector = new EmotionDetector();
    
    if (!recommender) {
        recommender = new HybridRecommender();
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// ============================================
// Export for use in script.js
// ============================================
// EMOTIONS, contentDatabase, emotionDetector, recommender are now globally available.
