// static/js/script.js

// UI Functions
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const typingIndicator = document.getElementById('typingIndicator');
const moodIndicator = document.getElementById('currentMood');

// Add a message to the chat window
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.innerHTML = content;
    
    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update the mood indicator in the header
function updateMoodIndicator(mood) {
    if (!mood) return;
    moodIndicator.textContent = `Current Mood: ${mood.charAt(0).toUpperCase() + mood.slice(1)}`;
}

// Create the HTML for the bot's response
function createBotResponseHTML(response) {
    // Handle emergency responses
    if (response.emergency) {
        let html = `<p>${response.message}</p>`;
        html += '<div class="emergency-alert">';
        html += '<h4>🚨 EMERGENCY RESOURCES</h4>';
        response.emergency_contacts.forEach(contact => {
            html += `<p style="margin: 5px 0;"><strong>${contact}</strong></p>`;
        });
        html += '</div>';
        return html;
    }

    // Handle regular responses
    let html = `<p>${response.recommendations.message}</p>`;
    html += `<div class="mood-badge mood-${response.mood}">
        Detected Mood: ${response.mood.charAt(0).toUpperCase() + response.mood.slice(1)} 
        (${Math.round(response.confidence * 100)}% confidence)
    </div>`;

    html += '<div class="recommendations">';
    const recs = response.recommendations;
    Object.keys(recs).forEach(key => {
        if (key !== 'message' && Array.isArray(recs[key])) {
            html += '<div class="rec-section">';
            html += `<div class="rec-title">${key.replace(/_/g, ' ').toUpperCase()}</div>`;
            html += '<ul class="rec-list">';
            recs[key].slice(0, 3).forEach(item => { // Show top 3 recommendations
                html += `<li>${item}</li>`;
            });
            html += '</ul></div>';
        }
    });

    if (recs.emergency_note) {
        html += `<div class="emergency-alert">${recs.emergency_note}</div>`;
    }
    html += '</div>';
    return html;
}

// Show/hide the typing indicator
function showTypingIndicator() { typingIndicator.style.display = 'block'; }
function hideTypingIndicator() { typingIndicator.style.display = 'none'; }

// Send message to the backend
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    // Display user's message
    addMessage(message, true);
    messageInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();

    try {
        // Send message to the Flask backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Display bot's response
        const responseHTML = createBotResponseHTML(data);
        addMessage(responseHTML, false);
        
        // Update mood indicator
        updateMoodIndicator(data.mood);

    } catch (error) {
        hideTypingIndicator();
        addMessage("<p>Sorry, something went wrong. Please try again later.</p>", false);
        console.error('Error sending message:', error);
    }
}

// Handle quick messages
function quickMessage(message) {
    messageInput.value = message;
    sendMessage();
}

// Handle 'Enter' key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Initial focus on the input field
document.addEventListener('DOMContentLoaded', () => {
    messageInput.focus();
});