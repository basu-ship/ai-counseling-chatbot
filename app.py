# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import re
from datetime import datetime

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load the Trained Model ---
try:
    with open('mood_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Mood detection model loaded successfully.")
except FileNotFoundError:
    print("Error: 'mood_model.pkl' not found. Please run train.py first to create the model.")
    model = None

# --- Helper Classes and Functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class ResourceProvider:
    """Provides resources and recommendations based on mood"""
    def __init__(self):
        # A simplified version of your resource dictionary
        self.resources = {
            'stressed': {
                "message": "I understand you're feeling stressed. It's completely normal, especially as a student. Here are some ways to help you cope:",
                'immediate_help': ["Take 5 deep breaths slowly", "Try the 5-4-3-2-1 grounding technique", "Take a 10-minute walk"],
                'study_tips': ["Break large tasks into smaller chunks", "Use the Pomodoro Technique", "Prioritize tasks by urgency"],
            },
            'anxious': {
                "message": "I can see you're feeling anxious. Anxiety is very common, and there are effective ways to manage it:",
                'immediate_help': ["Practice 4-7-8 breathing", "Challenge negative thoughts with positive self-talk", "Visualize a calm, peaceful place"],
                'coping_strategies': ["Identify and challenge anxious thought patterns", "Prepare thoroughly for events that make you anxious", "Keep a worry journal"],
            },
            'depressed': {
                "message": "I'm concerned about how you're feeling. Depression is serious, but it's treatable and you don't have to face it alone:",
                'immediate_support': ["Reach out to a trusted friend or family member today", "Engage in one small activity you usually enjoy", "Get some sunlight or fresh air"],
                'professional_help': ["Contact your campus counseling center immediately", "Look into mental health professionals", "Use crisis intervention services if needed"],
                'emergency_note': "If you're having thoughts of self-harm, please call 988 (National Suicide Prevention Lifeline) or go to your nearest emergency room immediately."
            },
            'happy': {
                "message": "It's wonderful that you're feeling happy! Let's talk about how to maintain and share this positive energy:",
                'maintain_positivity': ["Share your positive energy with friends", "Practice gratitude", "Celebrate your achievements"],
            },
            'motivated': {
                "message": "I love seeing your motivation! Let's harness this energy effectively:",
                'maximize_productivity': ["Create a detailed action plan for your goals", "Break large projects into manageable steps", "Track your progress regularly"],
            },
            'neutral': {
                "message": "You seem to be in a balanced state. This is a great time for maintenance and prevention:",
                'maintenance': ["Keep up with your current healthy routines", "Maintain your social connections", "Practice daily gratitude"],
            }
        }

    def get_recommendations(self, mood):
        return self.resources.get(mood, self.resources['neutral'])

class StudentCounselingChatbot:
    """Main chatbot class that integrates all components"""
    def __init__(self, model):
        self.model = model
        self.resource_provider = ResourceProvider()
        
    def analyze_input(self, user_input):
        # Emergency keyword check
        emergency_keywords = ['suicide', 'kill myself', 'end my life', 'want to die', 'harm myself', 'hurt myself']
        if any(keyword in user_input.lower() for keyword in emergency_keywords):
            return {
                'emergency': True,
                'message': "I'm very concerned about you. Please reach out for immediate help.",
                'emergency_contacts': [
                    "Call 988 (National Suicide Prevention Lifeline) right now",
                    "Go to your nearest emergency room",
                    "Text 'HELLO' to 741741 for the Crisis Text Line"
                ]
            }

        # Predict mood using the loaded ML model
        processed_text = preprocess_text(user_input)
        mood = self.model.predict([processed_text])[0]
        confidence = max(self.model.predict_proba([processed_text])[0])
        
        recommendations = self.resource_provider.get_recommendations(mood)
        
        response = {
            'emergency': False,
            'mood': mood,
            'confidence': round(confidence, 2),
            'recommendations': recommendations
        }
        return response

# --- Initialize Chatbot ---
if model:
    chatbot = StudentCounselingChatbot(model)
else:
    chatbot = None

# --- Define API Routes ---
@app.route("/")
def home():
    """Render the chat page."""
    return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat_endpoint():
    """Handle chat messages."""
    if not chatbot:
        return jsonify({"error": "Chatbot is not initialized. Please train the model."}), 500

    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    response = chatbot.analyze_input(user_input)
    return jsonify(response)

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)