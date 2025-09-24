# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import re
from datetime import datetime, timedelta
import uuid
import random

# --- NEW: Import libraries for Multimodality (Conceptual) ---
# NOTE: These libraries would need to be installed in your environment.
# from PIL import Image
# import io
# import requests # For API calls to a voice-to-text service or a multimodal model
# -----------------------------------------------------------------

# --- Initialize Flask App ---
app = Flask(__name__)

# --- In-memory user session state ---
user_sessions = {}

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


# --- NEW: Conceptual Image Analysis Function ---
def analyze_image(image_data):
    """
    (CONCEPTUAL) Analyzes an image for relevant context (e.g., a cluttered desk).
    This would use a pre-trained image analysis model or API.
    """
    # Placeholder logic
    # try:
    #     image = Image.open(io.BytesIO(image_data))
    #     # Here, you would call a multimodal model API (e.g., a local model or a cloud service)
    #     # to describe the image.
    #     description = "A messy desk with many books and papers."
    #     if "messy desk" in description or "cluttered room" in description:
    #         return "cluttered"
    #     return "neutral_image"
    # except Exception as e:
    #     print(f"Error analyzing image: {e}")
    #     return "error"
    return "cluttered"  # For demonstration purposes


# --- NEW: Conceptual Voice Analysis Function ---
def analyze_voice(voice_data):
    """
    (CONCEPTUAL) Analyzes a voice for sentiment and transcribes it.
    This would use a voice-to-text service and a sentiment analysis model.
    """
    # Placeholder logic
    # try:
    #     # Send voice_data to a voice-to-text API
    #     transcribed_text = "I'm feeling so stressed about my finals."
    #     # Analyze the sentiment of the transcribed text
    #     sentiment = "stressed"
    #     return transcribed_text, sentiment
    # except Exception as e:
    #     print(f"Error analyzing voice: {e}")
    #     return "", "error"
    return "I am feeling so stressed.", "stressed"  # For demonstration purposes


class ResourceProvider:
    """Provides resources and recommendations based on mood and conversational context"""

    def __init__(self):
        self.resources = {
            'stressed': {
                "message": "I understand you're feeling stressed. It's completely normal, especially as a student. Here are some ways to help you cope:",
                'immediate_help': ["Take 5 deep breaths slowly", "Try the 5-4-3-2-1 grounding technique",
                                   "Take a 10-minute walk"],
                'study_tips': ["Break large tasks into smaller chunks", "Use the Pomodoro Technique",
                               "Prioritize tasks by urgency"],
            },
            'anxious': {
                "message": "I can see you're feeling anxious. Anxiety is very common, and there are effective ways to manage it:",
                'immediate_help': ["Practice 4-7-8 breathing", "Challenge negative thoughts with positive self-talk",
                                   "Visualize a calm, peaceful place"],
                'coping_strategies': ["Identify and challenge anxious thought patterns",
                                      "Prepare thoroughly for events that make you anxious", "Keep a worry journal"],
            },
            'depressed': {
                "message": "I'm concerned about how you're feeling. Depression is serious, but it's treatable and you don't have to face it alone:",
                'immediate_support': ["Reach out to a trusted friend or family member today",
                                      "Engage in one small activity you usually enjoy",
                                      "Get some sunlight or fresh air"],
                'professional_help': ["Contact your campus counseling center immediately",
                                      "Look into mental health professionals",
                                      "Use crisis intervention services if needed"],
                'emergency_note': "If you're having thoughts of self-harm, please call 988 (National Suicide Prevention Lifeline) or go to your nearest emergency room immediately."
            },
            'happy': {
                "message": "It's wonderful that you're feeling happy! Let's talk about how to maintain and share this positive energy:",
                'maintain_positivity': ["Share your positive energy with friends", "Practice gratitude",
                                        "Celebrate your achievements"],
            },
            'motivated': {
                "message": "I love seeing your motivation! Let's harness this energy effectively:",
                'maximize_productivity': ["Create a detailed action plan for your goals",
                                          "Break large projects into manageable steps",
                                          "Track your progress regularly"],
            },
            'neutral': {
                "message": "You seem to be in a balanced state. This is a great time for maintenance and prevention:",
                'maintenance': ["Keep up with your current healthy routines", "Maintain your social connections",
                                "Practice daily gratitude"],
            },
            # --- NEW: Image-specific recommendations ---
            'cluttered': {
                "message": "I see you're dealing with a cluttered space. Sometimes a little organization can help clear the mind!",
                "tips": ["Break down the decluttering process into small steps.",
                         "Start with one small area at a time.", "Put on some motivating music while you work."],
            }
        }

        self.proactive_messages = [
            "Hi there! Just checking in. How are you feeling today?",
            "Hey! It's been a while. How can I help you right now?",
            "How's your day going? I'm here if you need to talk.",
            "I'm here to listen, whenever you're ready.",
            "Just wanted to say hi. Hope you're having a good day.",
        ]

        self.general_talks = [
            "Sometimes the first step is the hardest. Just sharing what’s on your mind can make a big difference.",
            "It's completely okay to not have all the answers. Let's just talk it through.",
            "You know, a lot of students feel this way. You're not alone in this.",
            "Wait, the more you think, the worse it gets. Let's try to focus on one thing at a time.",
            "It's like a mental traffic jam. Let's try to clear a path, one thought at a time.",
            "Remember, your feelings are valid. What you're going through is real."
        ]

        self.instruction_responses = {
            "schedule": "I can't create a real-time schedule, but here's a simple one to help you manage: 1. Take a 5-minute break. 2. Do one small task. 3. Take another break. This helps prevent burnout.",
        }

    def get_recommendations(self, mood):
        return self.resources.get(mood, self.resources['neutral'])

    def get_proactive_message(self):
        return random.choice(self.proactive_messages)

    def get_general_talk(self):
        return random.choice(self.general_talks)

    def get_instruction_response(self, instruction_key):
        return self.instruction_responses.get(instruction_key)


class StudentCounselingChatbot:
    """Main chatbot class that integrates all components and manages state"""

    def __init__(self, model):
        self.model = model
        self.resource_provider = ResourceProvider()

    def analyze_input(self, user_input, user_id):
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

        processed_text = preprocess_text(user_input)

        # Check for instruction-based queries
        if 'schedule' in processed_text or 'create a schedule' in processed_text:
            return {
                'emergency': False,
                'mood': 'instruction',
                'confidence': 1.0,
                'recommendations': {
                    'message': self.resource_provider.get_instruction_response("schedule")
                }
            }

        # Check for previous stressed state
        last_mood_data = user_sessions.get(user_id)
        current_time = datetime.now()

        STRESS_PERSISTENCE_MINS = 30

        if last_mood_data and last_mood_data['mood'] == 'stressed' and \
                (current_time - last_mood_data['timestamp']).total_seconds() / 60 < STRESS_PERSISTENCE_MINS:

            mood = 'stressed'
            confidence = last_mood_data['confidence']
        else:
            mood = self.model.predict([processed_text])[0]
            confidence = max(self.model.predict_proba([processed_text])[0])

        # Add general talk for non-stressed, non-emergency messages for a more human feel
        if mood not in ['stressed', 'depressed', 'anxious'] and confidence < 0.8:
            response_message = f"{self.resource_provider.get_general_talk()} {self.resource_provider.get_recommendations(mood)['message']}"
            recommendations = self.resource_provider.get_recommendations(mood)
            recommendations['message'] = response_message
        else:
            recommendations = self.resource_provider.get_recommendations(mood)

        user_sessions[user_id] = {
            'mood': mood,
            'confidence': confidence,
            'timestamp': current_time
        }

        response = {
            'emergency': False,
            'mood': mood,
            'confidence': round(confidence, 2),
            'recommendations': recommendations
        }
        return response

    def get_proactive_response(self, user_id):
        return {
            'emergency': False,
            'mood': 'proactive',
            'confidence': 1.0,
            'recommendations': {
                'message': self.resource_provider.get_proactive_message()
            }
        }


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

    user_data = request.json
    user_input = user_data.get("message")
    user_id = user_data.get("user_id")

    if not user_id:
        user_id = str(uuid.uuid4())
        response = chatbot.get_proactive_response(user_id)
        response['user_id'] = user_id
        return jsonify(response)

    last_interaction = user_sessions.get(user_id)
    if last_interaction and user_input == '':
        time_since_last_message_mins = (datetime.now() - last_interaction['timestamp']).total_seconds() / 60
        PROACTIVE_CHECKIN_THRESHOLD_MINS = 60

        if time_since_last_message_mins > PROACTIVE_CHECKIN_THRESHOLD_MINS:
            response = chatbot.get_proactive_response(user_id)
            response['user_id'] = user_id
            return jsonify(response)

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = chatbot.analyze_input(user_input, user_id)
    response['user_id'] = user_id
    return jsonify(response)


# --- NEW: API Endpoint for Image-based Analysis (Conceptual) ---
@app.route("/chat/image", methods=['POST'])
def chat_image_endpoint():
    """
    (CONCEPTUAL) Handles image messages.
    This would accept an image file and use a conceptual model to analyze it.
    """
    # if not chatbot:
    #     return jsonify({"error": "Chatbot is not initialized."}), 500

    # user_id = request.form.get("user_id")
    # image_file = request.files.get("image")

    # if not user_id or not image_file:
    #     return jsonify({"error": "Missing user ID or image file"}), 400

    # image_data = image_file.read()
    # image_analysis_result = analyze_image(image_data)

    # if image_analysis_result == "cluttered":
    #     recommendations = chatbot.resource_provider.resources['cluttered']
    #     return jsonify({
    #         "mood": "cluttered",
    #         "recommendations": recommendations,
    #         "user_id": user_id
    #     })
    # else:
    #     return jsonify({"error": "Could not analyze image or no relevant context found."}), 400
    return jsonify({"error": "Image analysis feature is not implemented in this version."}), 501


# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)