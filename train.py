# train.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re

print("Starting model training process...")

# This class contains the logic for creating data and training the model.
class MoodModelTrainer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # PASTE THIS CODE INTO train.py, REPLACING THE OLD create_training_data METHOD

    def create_training_data(self):
        """Create synthetic training data for mood detection"""
        training_data = []
        
        # Stressed examples (10 total)
        stressed_texts = [
            "I have three exams tomorrow and I haven't studied enough", "The assignment deadline is today and I'm nowhere near done",
            "I feel so overwhelmed with all the coursework", "Can't handle the pressure anymore, too much work",
            "Stressed about my grades, parents will be disappointed", "Haven't slept in days because of studying",
            "Everything is piling up and I can't cope", "The workload is making me have headaches",
            "I'm burning out from all these assignments", "Feeling exhausted from studying all night"
        ]
        
        # Anxious examples (10 total)
        anxious_texts = [
            "I'm really nervous about the presentation tomorrow", "What if I fail the exam? I'm so worried",
            "My heart is racing thinking about the interview", "I can't stop worrying about my future",
            "Feeling anxious about meeting new people", "I'm scared I won't get into graduate school",
            "The thought of public speaking makes me panic", "I'm on edge about everything lately",
            "Can't shake this uneasy feeling", "Butterflies in my stomach about tomorrow"
        ]
        
        # Depressed examples (10 total)
        depressed_texts = [
            "I don't see the point in anything anymore", "Feeling so lonely even when surrounded by people",
            "Everything feels meaningless and dark", "I'm tired all the time and don't want to do anything",
            "Nothing makes me happy anymore", "I feel empty and worthless",
            "Don't care about grades or future", "Life feels heavy and difficult",
            "I'm isolated from everyone", "Feel like giving up on everything"
        ]
        
        # Neutral examples (10 total)
        neutral_texts = [
            "Just finished my homework for today", "Had a regular day at university", "Attending classes as usual",
            "Nothing special happening today", "Just working on my assignments", "Normal day, nothing to complain about",
            "Going through my daily routine", "Classes were okay today", "Just studying for upcoming tests", "Regular university life"
        ]
        
        # Happy examples (10 total)
        happy_texts = [
            "Got an A on my exam, I'm so happy!", "Had an amazing day with friends", "Feeling grateful for this opportunity",
            "I'm excited about my new project", "Everything is going well in my life", "Proud of what I accomplished today",
            "Feeling confident about my abilities", "Such a wonderful day at university", "I'm optimistic about the future",
            "Feeling energetic and motivated"
        ]
        
        # Motivated examples (10 total)
        motivated_texts = [
            "Ready to tackle all my assignments", "Feeling inspired to study harder", "I can achieve my goals if I work hard",
            "Motivated to improve my grades", "Excited to learn new things today", "I'm determined to succeed",
            "Feeling productive and focused", "Ready to take on new challenges", "Enthusiastic about my studies",
            "I'm going to make today count"
        ]
        
        # Combine all data
        for text in stressed_texts: training_data.append((text, 'stressed'))
        for text in anxious_texts: training_data.append((text, 'anxious'))
        for text in depressed_texts: training_data.append((text, 'depressed'))
        for text in neutral_texts: training_data.append((text, 'neutral'))
        for text in happy_texts: training_data.append((text, 'happy'))
        for text in motivated_texts: training_data.append((text, 'motivated'))
        return training_data
    def train(self):
        print("1. Creating training data...")
        training_data = self.create_training_data()
        
        texts = [self.preprocess_text(text) for text, _ in training_data]
        labels = [label for _, label in training_data]
        
        X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

        print("2. Training the model...")
        self.model.fit(X_train, y_train)
        print("Model training complete.")
        return self.model

# Main training execution
if __name__ == "__main__":
    trainer = MoodModelTrainer()
    trained_model = trainer.train()

    # Save the trained model to a file
    with open('mood_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    print("3. Model saved successfully as 'mood_model.pkl'")