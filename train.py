# train.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import re

print("Starting model training process...")


# --- UPGRADED: Use a more robust training data source ---
def load_real_training_data():
    """
    Placeholder function to load a real-world dataset.
    You would replace this with code to load a CSV, JSON, or similar file.
    For this example, we'll expand the synthetic data.
    """
    training_data = []

    # Expanded stressed examples
    stressed_texts = [
        "I have three exams tomorrow and I haven't studied enough",
        "The assignment deadline is today and I'm nowhere near done",
        "I feel so overwhelmed with all the coursework", "Can't handle the pressure anymore, too much work",
        "Stressed about my grades, parents will be disappointed", "Haven't slept in days because of studying",
        "Everything is piling up and I can't cope", "The workload is making me have headaches",
        "I'm burning out from all these assignments", "Feeling exhausted from studying all night",
        "My head hurts from all this stress.", "I have a huge project due next week.",
        "The professor assigned another paper.", "I am so done with all these deadlines."
    ]

    # Expanded anxious examples
    anxious_texts = [
        "I'm really nervous about the presentation tomorrow", "What if I fail the exam? I'm so worried",
        "My heart is racing thinking about the interview", "I can't stop worrying about my future",
        "Feeling anxious about meeting new people", "I'm scared I won't get into graduate school",
        "The thought of public speaking makes me panic", "I'm on edge about everything lately",
        "Can't shake this uneasy feeling", "Butterflies in my stomach about tomorrow",
        "I'm freaking out about the results.", "I have a terrible feeling about this.",
        "The uncertainty is making me so nervous.", "I'm afraid of messing this up."
    ]

    # ... (You would add more examples for all moods here) ...
    depressed_texts = [
        "I don't see the point in anything anymore", "Feeling so lonely even when surrounded by people",
        "Everything feels meaningless and dark", "I'm tired all the time and don't want to do anything",
        "Nothing makes me happy anymore", "I feel empty and worthless",
        "Don't care about grades or future", "Life feels heavy and difficult",
        "I'm isolated from everyone", "Feel like giving up on everything",
        "I just want to be left alone.", "It's so hard to get out of bed.",
        "I feel so numb lately.", "Nothing excites me anymore."
    ]

    neutral_texts = [
        "Just finished my homework for today", "Had a regular day at university", "Attending classes as usual",
        "Nothing special happening today", "Just working on my assignments", "Normal day, nothing to complain about",
        "Going through my daily routine", "Classes were okay today", "Just studying for upcoming tests",
        "Regular university life",
        "I ate lunch at 12 PM today.", "The weather is nice outside.",
        "I need to buy groceries later.", "I have a meeting at 3 PM."
    ]

    happy_texts = [
        "Got an A on my exam, I'm so happy!", "Had an amazing day with friends",
        "Feeling grateful for this opportunity",
        "I'm excited about my new project", "Everything is going well in my life", "Proud of what I accomplished today",
        "Feeling confident about my abilities", "Such a wonderful day at university", "I'm optimistic about the future",
        "Feeling energetic and motivated", "I aced my exam today!", "My friends and I are going out later.",
        "I feel so great today!", "It was a wonderful day."
    ]

    motivated_texts = [
        "Ready to tackle all my assignments", "Feeling inspired to study harder",
        "I can achieve my goals if I work hard",
        "Motivated to improve my grades", "Excited to learn new things today", "I'm determined to succeed",
        "Feeling productive and focused", "Ready to take on new challenges", "Enthusiastic about my studies",
        "I'm going to make today count", "I'm ready to learn anything.", "I am determined to do well.",
        "I'm excited to finish this project.", "I can do this!"
    ]

    for text in stressed_texts: training_data.append((text, 'stressed'))
    for text in anxious_texts: training_data.append((text, 'anxious'))
    for text in depressed_texts: training_data.append((text, 'depressed'))
    for text in neutral_texts: training_data.append((text, 'neutral'))
    for text in happy_texts: training_data.append((text, 'happy'))
    for text in motivated_texts: training_data.append((text, 'motivated'))

    return training_data


class MoodModelTrainer:
    def __init__(self):
        # We define the pipeline but will use GridSearchCV to find the best parameters
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def train(self):
        print("1. Loading training data...")
        training_data = load_real_training_data()

        texts = [self.preprocess_text(text) for text, _ in training_data]
        labels = [label for _, label in training_data]

        X_train, _, y_train, _ = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

        # --- UPGRADED: Hyperparameter Tuning with GridSearchCV ---
        print("2. Starting hyperparameter tuning...")
        param_grid = {
            'tfidf__max_features': [500, 1000, 2000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10, 100],
        }
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        print("\nBest Parameters found: ", grid_search.best_params_)
        print("Best Cross-Validation Score: {:.2f}".format(grid_search.best_score_))

        best_model = grid_search.best_estimator_
        print("Model training complete.")
        return best_model


# Main training execution
if __name__ == "__main__":
    trainer = MoodModelTrainer()
    trained_model = trainer.train()

    # Save the trained model to a file
    with open('mood_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    print("\n3. Model saved successfully as 'mood_model.pkl'")