# train.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
import re

print("Starting model training process...")


# This class contains the logic for creating data and training the model.
class MoodModelTrainer:
    def __init__(self):
        # Initializing a general pipeline. Optimal parameters will be found via GridSearchCV.
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Define the parameter grid for comprehensive hyperparameter tuning
        self.param_grid = {
            'tfidf__max_features': [1000, 2000, 4000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.5, 1, 10],
        }

    def preprocess_text(self, text):
        """
        Refined preprocessing to correctly handle both English and Hindi text.
        It keeps English letters (a-z) AND Devanagari characters (\u0900-\u097F).
        """
        text = str(text).lower()
        # Regex keeps English, Hindi, and spaces. Removes all other symbols/numbers/junk.
        text = re.sub(r'[^a-zA-Z\s\u0900-\u097F]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_synthetic_data(self):
        """Load the original small synthetic dataset."""
        training_data = []

        # Stressed examples
        stressed_texts = ["I have three exams tomorrow and I haven't studied enough",
                          "The assignment deadline is today and I'm nowhere near done",
                          "I feel so overwhelmed with all the coursework",
                          "Can't handle the pressure anymore, too much work",
                          "I'm burning out from all these assignments"]
        # Anxious examples
        anxious_texts = ["I'm really nervous about the presentation tomorrow",
                         "What if I fail the exam? I'm so worried", "My heart is racing thinking about the interview",
                         "I can't stop worrying about my future", "Feeling anxious about meeting new people"]
        # Depressed examples
        depressed_texts = ["I don't see the point in anything anymore",
                           "Feeling so lonely even when surrounded by people", "Everything feels meaningless and dark",
                           "I'm tired all the time and don't want to do anything", "Nothing makes me happy anymore"]
        # Neutral examples
        neutral_texts = ["Just finished my homework for today", "Had a regular day at university",
                         "Attending classes as usual", "Nothing special happening today",
                         "Just working on my assignments"]
        # Happy examples
        happy_texts = ["Got an A on my exam, I'm so happy!", "Had an amazing day with friends",
                       "Feeling grateful for this opportunity", "I'm excited about my new project",
                       "Everything is going well in my life"]
        # Motivated examples
        motivated_texts = ["Ready to tackle all my assignments", "Feeling inspired to study harder",
                           "I can achieve my goals if I work hard", "Motivated to improve my grades",
                           "Excited to learn new things today"]

        # Combine data (5 samples per original category for demonstration)
        for text in stressed_texts: training_data.append((text, 'stressed'))
        for text in anxious_texts: training_data.append((text, 'anxious'))
        for text in depressed_texts: training_data.append((text, 'depressed'))
        for text in neutral_texts: training_data.append((text, 'neutral'))
        for text in happy_texts: training_data.append((text, 'happy'))
        for text in motivated_texts: training_data.append((text, 'motivated'))

        return training_data

    def load_csv_data(self):
        """Loads and processes data from the uploaded stress_phrases_dataset_2000.csv file."""
        try:
            # Using 'latin-1' encoding as a safe fallback for mixed-language files
            df = pd.read_csv('mood_dataset_2000.csv', encoding='latin-1')
        except FileNotFoundError:
            print("ERROR: 'stress_phrases_dataset_2000.csv' not found. Skipping CSV data.")
            return []
        except Exception as e:
            print(f"ERROR reading CSV file: {e}. Skipping CSV data.")
            return []

        # FIX 1: Strip whitespace from labels to prevent inconsistent data types (e.g., 'Medium ' vs 'Medium')
        if 'perceived_stress_level' in df.columns:
            df['perceived_stress_level'] = df['perceived_stress_level'].str.strip()

        # --- CONSOLIDATED MAPPING STRATEGY ---
        mood_mapping = {
            'High': 'stressed',
            'Medium': 'stressed',
            'Low': 'neutral',
            'Ambiguous': 'neutral',
            'None': 'neutral'
        }

        # Apply the mapping
        df['perceived_mood'] = df['perceived_stress_level'].map(mood_mapping)

        # Filter out any rows where input_text might be missing (NaN)
        df.dropna(subset=['input_text', 'perceived_mood'], inplace=True)

        csv_data = list(zip(df['input_text'].tolist(), df['perceived_mood'].tolist()))
        print(f"Loaded {len(csv_data)} samples from CSV.")
        return csv_data

    def create_training_data(self):
        """Combines data from both synthetic and CSV sources."""
        synthetic_data = self.load_synthetic_data()
        csv_data = self.load_csv_data()

        combined_data = synthetic_data + csv_data

        texts = [item[0] for item in combined_data]
        labels = [item[1] for item in combined_data]

        print(f"Total combined samples for training: {len(combined_data)}")
        return texts, labels

    def train(self):
        print("1. Loading and preparing training data...")
        texts, labels = self.create_training_data()

        if not texts:
            print("Training aborted due to missing data.")
            return None

        # Preprocess all text data
        preprocessed_texts = [self.preprocess_text(text) for text in texts]

        # Split data for training
        X_train, _, y_train, _ = train_test_split(
            preprocessed_texts, labels, test_size=0.1, random_state=42, stratify=labels
        )

        # --- ENHANCED: Hyperparameter Tuning with GridSearchCV ---
        print("2. Starting comprehensive hyperparameter tuning...")

        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )

        # This is the explicit step that runs the training
        grid_search.fit(X_train, y_train)

        print("\n" + "-" * 50)
        print("Model training complete. Best Model Found:")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print("-" * 50)

        # Return the best model found by the tuning process
        best_model = grid_search.best_estimator_
        return best_model


# Main training execution
if __name__ == "__main__":
    trainer = MoodModelTrainer()
    trained_model = trainer.train()

    if trained_model:
        # Save the trained model to a file
        with open('mood_model.pkl', 'wb') as f:
            pickle.dump(trained_model, f)
        print("3. Model saved successfully as 'mood_model.pkl'")
    else:
        print("3. Model saving skipped.")