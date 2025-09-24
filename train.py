# train.py
import pickle
import pandas as pd  # NEW: Import for efficient CSV reading
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

        # Define the parameter grid for thorough hyperparameter tuning
        self.param_grid = {
            'tfidf__max_features': [1000, 2000, 4000],  # Test different feature counts
            'tfidf__ngram_range': [(1, 1), (1, 2)],  # Test unigrams and bigrams
            'classifier__C': [0.5, 1, 10],  # Test different regularization strengths
        }

    def preprocess_text(self, text):
        """Preprocesses text for both English and Hindi phrases."""
        text = str(text).lower()
        # Keep letters (English a-z) and Devanagari characters (Hindi)
        text = re.sub(r'[^a-zA-Z\s\u0900-\u097F]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # UPGRADED METHOD: Reads training data from the uploaded CSV file
    def create_training_data(self):
        """Loads and prepares training data from the stress_phrases_dataset_2000.csv file"""
        try:
            # Assuming the CSV file is accessible from the current context
            df = pd.read_csv('stress_phrases_dataset_2000.csv')
        except FileNotFoundError:
            print("ERROR: 'stress_phrases_dataset_2000.csv' not found. Please ensure the file is correctly placed.")
            return [], []

        # Map 'None' stress level to 'neutral' for consistency with common ML labels
        df['perceived_stress_level'] = df['perceived_stress_level'].replace('None', 'neutral')

        texts = df['input_text'].tolist()
        labels = df['perceived_stress_level'].tolist()

        print(f"Loaded {len(texts)} samples for training.")
        return texts, labels

    def train(self):
        print("1. Loading and preparing training data...")
        texts, labels = self.create_training_data()

        if not texts:
            print("Training aborted due to missing data.")
            return None

        # Preprocess all text data
        preprocessed_texts = [self.preprocess_text(text) for text in texts]

        # Split data: 90% for training the best model, 10% held out (consistent with old project structure)
        # Stratify ensures all stress levels are represented proportionally in the split
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
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print("-" * 50)
        print("Model training complete.")
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print("-" * 50)

        # The best model found by GridSearchCV is selected
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