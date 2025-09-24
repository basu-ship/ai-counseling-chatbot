### 🧠 AI Student Counseling Chatbot README

This repository contains a simple AI-powered counseling chatbot designed to assist students with common mental health concerns such as stress, anxiety, and depression. The system detects the user's mood from their text input and provides tailored resources and coping strategies.

-----

### ⚙️ Workflow and Project Context

The project is structured into two main components:

1.  **Model Training (`train.py`):** This script is responsible for creating and training a machine learning model for mood detection.

      * It uses a `MoodModelTrainer` class to generate synthetic training data for six different moods: stressed, anxious, depressed, happy, motivated, and neutral.
      * The model uses a **TfidfVectorizer** to convert text into numerical features and a **Logistic Regression** classifier to predict the mood based on those features.
      * After training, the script saves the final model as a pickled file named `mood_model.pkl`.

2.  **Web Application (`app.py`):** This is a Flask web application that serves as the chatbot's front end and back end.

      * It loads the pre-trained `mood_model.pkl` file to enable mood prediction.
      * The application includes a `StudentCounselingChatbot` class that handles incoming messages.
      * It first checks for emergency keywords (e.g., 'suicide', 'harm myself') to provide immediate, critical resources if needed.
      * For other messages, it uses the loaded model to predict the user's mood and confidence level.
      * A `ResourceProvider` class then fetches a set of pre-defined recommendations based on the detected mood.
      * The chatbot's responses and the front-end user interface are managed by JavaScript (`script.js`), CSS (`style.css`), and HTML (`index.html`).

-----

### 🚀 How to Use

To set up and run the AI Student Counseling Chatbot, follow these steps:

1.  **Clone the repository.**
2.  **Set up the environment:** It is highly recommended to use a virtual environment.
3.  **Install dependencies:** Navigate to the project directory and install all required libraries using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the model:** Run the `train.py` script to create the machine learning model. This step is mandatory before running the application, as `app.py` requires the `mood_model.pkl` file.
    ```bash
    python train.py
    ```
    You will see confirmation messages indicating that the model is being trained and saved.
5.  **Run the application:** Once the model is trained, start the Flask application.
    ```bash
    python app.py
    ```
6.  **Access the chatbot:** Open a web browser and go to the URL provided by Flask (typically `http://127.0.0.1:5000/`).

-----

### 📂 Repository Contents

  * `app.py`: The main Flask application that runs the chatbot.
  * `train.py`: The script to train and save the mood detection model.
  * `mood_model.pkl`: The pre-trained model file created by `train.py`.
  * `requirements.txt`: A list of all Python libraries required for the project.
  * `templates/`: Contains `index.html`, the front-end HTML for the chat interface.
  * `static/`: Contains the `js` and `css` directories for front-end assets.
      * `js/script.js`: Handles client-side chat logic.
      * `css/style.css`: Defines the visual styles of the chatbot interface.
