from flask import Flask, request, jsonify
import spacy
import nltk
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator  # Use Deep Translator

# Initialize Flask app
app = Flask(__name__)

# Load NLP model
nlp = spacy.load("en_core_web_sm")  # Use "en_core_sci_sm" for medical NLP

# Load stopwords (ensure this is installed in requirements.txt)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to transliterate Hinglish to English
def transliterate_text(text):
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except:
        return text  # Return original text if translation fails

# Function to clean text and extract symptoms
def extract_symptoms(text):
    # Convert Hinglish to English
    text = transliterate_text(text)

    # Process text with NLP
    doc = nlp(text)

    # Remove stopwords & extract meaningful words
    filtered_words = [token.text for token in doc if token.text.lower() not in stop_words]

    # Extract possible symptoms (NER - Named Entity Recognition)
    symptoms = [token.text for token in doc if token.ent_type_ in ["DISEASE", "SYMPTOM"]]

    return symptoms if symptoms else filtered_words  # If no symptoms detected, return filtered words

# API endpoint for symptom extraction
@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    symptoms = extract_symptoms(user_text)
    return jsonify({"symptoms": symptoms})

# Health check endpoint
@app.route('/')
def home():
    return "✅ Symptom Extractor API is running!", 200

# Run locally
if __name__ == '__main__':
    app.run(debug=True)
