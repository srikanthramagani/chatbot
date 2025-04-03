import nltk
import random
import pandas as pd
import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, send_file
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from PyPDF2 import PdfReader, PdfWriter

# Secure your API key (REPLACE THIS in production)
GEMINI_API_KEY = "AIzaSyAMUyC5Hktnr28Gt64ZBKy6x1aKFT3tYcU"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Flask app
app = Flask(__name__)

# Set NLTK data path to a local directory (for pre-downloaded resources)
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))

# Ensure required NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
responses = defaultdict(list)

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    if not text:
        return "", []
    words = word_tokenize(text.lower())
    processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(processed_words), processed_words

def load_dataset(filename):
    """Loads Q&A from CSV into a dictionary."""
    try:
        dataset = pd.read_csv(filename)
        dataset.columns = ["Pattern", "Response"]
        for _, row in dataset.iterrows():
            processed_pattern, key_words = preprocess_text(row['Pattern'])
            responses[processed_pattern].append(row['Response'])
            for word in key_words:
                responses[word].append(row['Response'])
        print("✅ Dataset loaded successfully from", filename)
    except Exception as e:
        print("⚠️ Error loading dataset:", e)

def get_gemini_response(user_input):
    """Fetches a response from Gemini API."""
    try:
        response = model.generate_content(user_input)
        return response.text.strip()[:200] + "..." if len(response.text) > 200 else response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini API error: {str(e)}")
        return "Sorry, I couldn’t process that right now!"

def get_best_response(user_input):
    """Finds the best response from dataset or Gemini API."""
    processed_input, key_words = preprocess_text(user_input)
    
    # Check dataset for full match
    if processed_input in responses and responses[processed_input]:
        return random.choice(responses[processed_input])[:200] + "..." if len(responses[processed_input][0]) > 200 else random.choice(responses[processed_input])
    
    # Check dataset for keyword match
    for word in key_words:
        if word in responses and responses[word]:
            return random.choice(responses[word])[:200] + "..." if len(responses[word][0]) > 200 else random.choice(responses[word])
    
    # Fallback to Gemini API
    return get_gemini_response(user_input)

def compress_pdf(input_path, output_path):
    """Compresses a PDF file."""
    try:
        reader = PdfReader(input_path)
        writer = PdfWriter()

        for page in reader.pages:
            page.compress_content_streams()
            writer.add_page(page)

        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        return True
    except Exception as e:
        print(f"⚠️ Error compressing PDF: {str(e)}")
        return False

def summarize_content(file_path):
    """Summarizes content using Gemini API."""
    try:
        text = ""
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            text = df.to_string()
        else:
            return "Unsupported file type."

        response = model.generate_content(f"Summarize this in 200 characters: {text}")
        return response.text.strip()[:200] + "..." if len(response.text) > 200 else response.text.strip()
    except Exception as e:
        print(f"⚠️ Error summarizing content: {str(e)}")
        return "Failed to summarize."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file uploads (PDF & CSV) and processes them."""
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file uploaded"})
        
        file = request.files['file']
        if file.filename == "":
            return jsonify({"message": "No file selected"})
        
        upload_path = "uploaded_file"
        compressed_path = "compressed_file.pdf"
        file.save(upload_path)
        
        if file.filename.endswith('.csv'):
            load_dataset(upload_path)
            summary = summarize_content(upload_path)
            return jsonify({"message": f"✅ CSV uploaded successfully! Summary: {summary}"})
        elif file.filename.endswith('.pdf'):
            summary = summarize_content(upload_path)
            if compress_pdf(upload_path, compressed_path):
                return jsonify({
                    "message": f"✅ PDF uploaded and compressed! Summary: {summary}",
                    "download_available": True
                })
            else:
                return jsonify({"message": f"✅ PDF uploaded but compression failed. Summary: {summary}"})
        else:
            return jsonify({"message": "Unsupported file type. Please upload a CSV or PDF."})
    except Exception as e:
        return jsonify({"message": f"⚠️ Upload failed: {str(e)}"})

@app.route("/download", methods=["GET"])
def download_file():
    """Provides a compressed PDF for download."""
    compressed_path = "compressed_file.pdf"
    if os.path.exists(compressed_path):
        return send_file(compressed_path, as_attachment=True, download_name="compressed_file.pdf")
    return jsonify({"message": "No compressed file available for download."})

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot queries."""
    try:
        user_input = request.form.get("message", "").strip().lower()
        if not user_input:
            return "No message received"
        return get_best_response(user_input)
    except Exception as e:
        return f"⚠️ Chat processing failed: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
