import nltk
import random
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from PyPDF2 import PdfReader, PdfWriter
import os
import google.generativeai as genai

# Download required NLTK resources
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

app = Flask(__name__)
responses = defaultdict(list)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAMUyC5Hktnr28Gt64ZBKy6x1aKFT3tYcU"  # Replace with your actual key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')  # Initialize model globally

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    if not text:
        return "", []
    words = word_tokenize(text.lower())
    processed_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    key_words = processed_words
    return " ".join(processed_words), key_words

def load_dataset(filename):
    """Loads Q&A from CSV and merges with existing responses."""
    global responses
    try:
        dataset = pd.read_csv(filename)
        dataset.columns = ["Pattern", "Response"]
        for _, row in dataset.iterrows():
            processed_pattern, key_words = preprocess_text(row['Pattern'])
            responses[processed_pattern].append(row['Response'])
            for word in key_words:
                responses[word].append(row['Response'])
        print("Dataset loaded successfully from", filename)
    except Exception as e:
        print("Error loading dataset:", e)

def get_gemini_response(user_input):
    """Fetches a concise response from Gemini API."""
    try:
        prompt = f"Answer the following question or statement in a concise, readable way (max 200 characters):\n\n{user_input}"
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Truncate to 200 characters if needed
        if len(text) > 200:
            text = text[:197] + "..."
        return text
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return "Sorry, I couldn’t process that right now!"

def get_best_response(user_input):
    """Finds the best response from dataset or Gemini API, keeping it concise."""
    processed_input, key_words = preprocess_text(user_input)
    
    # Check dataset for full match first
    if processed_input in responses and responses[processed_input]:
        response = random.choice(responses[processed_input])
        # Truncate dataset response if too long
        if len(response) > 200:
            return response[:197] + "..."
        return response
    
    # Check dataset for keyword match
    for word in key_words:
        if word in responses and responses[word]:
            response = random.choice(responses[word])
            if len(response) > 200:
                return response[:197] + "..."
            return response
    
    # Fallback to Gemini API
    return get_gemini_response(user_input)

def compress_pdf(input_path, output_path):
    """Compresses a PDF file by rewriting it with basic optimization."""
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
        print(f"Error compressing PDF: {str(e)}")
        return False

def summarize_content(file_path):
    """Summarizes content concisely using Gemini API."""
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            text = df.to_string()
        else:
            return ""
        
        # Limit summary to 200 characters
        prompt = f"Summarize the following content in a concise, readable way (max 200 characters):\n\n{text}"
        response = model.generate_content(prompt)
        summary = response.text.strip()
        if len(summary) > 200:
            summary = summary[:197] + "..."
        return summary
    except Exception as e:
        print(f"Error summarizing content: {str(e)}")
        return f"Failed to summarize: {str(e)}"[:200]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles file upload, summarizes content, and prepares compressed PDF for download."""
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
        return jsonify({"message": f"Upload failed: {str(e)}"})

@app.route("/download", methods=["GET"])
def download_file():
    """Serves the compressed PDF for download."""
    compressed_path = "compressed_file.pdf"
    if os.path.exists(compressed_path):
        return send_file(compressed_path, as_attachment=True, download_name="compressed_file.pdf")
    return jsonify({"message": "No compressed file available for download."})

@app.route("/chat", methods=["POST"])
def chat():
    """Responds to user messages as plain text."""
    try:
        if "message" not in request.form:
            return "No message received"
        
        user_input = request.form["message"].strip().lower()
        response = get_best_response(user_input)
        return response
    except Exception as e:
        return f"Chat processing failed: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)