# Chatbot

## Project Overview

Chatbot is a Flask-based web application that provides a conversational interface, file summarization, and PDF compression capabilities. It uses the Google Gemini API for generating responses and summaries, NLTK for text preprocessing, and PyPDF2 for handling PDFs. The application features a modern, animated web interface built with HTML, CSS, and JavaScript.

### Features
- **Conversational Chat**: Responds to user queries using a local dataset or the Gemini API, with responses limited to 200 characters for concise, readable output.
- **File Upload and Summarization**: Supports uploading CSV files (to load Q&A datasets) and PDF files (for summarization), with summaries capped at 200 characters.
- **PDF Compression**: Compresses uploaded PDFs and allows users to download the optimized versions.
- **User Interface**: A visually appealing design with gradient backgrounds, animated message transitions, and a responsive layout.

## Requirements

To run this project, you’ll need the following:

### Software
- **Python**: Version 3.9 or higher (tested with 3.11).
- **Git**: Optional, for cloning the repository.
- **Web Browser**: For accessing the web interface (e.g., Chrome, Firefox).

### Dependencies
Listed in `requirements.txt`:
Flask==2.3.3
nltk==3.8.1
pandas==2.2.2
PyPDF2==3.0.1
google-generativeai==0.7.2
gunicorn==22.0.0

### Additional Setup
- **Gemini API Key**: Obtain an API key from Google and update it in `app.py`.
- **Static Files**: A `static/bot-icon.png` file (e.g., a 40x40 pixel image) for the chat interface.

## Instructions to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/chatbot.git
cd chatbot