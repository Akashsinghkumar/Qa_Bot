import logging
import json
import requests
import sqlite3
import os
import speech_recognition as sr
from gtts import gTTS
from flask import Flask, request, render_template_string, jsonify, Response, redirect, url_for, session, flash, send_file
from concurrent.futures import ThreadPoolExecutor
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from typing import cast
from io import BytesIO
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import re

# --- Additional imports for PDF QA ---
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
CONFIG = {
    'MODEL_URL': "http://localhost:11434/api/generate",
    'MODEL_NAME': "gemma:2b",
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'THREAD_POOL_SIZE': 4,
    'TTS_LANGUAGE': 'en'
}

# --- App Setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key")
app.config['MAX_CONTENT_LENGTH'] = CONFIG['MAX_CONTENT_LENGTH']
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
executor = ThreadPoolExecutor(max_workers=CONFIG['THREAD_POOL_SIZE'])

# --- Logging ---
logging.basicConfig(level=logging.INFO)

# --- Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = cast(str, "login")

# --- Database Setup ---
DB_NAME = "users.db"
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
init_db()

# --- User Loader ---
class User(UserMixin):
    def __init__(self, id_, username):
        self.id = id_
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    with sqlite3.connect(DB_NAME) as conn:
        user = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(*user) if user else None

# --- QABot Class ---
class QABot:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        self.initialized = True
        try:
            logging.info("🔁 Warming up model...")
            self.ask_ollama("Hello")
            logging.info("✅ Model warmed up.")
        except Exception as e:
            logging.warning(f"⚠️ Warm-up failed: {e}")
        return True

    def ask_ollama(self, question):
        payload = {
            "model": CONFIG["MODEL_NAME"],
            "prompt": question,
            "stream": False,
            "options": {"num_predict": 100}
        }
        try:
            logging.info(f"🧠 Received question: {question}")
            response = requests.post(CONFIG["MODEL_URL"], json=payload)
            response.raise_for_status()
            data = response.json()
            logging.info(f"📦 Model responded: {data.get('response')}")
            answer = data.get("response", "No response received.")

            # Store the chat into DB only if user is logged in
            try:
                if current_user and hasattr(current_user, 'id'):
                    with sqlite3.connect(DB_NAME) as conn:
                        conn.execute("""
                            INSERT INTO questions (user_id, question, answer)
                            VALUES (?, ?, ?)""",
                            (current_user.id, question, answer)
                        )
            except Exception as db_error:
                logging.warning(f"⚠️ Could not save to database: {db_error}")

            return answer
        except Exception as e:
            logging.error(f"❌ Model error: {e}", exc_info=True)
            return "Sorry, something went wrong with the model."

qa_bot = QABot()

# --- PDF Storage ---
PDF_VECTORSTORES = {}  # Store per-user vectorstore

# --- Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_NAME) as conn:
            row = conn.execute("SELECT id, username, password FROM users WHERE username = ?", (username,)).fetchone()
            if row and check_password_hash(row[2], password):
                login_user(User(row[0], row[1]))
                return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                flash('Account created! Please login.')
                return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken.')
    return render_template_string(SIGNUP_TEMPLATE)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    if not qa_bot.initialized:
        qa_bot.initialize()
    return render_template_string(HTML_TEMPLATE, username=current_user.username)



# --- Ask Endpoint (Modified for PDF) ---
@app.route('/api/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    lang = data.get("lang", "en")

    if not question:
        return jsonify({"message": "No question provided."}), 400

    # Get raw answer from your QA Bot
    raw_answer = qa_bot.ask_ollama(question)

    # Create heading from the question (text before '?'), fallback to 'Answer'
    heading = question.split('?')[0].strip() or 'Answer'

    # Clean markdown-like syntax from the body to plain text
    def clean_markdown(md: str) -> str:
        if not md:
            return ""
        text = md
        # Remove code fences
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove images ![alt](url) -> alt
        text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", text)
        # Links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Bold/italic markers ** __ * _ `
        text = text.replace("**", "").replace("__", "").replace("`", "")
        text = text.replace("*", "").replace("_", "")
        # Strip leading list bullets at line starts
        lines = []
        for line in text.splitlines():
            line = re.sub(r"^\s*([\-\*•\u2022\u25CF\u25E6\u2043])\s+", "", line)
            lines.append(line)
        text = "\n".join(lines)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    body = clean_markdown(raw_answer.strip())

    return jsonify({"heading": heading, "body": body})




# --- PDF Upload Endpoint ---
@app.route('/api/upload_pdf', methods=['POST'])
@login_required
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    pdf_file = request.files['pdf']
    try:
        temp_path = f"temp_{current_user.id}.pdf"
        pdf_file.save(temp_path)

        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        if not documents:
            return jsonify({"error": "PDF is empty or cannot be read"}), 400

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        PDF_VECTORSTORES[current_user.id] = vectorstore
        os.remove(temp_path)
        return jsonify({"status": "PDF uploaded and processed successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- OCR Endpoint ---
@app.route('/api/ocr', methods=['POST'])
@login_required
def ocr_question():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400
    image = Image.open(request.files['image'])
    try:
        extracted_text = pytesseract.image_to_string(image)
        return jsonify({"question": extracted_text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Voice-to-Text Endpoint ---
@app.route('/api/voice', methods=['POST'])
@login_required
def voice_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']
    # Language hint from client ("en" | "hi")
    lang_code = request.form.get('lang', 'en')
    stt_lang = 'hi-IN' if lang_code == 'hi' else 'en-US'
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language=stt_lang)
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError:
            return jsonify({'error': 'Speech recognition service failed'}), 500

# --- Text-to-Speech Function ---
def speak(text, lang='en'):
    """Convert text to speech using gTTS"""
    try:
        tts_lang = 'hi' if lang == 'hi' else 'en'
        tts = gTTS(text=text, lang=tts_lang)
        tts.save("output.mp3")
        
        # Play audio based on operating system
        import platform
        if platform.system() == "Windows":
            os.system("start output.mp3")
        else:
            # Linux/Mac
            os.system("mpg123 output.mp3")
            
        return True
    except Exception as e:
        logging.error(f"TTS failed: {str(e)}")
        return False

# --- Text-to-Speech Endpoint ---
@app.route('/api/tts', methods=['POST'])
@login_required
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Use the new speak function
    if speak(text, lang):
        return jsonify({"status": "ok", "audio_url": "/output.mp3", "answer": text})
    else:
        return jsonify({"error": "TTS failed"}), 500

# --- Serve TTS audio ---
@app.route('/output.mp3')
def serve_audio():
    return send_file("output.mp3", mimetype="audio/mpeg")

# --- History Endpoint ---
@app.route('/history')
@login_required
def history():
    user_id = current_user.id
    is_admin = (current_user.username == 'admin')
    with sqlite3.connect(DB_NAME) as conn:
        if is_admin:
            rows = conn.execute("""
                SELECT users.username, questions.question, questions.answer, questions.timestamp
                FROM questions JOIN users ON questions.user_id = users.id
                ORDER BY questions.timestamp DESC
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT question, answer, timestamp
                FROM questions
                WHERE user_id = ?
                ORDER BY timestamp DESC
            """, (user_id,)).fetchall()

    return render_template_string(HISTORY_TEMPLATE, history=rows, is_admin=is_admin)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Science Q&A Bot</title>
  <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    .loader {
      border-top-color: #6366f1;
      animation: spin 1s linear infinite;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    
    /* Dark mode styles */
    .dark {
      color-scheme: dark;
    }
    
    /* Body and main container dark mode */
    .dark body {
      background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
      color: #e2e8f0;
    }
    
    /* Main container dark mode */
    .dark .bg-gradient-to-br.from-white.to-purple-50 {
      background: linear-gradient(135deg, #1e1b4b 0%, #2d1b69 100%);
      border-color: #4c1d95;
    }
    
    /* Header section dark mode */
    .dark .bg-gradient-to-r.from-purple-600.via-pink-600.to-purple-700 {
      background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 0%, #4c1d95 100%);
    }
    
    /* Content area dark mode */
    .dark .p-6 {
      background: linear-gradient(135deg, #1e1b4b 0%, #2d1b69 100%);
    }
    
    /* Form inputs dark mode */
    .dark input[type="text"],
    .dark input[type="file"],
    .dark select {
      background: linear-gradient(135deg, #2d1b69 0%, #3b21a8 100%);
      border-color: #4c1d95;
      color: #e2e8f0;
    }
    
    .dark input[type="text"]::placeholder {
      color: #a1a1aa;
    }
    
    .dark input[type="text"]:focus,
    .dark input[type="file"]:focus,
    .dark select:focus {
      border-color: #7c3aed;
      background: linear-gradient(135deg, #3b21a8 0%, #4c1d95 100%);
    }
    
    /* Button dark mode enhancements */
    .dark button {
      border-color: #4c1d95;
    }
    
    /* Quick action buttons dark mode */
    .dark .bg-gradient-to-r.from-purple-500.to-pink-500 {
      background: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
    }
    
    .dark .bg-gradient-to-r.from-blue-500.to-purple-500 {
      background: linear-gradient(135deg, #3b82f6 0%, #7c3aed 100%);
    }
    
    .dark .bg-gradient-to-r.from-green-500.to-blue-500 {
      background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
    }
    
    .dark .bg-gradient-to-r.from-yellow-500.to-orange-500 {
      background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
    }
    
    /* Form buttons dark mode */
    .dark .bg-gradient-to-r.from-green-500.to-emerald-600 {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .dark .bg-gradient-to-r.from-blue-600.to-blue-700 {
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    .dark .bg-gradient-to-r.from-red-500.to-pink-500 {
      background: linear-gradient(135deg, #ef4444 0%, #ec4899 100%);
    }
    
    .dark .bg-gradient-to-r.from-yellow-400.to-orange-500 {
      background: linear-gradient(135deg, #fbbf24 0%, #f97316 100%);
    }
    
    .dark .bg-gradient-to-r.from-blue-500.to-blue-600 {
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .dark .bg-gradient-to-r.from-green-500.to-green-600 {
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    /* Progress bar dark mode */
    .dark .bg-gray-200 {
      background: #374151;
    }
    
    .dark .bg-gray-700 {
      background: #4b5563;
    }
    
    /* Answer container dark mode */
    .dark .bg-gradient-to-br.from-blue-50.to-white {
      background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
      border-color: #3b82f6;
    }
    
    /* Error container dark mode */
    .dark .bg-gradient-to-r.from-red-50.to-pink-50 {
      background: linear-gradient(135deg, #7f1d1d 0%, #9d174d 100%);
      border-color: #ef4444;
    }
    
    /* Developer section dark mode */
    .dark .text-gray-700 {
      color: #d1d5db;
    }
    
    .dark .text-gray-800 {
      color: #e5e7eb;
    }
    
    .dark .text-gray-600 {
      color: #9ca3af;
    }
    
    /* History button dark mode */
    .dark .bg-gradient-to-r.from-purple-100.to-pink-100 {
      background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 100%);
      color: #e5e7eb;
      border-color: #7c3aed;
    }
    
    .dark .hover\\:from-purple-200.hover\\:to-pink-200:hover {
      background: linear-gradient(135deg, #5b21b6 0%, #8b5cf6 100%);
    }
    
    /* Social media icons dark mode */
    .dark .bg-gray-800 {
      background: #1f2937;
    }
    
    .dark .hover\\:bg-gray-900:hover {
      background: #111827;
    }
    
    .dark .bg-blue-600 {
      background: #2563eb;
    }
    
    .dark .hover\\:bg-blue-700:hover {
      background: #1d4ed8;
    }
    
    /* Text colors for dark mode */
    .dark .text-gray-700 {
      color: #d1d5db;
    }
    
    .dark .text-gray-500 {
      color: #9ca3af;
    }
    
    .dark .text-gray-600 {
      color: #9ca3af;
    }
    
    .dark .text-gray-800 {
      color: #e5e7eb;
    }
    
    .dark .text-gray-300 {
      color: #d1d5db;
    }
    
    .dark .text-gray-200 {
      color: #e5e7eb;
    }
    
    .dark .text-gray-400 {
      color: #9ca3af;
    }
    
    /* Border colors for dark mode */
    .dark .border-gray-200 {
      border-color: #4c1d95;
    }
    
    .dark .border-blue-200 {
      border-color: #3b82f6;
    }
    
    .dark .border-blue-300 {
      border-color: #60a5fa;
    }
    
    .dark .border-purple-200 {
      border-color: #7c3aed;
    }
    
    /* Focus states for dark mode */
    .dark .focus\\:ring-blue-500:focus {
      --tw-ring-color: #7c3aed;
    }
    
    .dark .focus\\:border-blue-500:focus {
      border-color: #7c3aed;
    }
  </style>
</head>
<body class="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900 dark:to-pink-900 min-h-screen flex items-center justify-center p-4 transition-colors duration-300" data-username="{{ username }}">
  <div class="w-full max-w-4xl bg-gradient-to-br from-white to-purple-50 dark:from-gray-900 dark:to-purple-900 rounded-2xl shadow-2xl overflow-hidden border border-purple-200 dark:border-purple-700">
    <div class="p-6 bg-gradient-to-r from-purple-600 via-pink-600 to-purple-700 text-white relative">
      <div class="flex justify-between items-start">
        <div>
          <h1 class="text-3xl font-bold bg-gradient-to-r from-white to-pink-100 bg-clip-text text-transparent">Science Q&A Bot</h1>
          <p class="text-purple-100 mt-2">Ask questions about science topics or upload a PDF</p>
          <p class="text-purple-100 mt-1">Welcome, {{ username }}</p>
        </div>
        <button id="dark-mode-toggle" class="p-2 rounded-lg bg-purple-500 hover:bg-purple-600 transition-colors">
          <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
          </svg>
        </button>
      </div>
    </div>

    <div class="p-6 dark:bg-gray-900 dark:bg-opacity-50">
      <!-- Quick Action Buttons -->
      <div class="mb-6">
        <h3 class="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3">Quick Questions</h3>
        <div class="flex flex-wrap gap-2">
          <button onclick="askQuickQuestion('What is photosynthesis?')" class="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105">
            🌱 Photosynthesis
          </button>
          <button onclick="askQuickQuestion('How does gravity work?')" class="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-300 transform hover:scale-105">
            🌍 Gravity
          </button>
          <button onclick="askQuickQuestion('What is DNA?')" class="px-4 py-2 bg-gradient-to-r from-green-500 to-blue-500 text-white rounded-lg hover:from-green-600 hover:to-blue-600 transition-all duration-300 transform hover:scale-105">
            🧬 DNA
          </button>
          <button onclick="askQuickQuestion('How do atoms work?')" class="px-4 py-2 bg-gradient-to-r from-yellow-500 to-orange-500 text-white rounded-lg hover:from-yellow-600 hover:to-orange-600 transition-all duration-300 transform hover:scale-105">
            ⚛️ Atoms
          </button>
        </div>
      </div>

      <!-- PDF Upload Form -->
      <form id="pdf-form" class="mb-6">
        <div class="flex space-x-2 mb-2">
          <input type="file" id="pdf-input" accept="application/pdf" class="flex-1 px-4 py-3 border border-blue-300 dark:border-blue-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-white" />
          <button type="button" onclick="uploadPDF()" class="px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg shadow-lg hover:from-green-600 hover:to-emerald-700 transition-all duration-300 transform hover:scale-105">
            📄 Upload PDF
          </button>
          <button type="button" onclick="viewPDF()" class="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg shadow-lg hover:from-blue-600 hover:to-indigo-700 transition-all duration-300 transform hover:scale-105" id="view-pdf-btn">
            📋 View PDF
          </button>
        </div>
      </form>

      <!-- Question Form -->
      <form id="qa-form" class="mb-6">
        <div class="flex space-x-2 mb-2">
          <input type="text" name="question" placeholder="Ask a question..." 
            class="flex-1 px-4 py-3 border border-blue-300 dark:border-blue-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-white dark:placeholder-gray-400" required>
          <button type="submit"
            class="px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg shadow-lg
              animate-pulse hover:animate-none hover:scale-105 hover:from-blue-700 hover:to-blue-800
              transform transition duration-300 ease-in-out flex items-center justify-center min-w-32">
            <span id="submit-text">Get Answer</span>
            <span id="spinner" class="hidden ml-2">
              <svg class="w-5 h-5 text-white loader" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 
                  1.135 5.824 3 7.938l3-2.647z">
                </path>
              </svg>
            </span>
          </button>
          <button type="button" id="stop-btn"
            class="px-4 py-3 bg-gradient-to-r from-red-500 to-pink-500 text-white rounded-lg shadow-lg hover:from-red-600 hover:to-pink-600 transition-all duration-300 transform hover:scale-105 hidden">
            ⏹️ Stop
          </button>
        </div>

        <div class="flex space-x-2 items-center">
          <select id="lang" class="px-3 py-2 border border-blue-300 dark:border-blue-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:text-white">
            <option value="en" selected class="dark:bg-gray-800 dark:text-white">🇺🇸 English</option>
            <option value="hi" class="dark:bg-gray-800 dark:text-white">🇮🇳 Hindi</option>
          </select>
          <button type="button" onclick="triggerOCR()" class="px-3 py-2 bg-gradient-to-r from-yellow-400 to-orange-500 text-white rounded-lg hover:from-yellow-500 hover:to-orange-600 transition-all duration-300 transform hover:scale-105">📸 OCR</button>
          <button type="button" onclick="openCamera()" class="px-3 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105">📷 Camera</button>
          <button type="button" onclick="recordVoice()" class="px-3 py-2 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all duration-300 transform hover:scale-105 animate-pulse">🎤 Voice</button>
        </div>
        <input type="file" id="ocr-input" accept="image/*" hidden onchange="uploadOCRImage(this.files[0])" />
      </form>

      <!-- Progress Bar -->
      <div id="progress-container" class="hidden mb-4">
        <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div id="progress-bar" class="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
        <p id="progress-text" class="text-sm text-gray-600 dark:text-gray-400 mt-1">Processing...</p>
      </div>

      <div id="answer-container" class="hidden bg-gradient-to-br from-blue-50 to-white rounded-lg p-4 min-h-32 border border-blue-200 dark:from-blue-900 dark:to-blue-800 dark:border-blue-700">
        <div class="flex justify-between items-start mb-3">
          <h2 class="font-semibold text-lg text-gray-800 dark:text-gray-200">Answer:</h2>
          <button onclick="copyAnswer()" class="px-3 py-1 bg-gradient-to-r from-blue-500 to-blue-600 text-white text-sm rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all duration-300 transform hover:scale-105">
            📋 Copy
          </button>
        </div>
        <div id="answer-content" class="prose max-w-none whitespace-pre-wrap text-gray-700 dark:text-gray-300"></div>
      </div>

      <!-- Developer Info Section -->
      <div class="mt-6 flex justify-between items-center">
        <div class="flex items-center space-x-4">
          <!-- Profile Picture -->
          <div class="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg">
            AK
          </div>
          
          <!-- Developer Details -->
          <div class="text-sm text-gray-700 dark:text-gray-300">
            <div class="font-semibold text-gray-800 dark:text-gray-200">Developed by Akash Kumar</div>
            <div class="text-gray-600 dark:text-gray-400">Full Stack Developer & AI Enthusiast</div>
          </div>
          
          <!-- Social Media Links -->
          <div class="flex space-x-3">
            <a href="https://github.com/Akashsinghkumar" target="_blank" rel="noopener noreferrer" 
               class="w-8 h-8 bg-gray-800 hover:bg-gray-900 rounded-full flex items-center justify-center text-white transition-all duration-300 transform hover:scale-110 shadow-lg">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
            <a href="https://www.linkedin.com/in/akash-kumar-3baa04250?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" rel="noopener noreferrer" 
               class="w-8 h-8 bg-blue-600 hover:bg-blue-700 rounded-full flex items-center justify-center text-white transition-all duration-300 transform hover:scale-110 shadow-lg">
              <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.047-1.852-3.047-1.853 0-2.136 1.445-2.136 2.939v5.677H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
              </svg>
            </a>
          </div>
        </div>
        
        <a href="/history"
           class="inline-block bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 font-medium py-2 px-4 rounded-lg border border-purple-200 hover:from-purple-200 hover:to-pink-200 transition-all duration-300 transform hover:scale-105 dark:from-purple-800 dark:to-pink-800 dark:text-purple-200 dark:border-purple-600">
          📜 View History
        </a>
      </div>

      <div id="error-container" class="hidden bg-gradient-to-r from-red-50 to-pink-50 border-l-4 border-red-500 p-4 mt-4 rounded-r-lg dark:from-red-900 dark:to-pink-900 dark:border-red-400">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
          </div>
          <div class="ml-3">
            <h3 id="error-title" class="text-sm font-medium text-red-800 dark:text-red-200"></h3>
            <div id="error-detail" class="mt-2 text-sm text-red-700 dark:text-red-300"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
let currentController = null;
let currentAudio = null;
let voiceController = null;
let isRecording = false;

// Camera modal markup injected dynamically
document.addEventListener('DOMContentLoaded', async () => {
  const modalHtml = `
  <div id="camera-modal" class="fixed inset-0 bg-black bg-opacity-60 hidden flex items-center justify-center z-50">
    <div class="bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-xl w-11/12 max-w-xl">
      <div class="p-4 border-b border-gray-200 dark:border-gray-600 flex justify-between items-center">
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">Camera</h3>
        <button onclick="closeCamera()" class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-700 dark:text-white">Close</button>
      </div>
      <div class="p-4">
        <video id="camera-video" class="w-full rounded" autoplay playsinline></video>
      </div>
      <div class="p-4 border-t border-gray-200 dark:border-gray-600 flex justify-end space-x-2">
        <button onclick="capturePhoto()" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800">Capture</button>
      </div>
    </div>
  </div>`;
  document.body.insertAdjacentHTML('beforeend', modalHtml);
  
  // Play welcome message
  setTimeout(() => playWelcomeMessage(), 1000);
  
  // Initialize dark mode toggle
  initializeDarkMode();
  
  // Add keyboard shortcuts
  addKeyboardShortcuts();
  
  // Initialize PDF state
  window.currentPDF = null;
});

async function uploadPDF() {
  const pdfInput = document.getElementById("pdf-input");
  if (!pdfInput.files.length) return alert("Select a PDF file first");
  const formData = new FormData();
  formData.append("pdf", pdfInput.files[0]);

  const res = await fetch("/api/upload_pdf", { method: "POST", body: formData });
  const data = await res.json();
  if (data.status) {
    alert(data.status);
    // Store PDF info for viewing
    window.currentPDF = {
      name: pdfInput.files[0].name,
      size: pdfInput.files[0].size,
      uploadedAt: new Date().toLocaleString()
    };
    
    // Update the view PDF button to show it's active
    const viewPdfBtn = document.getElementById("view-pdf-btn");
    if (viewPdfBtn) {
      viewPdfBtn.innerHTML = "📋 View PDF ✓";
      viewPdfBtn.classList.add("from-green-500", "to-green-600");
      viewPdfBtn.classList.remove("from-blue-500", "to-indigo-600");
    }
  } else {
    alert("PDF upload failed: " + (data.error || "Unknown error"));
  }
}

async function viewPDF() {
  // Check if user has uploaded a PDF
  if (!window.currentPDF) {
    alert("Please upload a PDF first!");
    return;
  }
  
  try {
    // Create a modal to display PDF information
    const modalHtml = `
      <div id="pdf-modal" class="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
        <div class="bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-xl w-11/12 max-w-2xl max-h-[80vh] overflow-y-auto">
          <div class="p-4 border-b border-gray-200 dark:border-gray-600 flex justify-between items-center">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white">📋 PDF Information</h3>
            <button onclick="closePDFModal()" class="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300 dark:bg-gray-600 dark:hover:bg-gray-700 dark:text-white">✕</button>
          </div>
          <div class="p-6">
            <div class="mb-4">
              <h4 class="font-semibold text-gray-800 dark:text-white mb-2">📄 Current PDF:</h4>
              <p class="text-gray-600 dark:text-gray-300">${window.currentPDF.name}</p>
            </div>
            <div class="mb-4">
              <h4 class="text-sm text-gray-500 dark:text-gray-400">💡 You can now ask questions about this PDF!</h4>
            </div>
            <div class="bg-blue-50 dark:bg-blue-900 p-3 rounded-lg">
              <p class="text-sm text-blue-800 dark:text-blue-200">
                <strong>Tip:</strong> Ask questions like "What is this document about?" or "Summarize the main points" to get answers from your PDF.
              </p>
            </div>
          </div>
        </div>
      </div>`;
    
    // Remove existing modal if any
    const existingModal = document.getElementById("pdf-modal");
    if (existingModal) existingModal.remove();
    
    // Add new modal
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
  } catch (error) {
    alert("Error viewing PDF: " + error.message);
  }
}

function closePDFModal() {
  const modal = document.getElementById("pdf-modal");
  if (modal) modal.remove();
}

function resetPDFButton() {
  const viewPdfBtn = document.getElementById("view-pdf-btn");
  if (viewPdfBtn) {
    viewPdfBtn.innerHTML = "📋 View PDF";
    viewPdfBtn.classList.remove("from-green-500", "to-green-600");
    viewPdfBtn.classList.add("from-blue-500", "to-indigo-600");
  }
}

document.getElementById("qa-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const questionInput = e.target.elements["question"];
  const question = questionInput.value.trim();

  document.getElementById("submit-text").classList.add("hidden");
  document.getElementById("spinner").classList.remove("hidden");
  document.getElementById("answer-container").classList.add("hidden");
  document.getElementById("error-container").classList.add("hidden");
  document.getElementById("answer-content").innerHTML = "";
  document.getElementById("stop-btn").classList.remove("hidden");

  try {
    // stop any previous audio playback
    if (currentAudio) {
      try { currentAudio.pause(); currentAudio.currentTime = 0; } catch (e) {}
      currentAudio = null;
    }

    // setup abort controller
    currentController = new AbortController();
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: currentController.signal
    });

    if (!response.ok) throw new Error("Server error");

    const result = await response.json();
    document.getElementById("answer-container").classList.remove("hidden");
    const container = document.getElementById("answer-content");
    const headingSafe = (result.heading || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    const bodySafe = (result.body || '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
    container.innerHTML = `<strong>${headingSafe}</strong> ${bodySafe}`;

    const selectedLang = document.getElementById("lang").value || "en";
    const ttsRes = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: `${result.heading ? result.heading + ': ' : ''}${result.body || ''}`, lang: selectedLang })
    });
    if (!ttsRes.ok) {
      const err = await ttsRes.json().catch(() => ({ error: 'TTS error' }));
      throw new Error(err.error || 'TTS request failed');
    }
    const ttsData = await ttsRes.json();
    if (ttsData.audio_url) {
      currentAudio = new Audio(ttsData.audio_url + `?t=${Date.now()}`);
      currentAudio.play();
    } else if (ttsData.error) {
      throw new Error(ttsData.error);
    }

  } catch (err) {
    if (err && err.name === 'AbortError') {
      document.getElementById("error-title").textContent = "Cancelled:";
      document.getElementById("error-detail").textContent = "Request was cancelled.";
      document.getElementById("error-container").classList.remove("hidden");
    } else {
      document.getElementById("error-title").textContent = "Error:";
      document.getElementById("error-detail").textContent = err.message || String(err);
      document.getElementById("error-container").classList.remove("hidden");
    }
  } finally {
    document.getElementById("submit-text").classList.remove("hidden");
    document.getElementById("spinner").classList.add("hidden");
    document.getElementById("stop-btn").classList.add("hidden");
    currentController = null;
  }
});

function triggerOCR() {
  document.getElementById("ocr-input").click();
}

async function uploadOCRImage(file) {
  const formData = new FormData();
  formData.append("image", file);
  const response = await fetch("/api/ocr", { method: "POST", body: formData });
  const data = await response.json();
  if (data.question) {
    document.querySelector("input[name='question']").value = data.question;
  } else {
    alert("OCR failed: " + (data.error || "Unknown error"));
  }
}

let mediaRecorder;
let audioChunks = [];
async function recordVoice() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];
  isRecording = true;
  mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");
    const selectedLang = document.getElementById("lang").value || "en";
    formData.append("lang", selectedLang);
    try {
      voiceController = new AbortController();
      const response = await fetch("/api/voice", { method: "POST", body: formData, signal: voiceController.signal });
      const result = await response.json();
      if (result.text) {
        document.querySelector("input[name='question']").value = result.text;
        document.getElementById("qa-form").dispatchEvent(new Event('submit'));
      } else if (result.error) {
        alert("Voice recognition failed: " + result.error);
      }
    } catch (e) {
      if (!(e && e.name === 'AbortError')) {
        console.error(e);
      }
    } finally {
      voiceController = null;
    }
    // stop mic tracks
    try { stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    isRecording = false;
  };
  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 4000);
}

// Camera capture for OCR
let cameraStream;
async function openCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('Camera not supported in this browser');
      return;
    }
    const constraints = { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };
    cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
    const modal = document.getElementById('camera-modal');
    const videoEl = document.getElementById('camera-video');
    videoEl.srcObject = cameraStream;
    await videoEl.play();
    modal.classList.remove('hidden');
  } catch (e) {
    alert('Unable to access camera: ' + (e.message || e));
  }
}

function closeCamera() {
  const modal = document.getElementById('camera-modal');
  modal.classList.add('hidden');
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
}

async function capturePhoto() {
  const videoEl = document.getElementById('camera-video');
  if (!videoEl) return;
  const canvas = document.createElement('canvas');
  const w = videoEl.videoWidth || 640;
  const h = videoEl.videoHeight || 480;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, w, h);
  canvas.toBlob(async (blob) => {
    if (!blob) { alert('Failed to capture image'); return; }
    const file = new File([blob], 'capture.png', { type: 'image/png' });
    await uploadOCRImage(file);
    closeCamera();
  }, 'image/png', 0.92);
}

async function playWelcomeMessage() {
  const selectedLang = document.getElementById("lang").value || "en";
  const welcomeText = selectedLang === "hi" ? 
    "नमस्ते! मैं आपकी विज्ञान प्रश्नोत्तरी में मदद करने के लिए यहाँ हूँ। कोई भी प्रश्न पूछें।" : 
    "Hello! I'm here to help you with science questions. Ask me anything.";
  
  try {
    const ttsRes = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: welcomeText, lang: selectedLang })
    });
    if (ttsRes.ok) {
      const ttsData = await ttsRes.json();
      if (ttsData.audio_url) {
        const audio = new Audio(ttsData.audio_url + `?t=${Date.now()}`);
        audio.play();
      }
    }
  } catch (e) {
    console.log('Welcome message failed:', e);
  }
}

// Stop Answering button logic
document.getElementById('stop-btn').addEventListener('click', () => {
  try {
    if (currentController) {
      currentController.abort();
    }
    if (currentAudio) {
      currentAudio.pause();
      currentAudio.currentTime = 0;
      currentAudio = null;
    }
    if (voiceController) {
      try { voiceController.abort(); } catch (e) {}
      voiceController = null;
    }
    if (mediaRecorder && isRecording) {
      try { mediaRecorder.stop(); } catch (e) {}
      isRecording = false;
    }
  } catch (e) {}
  document.getElementById("submit-text").classList.remove("hidden");
  document.getElementById("spinner").classList.add("hidden");
  document.getElementById("stop-btn").classList.add("hidden");
});

// Quick question function
function askQuickQuestion(question) {
  document.querySelector('input[name="question"]').value = question;
  document.getElementById('qa-form').dispatchEvent(new Event('submit'));
}

// Copy answer function
async function copyAnswer() {
  const answerContent = document.getElementById('answer-content');
  const text = answerContent.textContent || answerContent.innerText;
  
  try {
    await navigator.clipboard.writeText(text);
    showNotification('Answer copied to clipboard!', 'success');
  } catch (err) {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    showNotification('Answer copied to clipboard!', 'success');
  }
}

// Progress bar functions
function startProgress() {
  const container = document.getElementById('progress-container');
  const bar = document.getElementById('progress-bar');
  const text = document.getElementById('progress-text');
  
  container.classList.remove('hidden');
  bar.style.width = '0%';
  text.textContent = 'Processing...';
  
  let progress = 0;
  const interval = setInterval(() => {
    progress += Math.random() * 15;
    if (progress > 90) progress = 90;
    bar.style.width = progress + '%';
  }, 200);
  
  return interval;
}

function completeProgress(interval) {
  if (interval) clearInterval(interval);
  
  const bar = document.getElementById('progress-bar');
  const text = document.getElementById('progress-text');
  
  bar.style.width = '100%';
  text.textContent = 'Complete!';
  
  setTimeout(() => {
    document.getElementById('progress-container').classList.add('hidden');
  }, 1000);
}

// Dark mode functions
function initializeDarkMode() {
  const toggle = document.getElementById('dark-mode-toggle');
  const body = document.body;
  
  // Check for saved preference
  const isDark = localStorage.getItem('darkMode') === 'true';
  if (isDark) {
    body.classList.add('dark');
    updateDarkModeIcon(true);
  }
  
  toggle.addEventListener('click', () => {
    body.classList.toggle('dark');
    const isDark = body.classList.contains('dark');
    localStorage.setItem('darkMode', isDark);
    updateDarkModeIcon(isDark);
  });
}

function updateDarkModeIcon(isDark) {
  const toggle = document.getElementById('dark-mode-toggle');
  if (isDark) {
    toggle.innerHTML = `<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
      <path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd"></path>
    </svg>`;
  } else {
    toggle.innerHTML = `<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
      <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"></path>
    </svg>`;
  }
}

// Keyboard shortcuts
function addKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      document.getElementById('qa-form').dispatchEvent(new Event('submit'));
    }
    
    // Ctrl/Cmd + K to focus input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      document.querySelector('input[name="question"]').focus();
    }
    
    // Escape to clear
    if (e.key === 'Escape') {
      document.querySelector('input[name="question"]').value = '';
      document.getElementById('answer-container').classList.add('hidden');
      document.getElementById('error-container').classList.add('hidden');
    }
  });
}

// Notification system
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 transition-all duration-300 transform translate-x-full ${
    type === 'success' ? 'bg-green-500 text-white' : 
    type === 'error' ? 'bg-red-500 text-white' : 
    'bg-blue-500 text-white'
  }`;
  notification.textContent = message;
  
  document.body.appendChild(notification);
  
  // Animate in
  setTimeout(() => notification.classList.remove('translate-x-full'), 100);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.classList.add('translate-x-full');
    setTimeout(() => document.body.removeChild(notification), 300);
  }, 3000);
}
</script>
</body>
</html>
"""

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login - Q&A Bot</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gradient-to-br from-purple-50 to-pink-50 flex items-center justify-center h-screen">
    <div class="bg-gradient-to-br from-white to-purple-50 p-8 rounded-2xl shadow-2xl w-full max-w-sm border border-purple-200">
        <h2 class="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required class="w-full mb-4 px-4 py-3 border border-purple-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500">
            <div class="relative mb-6">
                <input type="password" name="password" id="password" placeholder="Password" required class="w-full px-4 py-3 pr-12 border border-purple-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500">
                <button type="button" onclick="togglePassword()" class="absolute inset-y-0 right-0 pr-3 flex items-center">
                    <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                </button>
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 font-semibold">Login</button>
        </form>
        <p class="mt-4 text-center text-sm text-gray-600">Don't have an account? <a href="{{ url_for('signup') }}" class="text-purple-600 hover:text-pink-600 font-medium">Sign up</a></p>
    </div>
    
    <script>
    function togglePassword() {
        const passwordInput = document.getElementById('password');
        const button = event.target.closest('button');
        const svg = button.querySelector('svg');
        
        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" />';
        } else {
            passwordInput.type = 'password';
            svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />';
        }
    }
    </script>
</body>
</html>
"""

SIGNUP_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Signup - Q&A Bot</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gradient-to-br from-purple-50 to-pink-50 flex items-center justify-center h-screen">
    <div class="bg-gradient-to-br from-white to-purple-50 p-8 rounded-2xl shadow-2xl w-full max-w-sm border border-purple-200">
        <h2 class="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Sign Up</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required class="w-full mb-4 px-4 py-3 border border-purple-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500">
            <div class="relative mb-6">
                <input type="password" name="password" id="signup-password" placeholder="Password" required class="w-full px-4 py-3 pr-12 border border-purple-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500">
                <button type="button" onclick="toggleSignupPassword()" class="absolute inset-y-0 right-0 pr-3 flex items-center">
                    <svg class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                </button>
            </div>
            <button type="submit" class="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 font-semibold">Sign Up</button>
        </form>
        <p class="mt-4 text-center text-sm text-gray-600">Already have an account? <a href="{{ url_for('login') }}" class="text-purple-600 hover:text-pink-600 font-medium">Login</a></p>
    </div>
    
    <script>
    function toggleSignupPassword() {
        const passwordInput = document.getElementById('signup-password');
        const button = event.target.closest('button');
        const svg = button.querySelector('svg');
        
        if (passwordInput.type === 'password') {
            passwordInput.type = 'text';
            svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.878 9.878L3 3m6.878 6.878L21 21" />';
        } else {
            passwordInput.type = 'password';
            svg.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />';
        }
    }
    </script>
</body>
</html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chat History</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gradient-to-br from-purple-50 to-pink-50 min-h-screen p-6">
    <div class="max-w-4xl mx-auto bg-gradient-to-br from-white to-purple-50 p-6 rounded-2xl shadow-2xl border border-purple-200">
        <h2 class="text-3xl font-bold mb-6 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Chat History</h2>
        <a href="/" class="inline-block bg-gradient-to-r from-purple-100 to-pink-100 text-purple-700 font-medium py-2 px-4 rounded-lg border border-purple-200 hover:from-purple-200 hover:to-pink-200 transition-all duration-300 transform hover:scale-105 mb-6">
            ⬅ Back to Home
        </a>
        <div class="mt-4 space-y-4">
            {% for row in history %}
                <div class="p-4 border border-purple-200 rounded-lg bg-gradient-to-br from-purple-50 to-pink-50 hover:from-purple-100 hover:to-pink-100 transition-all duration-300">
                    {% if is_admin %}
                        <p class="text-sm text-gray-700 mb-1"><strong>User:</strong> {{ row[0] }}</p>
                        <p><strong>Q:</strong> {{ row[1] }}</p>
                        <p><strong>A:</strong> {{ row[2] }}</p>
                        <p class="text-xs text-gray-500 mt-1">{{ row[3] }}</p>
                    {% else %}
                        <p><strong>Q:</strong> {{ row[0] }}</p>
                        <p><strong>A:</strong> {{ row[1] }}</p>
                        <p class="text-xs text-gray-500 mt-1">{{ row[2] }}</p>
                    {% endif %}
                </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render देगा PORT env var
    app.run(host="0.0.0.0", port=port)

# Ensure the Tesseract OCR executable is in your PATH
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
  
# Ensure the Flask app is running with the correct configurations
if not qa_bot.initialized:
    qa_bot.initialize()
# Ensure the database is initialized
# (Database is already initialized by init_db() above)