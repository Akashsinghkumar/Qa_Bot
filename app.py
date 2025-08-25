import logging
import json
import requests
import sqlite3
import os
import speech_recognition as sr
import pyttsx3
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

            # Store the chat into DB
            with sqlite3.connect(DB_NAME) as conn:
                conn.execute("""
                    INSERT INTO questions (user_id, question, answer)
                    VALUES (?, ?, ?)""",
                    (current_user.id, question, answer)
                )

            return answer
        except Exception as e:
            logging.error(f"❌ Model error: {e}", exc_info=True)
            return "Sorry, something went wrong with the model."

qa_bot = QABot()

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
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/ask', methods=['POST'])
@login_required
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    lang = data.get("lang", "en")

    if not question:
        return jsonify({"message": "No question provided."}), 400

    answer = qa_bot.ask_ollama(question)
    return jsonify({"answer": answer})

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

@app.route('/api/voice', methods=['POST'])
@login_required
def voice_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError:
            return jsonify({'error': 'Speech recognition service failed'}), 500

@app.route('/api/tts', methods=['POST'])
@login_required
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if lang == 'hi' and 'hindi' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
        elif lang == 'en' and 'english' in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    audio_fp = BytesIO()
    engine.save_to_file(text, "output.mp3")
    engine.runAndWait()
    return jsonify({"status": "ok", "audio_url": "/output.mp3", "answer": text})

@app.route('/output.mp3')
def serve_audio():
    return send_file("output.mp3", mimetype="audio/mpeg")

@app.route('/history')
@login_required
def history():
    user_id = current_user.id
    is_admin = current_user.username == "admin"
    with sqlite3.connect(DB_NAME) as conn:
        if is_admin:
            history_data = conn.execute("""
                SELECT users.username, questions.question, questions.answer, questions.timestamp
                FROM questions
                JOIN users ON questions.user_id = users.id
                ORDER BY questions.timestamp DESC
            """).fetchall()
        else:
            history_data = conn.execute("""
                SELECT question, answer, timestamp FROM questions
                WHERE user_id = ? ORDER BY timestamp DESC
            """, (user_id,)).fetchall()
    return render_template_string(HISTORY_TEMPLATE, history=history_data, is_admin=is_admin)

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
  </style>
</head>
<body class="bg-gray-50 min-h-screen flex items-center justify-center p-4">
  <div class="w-full max-w-2xl bg-white rounded-xl shadow-lg overflow-hidden">
    <div class="p-6 bg-indigo-600 text-white">
      <h1 class="text-2xl font-bold">Science Q&A Bot</h1>
      <p class="text-indigo-100">Ask questions about science topics</p>
    </div>

    <div class="p-6">
      <form id="qa-form" class="mb-6">
        <div class="flex space-x-2 mb-2">
          <input type="text" name="question" placeholder="Ask a question..." 
            class="flex-1 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-indigo-500" required>
          <button type="submit"
            class="px-6 py-3 bg-indigo-600 text-white rounded-lg shadow-lg
              animate-pulse hover:animate-none hover:scale-105 hover:bg-indigo-700
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
        </div>

        <div class="flex space-x-2">
          <button type="button" onclick="triggerOCR()" class="px-3 py-2 bg-yellow-400 rounded">📸 OCR</button>
          <button type="button" onclick="recordVoice()" class="px-3 py-2 bg-green-400 text-white rounded">🎤 Voice</button>
        </div>
        <input type="file" id="ocr-input" accept="image/*" hidden onchange="uploadOCRImage(this.files[0])" />
      </form>

      <div id="answer-container" class="hidden bg-gray-50 rounded-lg p-4 min-h-32 border border-gray-200">
        <h2 class="font-semibold text-lg mb-2">Answer:</h2>
        <div id="answer-content" class="prose max-w-none whitespace-pre-wrap"></div>
      </div>

      <div class="mt-4 text-right">
        <a href="/history"
           class="inline-block bg-gray-100 text-indigo-700 font-medium py-2 px-4 rounded-lg border border-indigo-200 hover:bg-indigo-200 transition">
          📜 View History
        </a>
      </div>

      <div id="error-container" class="hidden bg-red-50 border-l-4 border-red-500 p-4 mt-4">
        <div class="flex">
          <div class="flex-shrink-0">
            <svg class="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
            </svg>
          </div>
          <div class="ml-3">
            <h3 id="error-title" class="text-sm font-medium text-red-800"></h3>
            <div id="error-detail" class="mt-2 text-sm text-red-700"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
document.getElementById("qa-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const questionInput = e.target.elements["question"];
  const question = questionInput.value.trim();

  document.getElementById("submit-text").classList.add("hidden");
  document.getElementById("spinner").classList.remove("hidden");
  document.getElementById("answer-container").classList.add("hidden");
  document.getElementById("error-container").classList.add("hidden");
  document.getElementById("answer-content").innerHTML = "";

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });

    if (!response.ok) throw new Error("Server error");

    const result = await response.json();
    document.getElementById("answer-container").classList.remove("hidden");
    document.getElementById("answer-content").innerHTML = result.answer;

    // Auto TTS playback
    const ttsRes = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: result.answer, lang: "en" })
    });
    const ttsData = await ttsRes.json();
    if (ttsData.audio_url) {
      const audio = new Audio(ttsData.audio_url);
      audio.play();
    }

  } catch (err) {
    document.getElementById("error-title").textContent = "Error:";
    document.getElementById("error-detail").textContent = err.message;
    document.getElementById("error-container").classList.remove("hidden");
  } finally {
    document.getElementById("submit-text").classList.remove("hidden");
    document.getElementById("spinner").classList.add("hidden");
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

  mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append("audio", audioBlob, "audio.wav");

    const response = await fetch("/api/voice", { method: "POST", body: formData });
    const result = await response.json();
    if (result.text) {
      document.querySelector("input[name='question']").value = result.text;
    } else {
      alert("Voice recognition failed: " + result.error);
    }
  };

  mediaRecorder.start();
  setTimeout(() => mediaRecorder.stop(), 4000);  // Record for 4 seconds
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
<body class="bg-indigo-100 flex items-center justify-center h-screen">
    <div class="bg-white p-8 rounded-xl shadow-md w-full max-w-sm">
        <h2 class="text-2xl font-bold mb-6 text-center text-indigo-600">Login</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required class="w-full mb-4 px-4 py-2 border rounded-lg">
            <input type="password" name="password" placeholder="Password" required class="w-full mb-6 px-4 py-2 border rounded-lg">
            <button type="submit" class="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition">Login</button>
        </form>
        <p class="mt-4 text-center text-sm">Don't have an account? <a href="{{ url_for('signup') }}" class="text-indigo-600 hover:underline">Sign up</a></p>
    </div>
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
<body class="bg-indigo-100 flex items-center justify-center h-screen">
    <div class="bg-white p-8 rounded-xl shadow-md w-full max-w-sm">
        <h2 class="text-2xl font-bold mb-6 text-center text-indigo-600">Sign Up</h2>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required class="w-full mb-4 px-4 py-2 border rounded-lg">
            <input type="password" name="password" placeholder="Password" required class="w-full mb-6 px-4 py-2 border rounded-lg">
            <button type="submit" class="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-700 transition">Sign Up</button>
        </form>
        <p class="mt-4 text-center text-sm">Already have an account? <a href="{{ url_for('login') }}" class="text-indigo-600 hover:underline">Login</a></p>
    </div>
</body>
</html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Chat History</title>
    <link href=\"https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css\" rel=\"stylesheet\">
</head>
<body class=\"bg-gray-100 p-6\">
    <div class=\"max-w-4xl mx-auto bg-white p-6 rounded-lg shadow\">
        <h2 class=\"text-2xl font-bold mb-4 text-indigo-600\">Chat History</h2>
        <a href=\"/\" class=\"text-blue-600 hover:underline text-sm\">⬅ Back to Home</a>
        <div class=\"mt-4 space-y-4\">
            {% for row in history %}
                <div class=\"p-4 border border-gray-200 rounded-lg bg-gray-50\">
                    {% if is_admin %}
                        <p class=\"text-sm text-gray-700 mb-1\"><strong>User:</strong> {{ row[0] }}</p>
                        <p><strong>Q:</strong> {{ row[1] }}</p>
                        <p><strong>A:</strong> {{ row[2] }}</p>
                        <p class=\"text-xs text-gray-500 mt-1\">{{ row[3] }}</p>
                    {% else %}
                        <p><strong>Q:</strong> {{ row[0] }}</p>
                        <p><strong>A:</strong> {{ row[1] }}</p>
                        <p class=\"text-xs text-gray-500 mt-1\">{{ row[2] }}</p>
                    {% endif %}
                </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)

# Ensure the Tesseract OCR executable is in your PATH
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSER   ACT_CMD", "tesseract")
# Ensure the TTS engine is properly configured
pyttsx3.init()  
# Ensure the Flask app is running with the correct configurations
if not qa_bot.initialized:
    qa_bot.initialize()
# Ensure the database is initialized
# (Database is already initialized by init_db() above)