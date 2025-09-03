from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from dotenv import load_dotenv
import os
import json
import sys
import locale
import traceback

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Set locale for proper text handling
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except:
        pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for the system
soybot_agent = None
knowledge_base = None
is_initialized = False

def initialize_soybot():
    """Initialize SoyBot with PDF-only knowledge base"""
    global soybot_agent, knowledge_base, is_initialized
    
    try:
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise Exception("GROQ_API_KEY not found in environment variables")
        
        print("🔄 Setting up knowledge base...")
        # Knowledge Base setup
        knowledge_base = PDFKnowledgeBase(
            path="Soybeanpackageofpractices.pdf",
            vector_db=LanceDb(
                table_name="soybean_practices",
                uri="./vectordb/soybot_db",
                search_type=SearchType.vector,
                embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2"),
                nprobes=10,
                distance="cosine"
            )
        )
        
        print("🔄 Loading knowledge base...")
        knowledge_base.load(recreate=False)
        print("✅ Knowledge base loaded successfully!")
        
        print("🔄 Creating Main SoyBot Agent...")
        # Main SoyBot Agent
        soybot_agent = Agent(
            name="Multilingual SoyBot",
            role="Expert soybean farming advisor with multilingual support",
            model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
            knowledge=knowledge_base,
            instructions=[
                "You are an expert soybean farming advisor with deep knowledge of Indian farming practices.",
                "You can understand questions in Hindi (हिंदी), English, and Marathi (मराठी).",
                "Always respond in the SAME language as the question was asked.",
                "CRITICAL: Use ONLY the content from the provided PDF to answer questions.",
                "Do NOT use any external knowledge or information not present in the PDF.",
                "If the PDF doesn't contain information about the specific question, clearly state: 'This information is not available in the provided soybean farming guide.'",
                "Do NOT make up information or provide general farming advice not in the PDF.",
                "Use simple, clear language suitable for farmers.",
                "Provide practical, actionable farming advice that farmers can implement based on the PDF.",
                "Format responses clearly with proper structure using headings and bullet points when helpful.",
                "Be comprehensive but concise in your answers.",
                "Include relevant details like timing, quantities, methods, and precautions as specified in the PDF.",
                "If you don't have specific information in the PDF, clearly state this and avoid guessing.",
                "Keep responses conversational and easy to understand when spoken aloud.",
                "Use shorter sentences when possible for better text-to-speech clarity.",
            ],
            show_tool_calls=False,
            markdown=True
        )
        
        is_initialized = True
        print("✅ SoyBot initialized successfully!")
        print("   - PDF Knowledge Base: Loaded")
        print("   - Multilingual Support: Ready")
        print("   - PDF-Only Responses: Enforced")
        print("   - Speech Integration: Ready")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing SoyBot: {str(e)}")
        print(traceback.format_exc())
        is_initialized = False
        return False

def detect_language(text):
    """Detect the language of input text"""
    # Check for Devanagari script (Hindi)
    if any('\u0900' <= char <= '\u097F' for char in text):
        return 'hindi'
    # Check for basic English
    elif text.isascii():
        return 'english'
    else:
        return 'unknown'

def format_response(response_text, language):
    """Format response text properly for display and speech"""
    try:
        # Clean up any encoding issues
        if isinstance(response_text, bytes):
            response_text = response_text.decode('utf-8')
        
        # Ensure proper line breaks and formatting
        formatted_text = response_text.strip()
        
        # Add language-specific formatting improvements
        if language == 'hindi':
            # Ensure proper spacing around Hindi text
            formatted_text = formatted_text.replace('।', '। ')
            formatted_text = formatted_text.replace(':', ': ')
        
        # Clean up markdown for better speech synthesis
        # Remove excessive markdown formatting that doesn't help with speech
        formatted_text = formatted_text.replace('**', '')
        formatted_text = formatted_text.replace('*', '')
        formatted_text = formatted_text.replace('###', '')
        formatted_text = formatted_text.replace('##', '')
        formatted_text = formatted_text.replace('#', '')
        
        return formatted_text
    except Exception as e:
        return f"Response formatting error: {str(e)}"

# Enhanced HTML template with speech integration
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SoyBot - Voice-Enabled AI Farming Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 900px;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #2e7d32, #4caf50);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .header .subtitle {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 5px;
        }

        .hindi-text {
            font-family: 'Noto Sans Devanagari', 'Arial Unicode MS', sans-serif;
            font-size: 1.2rem;
            margin-top: 10px;
        }

        .agents-status {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            margin-top: 15px;
            border-radius: 10px;
            font-size: 0.9rem;
        }

        .speech-controls {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            margin-top: 10px;
            border-radius: 10px;
            font-size: 0.9rem;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .speech-btn {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .speech-btn:hover {
            background: rgba(255,255,255,0.3);
        }

        .speech-btn.active {
            background: #ff5722;
            border-color: #ff5722;
        }

        .language-btn {
            background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.3s;
        }

        .language-btn.active {
            background: #2196f3;
            border-color: #2196f3;
        }

        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 30px;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            gap: 15px;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message-content {
            background: white;
            padding: 15px 20px;
            border-radius: 18px;
            max-width: 70%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 1rem;
            line-height: 1.5;
            white-space: pre-wrap;
            position: relative;
        }

        .message.user .message-content {
            background: #2e7d32;
            color: white;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
        }

        .user .message-avatar {
            background: #1976d2;
        }

        .bot .message-avatar {
            background: #4caf50;
        }

        .speak-btn {
            position: absolute;
            top: 10px;
            right: 10px;
            background: #4caf50;
            color: white;
            border: none;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            cursor: pointer;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.3s;
        }

        .speak-btn:hover {
            background: #2e7d32;
        }

        .speak-btn.speaking {
            background: #ff5722;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }

        .input-container {
            padding: 25px 30px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .input-box {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
            font-family: 'Noto Sans Devanagari', 'Arial Unicode MS', sans-serif;
        }

        .input-box:focus {
            border-color: #4caf50;
        }

        .mic-btn {
            background: #ff5722;
            color: white;
            border: none;
            padding: 15px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.2rem;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }

        .mic-btn:hover {
            background: #e64a19;
        }

        .mic-btn.listening {
            background: #4caf50;
            animation: pulse 1s infinite;
        }

        .send-btn {
            background: #4caf50;
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: bold;
            transition: background 0.3s;
        }

        .send-btn:hover {
            background: #2e7d32;
        }

        .send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .sample-questions {
            padding: 20px 30px;
            background: #f5f5f5;
            border-top: 1px solid #e0e0e0;
        }

        .sample-questions h3 {
            margin-bottom: 15px;
            color: #2e7d32;
            font-size: 1.1rem;
        }

        .question-btn {
            display: inline-block;
            background: white;
            border: 1px solid #4caf50;
            color: #2e7d32;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s;
            font-family: 'Noto Sans Devanagari', 'Arial Unicode MS', sans-serif;
        }

        .question-btn:hover {
            background: #4caf50;
            color: white;
        }

        .typing-indicator {
            display: none;
            padding: 10px 0;
            font-style: italic;
            color: #666;
            text-align: center;
        }

        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #c62828;
        }

        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
        }

        .status-online {
            background: #4caf50;
            color: white;
        }

        .status-offline {
            background: #f44336;
            color: white;
        }

        .speech-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            background: #2196f3;
            color: white;
            display: none;
        }
    </style>
</head>
<body>
    <div class="status-indicator" id="statusIndicator">🔄 Connecting...</div>
    <div class="speech-status" id="speechStatus">🎤 Listening...</div>
    
    <div class="container">
        <div class="header">
            <h1>🤖 SoyBot</h1>
            <p>Voice-Enabled Soybean Farming Expert</p>
            <div class="subtitle">Multilingual AI Assistant with Speech</div>
            <div class="hindi-text">आवाज़ से बोलें - सोयाबीन खेती सलाहकार</div>
            <div class="agents-status" id="agentsStatus">
                🔄 Initializing SoyBot...
            </div>
            <div class="speech-controls">
    <button class="speech-btn" id="autoSpeakBtn" onclick="toggleAutoSpeak()">🔊 Auto-Speak: ON</button>
    <button class="speech-btn" onclick="testSpeech()">🎵 Test Speech</button>
    <button class="speech-btn" id="stopSpeakBtn" onclick="stopSpeaking()">⏹️ Stop Audio</button>
    <button class="language-btn" id="hindiBtn" onclick="setLanguage('hi-IN')">🇮🇳 हिंदी</button>
    <button class="language-btn" id="englishBtn" onclick="setLanguage('en-US')">🇺🇸 English</button>
    <select id="voiceSelect" class="speech-btn" onchange="setVoice()">
        <option value="">Default Voice</option>
    </select>
</div>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message bot">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <strong>नमस्कार! Welcome to Voice-Enabled SoyBot!</strong>

I'm your soybean farming expert with speech capabilities:
- 🎤 Click the microphone to speak your questions
- 🔊 I'll speak my answers back to you
- ⏹️ Click "Stop Audio" to stop speech anytime
- 📚 All answers based strictly on soybean farming PDF
- 🌐 Support for हिंदी, English, and मराठी
- 🎯 No external knowledge - only PDF content

Try speaking: "सोयाबीन की बुवाई कब करें?" or "When to sow soybean?"
Keyboard shortcuts: Ctrl+M (mic), Ctrl+S (stop audio)
                    <button class="speak-btn" onclick="speakMessage(this)">🔊</button>
                </div>
            </div>
        </div>

        <div class="sample-questions">
            <h3>Sample Questions / नमूना प्रश्न (Click to ask or speak them):</h3>
            <span class="question-btn" onclick="askQuestion('सोयाबीन की बुवाई का सबसे अच्छा समय क्या है?')">सोयाबीन की बुवाई का समय?</span>
            <span class="question-btn" onclick="askQuestion('What is the best time for soybean sowing?')">Best sowing time?</span>
            <span class="question-btn" onclick="askQuestion('सोयाबीन में कौन से रोग लगते हैं?')">सोयाबीन के रोग?</span>
            <span class="question-btn" onclick="askQuestion('How much fertilizer should be used for soybean?')">Fertilizer amount?</span>
            <span class="question-btn" onclick="askQuestion('सोयाबीन को कितना पानी चाहिए?')">पानी की आवश्यकता?</span>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            🤖 SoyBot is thinking...
        </div>

        <div class="input-container">
            <button class="mic-btn" id="micBtn" onclick="toggleListening()">🎤</button>
            <input type="text" 
                   id="questionInput" 
                   class="input-box" 
                   placeholder="Type or speak your question... टाइप करें या बोलें..."
                   onkeypress="handleKeyPress(event)">
            <button class="send-btn" onclick="sendQuestion()" id="sendBtn">Send</button>
        </div>
    </div>

    <script>
        // Global variables for speech functionality
        let recognition = null;
        let isListening = false;
        let autoSpeak = true;
        let currentVoice = null;
        let speechSynthesis = window.speechSynthesis;
        let availableVoices = [];
        let recognitionLanguage = 'hi-IN'; // Default to Hindi

        // DOM elements
        const chatContainer = document.getElementById('chatContainer');
        const questionInput = document.getElementById('questionInput');
        const sendBtn = document.getElementById('sendBtn');
        const micBtn = document.getElementById('micBtn');
        const typingIndicator = document.getElementById('typingIndicator');
        const statusIndicator = document.getElementById('statusIndicator');
        const agentsStatus = document.getElementById('agentsStatus');
        const speechStatus = document.getElementById('speechStatus');
        const autoSpeakBtn = document.getElementById('autoSpeakBtn');
        const voiceSelect = document.getElementById('voiceSelect');
        const hindiBtn = document.getElementById('hindiBtn');
        const englishBtn = document.getElementById('englishBtn');

        // Set recognition language
        function setLanguage(lang) {
            recognitionLanguage = lang;
            recognition.lang = lang;
            
            // Update UI
            hindiBtn.classList.toggle('active', lang === 'hi-IN');
            englishBtn.classList.toggle('active', lang === 'en-US');
            
            // Set status message
            speechStatus.textContent = lang === 'hi-IN' 
                ? 'भाषा हिंदी में सेट की गई' 
                : 'Language set to English';
            speechStatus.style.display = 'block';
            setTimeout(() => {
                speechStatus.style.display = 'none';
            }, 2000);
        }

        // Initialize speech recognition
        function initSpeechRecognition() {
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                
                // Set initial language
                recognition.lang = recognitionLanguage;
                
                recognition.onstart = function() {
                    console.log('Speech recognition started');
                    isListening = true;
                    micBtn.classList.add('listening');
                    speechStatus.style.display = 'block';
                    speechStatus.textContent = recognitionLanguage === 'hi-IN' 
                        ? '🎤 सुन रहा हूं...' 
                        : '🎤 Listening...';
                };
                
                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    console.log('Speech recognized:', transcript);
                    
                    // Check for voice commands first
                    if (handleVoiceCommand(transcript)) {
                        speechStatus.textContent = recognitionLanguage === 'hi-IN' 
                            ? '✅ कमांड क्रियान्वित!' 
                            : '✅ Command executed!';
                        setTimeout(() => {
                            speechStatus.style.display = 'none';
                        }, 2000);
                        return;
                    }
                    
                    // Otherwise, use as regular input
                    questionInput.value = transcript;
                    speechStatus.textContent = recognitionLanguage === 'hi-IN' 
                        ? '✅ समझ गया!' 
                        : '✅ Got it!';
                    setTimeout(() => {
                        speechStatus.style.display = 'none';
                    }, 2000);
                };
                
                recognition.onerror = function(event) {
                    console.error('Speech recognition error:', event.error);
                    speechStatus.textContent = '❌ Error: ' + event.error;
                    setTimeout(() => {
                        speechStatus.style.display = 'none';
                    }, 3000);
                };
                
                recognition.onend = function() {
                    console.log('Speech recognition ended');
                    isListening = false;
                    micBtn.classList.remove('listening');
                };
                
                console.log('Speech recognition initialized');
            } else {
                console.log('Speech recognition not supported');
                micBtn.style.display = 'none';
            }
        }

        // Initialize text-to-speech
        function initTextToSpeech() {
            if ('speechSynthesis' in window) {
                speechSynthesis.onvoiceschanged = function() {
                    availableVoices = speechSynthesis.getVoices();
                    populateVoiceSelect();
                    
                    // Set default voice to first Hindi voice if available
                    const hindiVoices = availableVoices.filter(v => v.lang.includes('hi'));
                    if (hindiVoices.length > 0) {
                        currentVoice = hindiVoices[0];
                        voiceSelect.value = availableVoices.indexOf(hindiVoices[0]);
                    }
                };
                
                // Load voices
                availableVoices = speechSynthesis.getVoices();
                if (availableVoices.length > 0) {
                    populateVoiceSelect();
                }
                
                console.log('Text-to-speech initialized');
            } else {
                console.log('Text-to-speech not supported');
            }
        }

        // Populate voice selection dropdown
        function populateVoiceSelect() {
            voiceSelect.innerHTML = '<option value="">Default Voice</option>';
            
            availableVoices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                voiceSelect.appendChild(option);
            });
        }

        // Set selected voice
        function setVoice() {
            const selectedIndex = voiceSelect.value;
            if (selectedIndex !== '') {
                currentVoice = availableVoices[selectedIndex];
                console.log('Voice set to:', currentVoice.name);
                
                // Show confirmation
                speechStatus.textContent = `Voice set to: ${currentVoice.name}`;
                speechStatus.style.display = 'block';
                setTimeout(() => {
                    speechStatus.style.display = 'none';
                }, 2000);
            } else {
                currentVoice = null;
            }
        }

        // Toggle listening
        function toggleListening() {
            if (!recognition) {
                alert('Speech recognition not supported in this browser');
                return;
            }
            
            if (isListening) {
                recognition.stop();
            } else {
                recognition.start();
            }
        }

        // Speak text
        function speakText(text) {
            if (!('speechSynthesis' in window)) {
                console.log('Text-to-speech not supported');
                return;
            }
            
            // Stop any current speech
            speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Set language based on detected content
            if (text.match(/[\u0900-\u097F]/)) {
                utterance.lang = 'hi-IN';
            } else {
                utterance.lang = 'en-US';
            }
            
            // Set voice if selected
            if (currentVoice) {
                utterance.voice = currentVoice;
            }
            
            // Set speech parameters
            utterance.rate = 0.9; // Slightly faster than before
            utterance.pitch = 1;
            utterance.volume = 1;
            
            utterance.onstart = function() {
                console.log('Speech synthesis started');
            };
            
            utterance.onend = function() {
                console.log('Speech synthesis ended');
            };
            
            utterance.onerror = function(event) {
                console.error('Speech synthesis error:', event.error);
            };
            
            speechSynthesis.speak(utterance);
        }

        // Speak message from button
        function speakMessage(button) {
            const messageContent = button.parentElement;
            const text = messageContent.textContent.replace('🔊', '').trim();
            
            button.classList.add('speaking');
            speakText(text);
            
            // Remove speaking class after a delay
            setTimeout(() => {
                button.classList.remove('speaking');
            }, 3000);
        }

        // Toggle auto-speak
        function toggleAutoSpeak() {
            autoSpeak = !autoSpeak;
            autoSpeakBtn.textContent = autoSpeak ? '🔊 Auto-Speak: ON' : '🔊 Auto-Speak: OFF';
            autoSpeakBtn.classList.toggle('active', autoSpeak);
        }

        // Test speech
        function testSpeech() {
            const testText = "नमस्कार! मैं सोयबॉट हूं। Hello! I am SoyBot.";
            speakText(testText);
        }

        // Check server status
        function checkServerStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'ready') {
                        statusIndicator.textContent = '🟢 SoyBot Online';
                        statusIndicator.className = 'status-indicator status-online';
                        agentsStatus.innerHTML = '✅ SoyBot Ready<br>📚 PDF Knowledge Loaded<br>🎤 Speech Ready';
                    } else {
                        statusIndicator.textContent = '🟡 Initializing...';
                        statusIndicator.className = 'status-indicator status-offline';
                        agentsStatus.textContent = '🔄 Loading SoyBot...';
                    }
                })
                .catch(error => {
                    statusIndicator.textContent = '🔴 Offline';
                    statusIndicator.className = 'status-indicator status-offline';
                    agentsStatus.textContent = '❌ SoyBot Offline';
                });
        }

        // Stop speaking function
function stopSpeaking() {
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
        console.log('Speech stopped by user');
        
        // Update all speaking buttons
        const speakBtns = document.querySelectorAll('.speak-btn.speaking');
        speakBtns.forEach(btn => btn.classList.remove('speaking'));
        
        // Show confirmation
        speechStatus.textContent = recognitionLanguage === 'hi-IN' 
            ? '⏹️ आवाज़ बंद की गई' 
            : '⏹️ Audio stopped';
        speechStatus.style.display = 'block';
        setTimeout(() => {
            speechStatus.style.display = 'none';
        }, 2000);
    }
}

        // Add message to chat
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = isUser ? '👤' : '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.textContent = content;
            
            // Add speak button for bot messages
            if (!isUser) {
                const speakBtn = document.createElement('button');
                speakBtn.className = 'speak-btn';
                speakBtn.textContent = '🔊';
                speakBtn.onclick = function() { speakMessage(this); };
                messageContent.appendChild(speakBtn);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Auto-speak bot responses
            if (!isUser && autoSpeak) {
                setTimeout(() => {
                    speakText(content);
                }, 500);
            }
        }

        // Show/hide typing indicator
        function showTyping() {
            typingIndicator.style.display = 'block';
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function hideTyping() {
            typingIndicator.style.display = 'none';
        }

        // Send question
        async function sendQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            // Add user message
            addMessage(question, true);
            questionInput.value = '';
            
            // Disable send button and show typing
            sendBtn.disabled = true;
            sendBtn.textContent = 'Processing...';
            showTyping();
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();
                
                hideTyping();
                
                if (data.success) {
                    addMessage(data.response);
                } else {
                    addMessage(`Error: ${data.error}`, false);
                }
                
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered a network error. Please try again later.', false);
                console.error('Error:', error);
            } finally {
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
            }
        }

        // Ask predefined question
        function askQuestion(question) {
            questionInput.value = question;
            sendQuestion();
        }

        // Handle key press
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuestion();
            }
        }

        // Initialize everything when page loads
        window.onload = function() {
            console.log('Initializing SoyBot with speech capabilities...');
            
            // Initialize speech features
            initSpeechRecognition();
            initTextToSpeech();
            
            // Set initial language
            setLanguage(recognitionLanguage);
            
            // Check server status
            checkServerStatus();
            
            // Focus on input
            questionInput.focus();
            
            // Add keyboard shortcuts
            document.addEventListener('keydown', function(event) {
                // Ctrl/Cmd + M for microphone
                if ((event.ctrlKey || event.metaKey) && event.key === 'm') {
                    event.preventDefault();
                    toggleListening();
                }
                
                // Ctrl/Cmd + S for stop speech
                if ((event.ctrlKey || event.metaKey) && event.key === 's') {
                    event.preventDefault();
                    speechSynthesis.cancel();
                }
            });
            
            console.log('SoyBot speech integration initialized');
            console.log('Keyboard shortcuts:');
            console.log('- Ctrl/Cmd + M: Toggle microphone');
            console.log('- Ctrl/Cmd + S: Stop speech');
        };

        // Periodic status check
        setInterval(checkServerStatus, 30000); // Check every 30 seconds
        
        // Handle page visibility change (pause speech when tab is hidden)
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                speechSynthesis.cancel();
                if (recognition && isListening) {
                    recognition.stop();
                }
            }
        });
        
        // Add some voice commands for better UX
        function handleVoiceCommand(transcript) {
            const command = transcript.toLowerCase();
            
            // Check for voice commands
            if (command.includes('clear') || command.includes('साफ करें')) {
                chatContainer.innerHTML = '';
                return true;
            }
            
            if (command.includes('stop speaking') || command.includes('बंद करें')) {
                speechSynthesis.cancel();
                return true;
            }
            
            if (command.includes('repeat') || command.includes('दोहराएं')) {
                const lastBotMessage = chatContainer.querySelector('.message.bot:last-child .message-content');
                if (lastBotMessage) {
                    const text = lastBotMessage.textContent.replace('🔊', '').trim();
                    speakText(text);
                }
                return true;
            }
            
            if (command.includes('hindi') || command.includes('हिंदी')) {
                setLanguage('hi-IN');
                return true;
            }
            
            if (command.includes('english') || command.includes('अंग्रेजी')) {
                setLanguage('en-US');
                return true;
            }
            
            return false;
        }
    </script>
</body>
</html>
"""

# Routes
@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server and SoyBot status"""
    global is_initialized
    return jsonify({
        'status': 'ready' if is_initialized else 'initializing',
        'message': 'SoyBot is ready' if is_initialized else 'SoyBot is initializing...',
        'features': {
            'speech_recognition': True,
            'text_to_speech': True,
            'multilingual': True,
            'pdf_knowledge': True
        }
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle question requests"""
    global soybot_agent, is_initialized
    
    try:
        # Check if SoyBot is initialized
        if not is_initialized or soybot_agent is None:
            return jsonify({
                'success': False,
                'error': 'SoyBot is still initializing. Please wait a moment and try again.'
            }), 503
        
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'No question provided'
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'success': False,
                'error': 'Empty question provided'
            }), 400
        
        # Detect language for better error handling
        language = detect_language(question)
        
        print(f"🤖 Processing question: {question}")
        print(f"🌐 Detected language: {language}")
        
        # Get response from SoyBot
        response = soybot_agent.run(question)
        
        # Extract response content
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Format the response for both display and speech
        formatted_response = format_response(response_text, language)
        
        print(f"✅ Response generated successfully")
        
        return jsonify({
            'success': True,
            'response': formatted_response,
            'language': language,
            'speech_enabled': True
        })
        
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        print(traceback.format_exc())
        
        # Language-appropriate error messages
        if 'language' in locals() and language == 'hindi':
            error_msg = f"क्षमा करें, तकनीकी समस्या है। कृपया बाद में कोशिश करें।"
        else:
            error_msg = f"Sorry, there's a technical issue. Please try again later."
        
        return jsonify({
            'success': False,
            'error': error_msg,
            'details': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'soybot_initialized': is_initialized,
        'speech_features': {
            'recognition': True,
            'synthesis': True,
            'multilingual': True
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🌱 Starting Voice-Enabled SoyBot Flask Server...")
    print("🔄 Initializing SoyBot with Speech Integration...")
    
    # Initialize SoyBot
    if initialize_soybot():
        print("✅ SoyBot initialized successfully!")
        print("   📚 PDF Knowledge Base: Loaded")
        print("   🌐 Multilingual Support: Active")
        print("   🎯 PDF-Only Responses: Enforced")
        print("   🎤 Speech Recognition: Ready")
        print("   🔊 Text-to-Speech: Ready")
        print("   ⌨️  Keyboard Shortcuts: Enabled")
        print("🚀 Starting Flask server...")
        print("🌐 Access the web interface at: http://localhost:5000")
        print("📡 API endpoints available at: http://localhost:5000/api/")
        print("🎤 Speech Features:")
        print("   - Click microphone button to speak")
        print("   - Auto-speak responses (toggle on/off)")
        print("   - Voice commands: 'clear', 'stop speaking', 'repeat'")
        print("   - Keyboard shortcuts: Ctrl+M (mic), Ctrl+S (stop speech)")
        print("   - Multilingual speech support")
        print("-" * 60)
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to True for development
            threaded=True
        )
    else:
        print("❌ Failed to initialize SoyBot. Please check your configuration.")
        print("🔍 Common issues:")
        print("   - Missing GROQ_API_KEY in .env file")
        print("   - Missing Soybeanpackageofpractices.pdf file")
        print("   - Network connectivity issues")
        sys.exit(1)