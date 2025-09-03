# soybot_gui.py - Desktop GUI version with proper Hindi support
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.vectordb.search import SearchType
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from dotenv import load_dotenv
import os

class SoyBotGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_soybot()
        self.create_widgets()
        
    def setup_window(self):
        """Setup main window"""
        self.root.title("🌱 SoyBot - Multilingual Farming Assistant")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f8ff')
        
        # Set font for Hindi support
        self.hindi_font = ('Noto Sans Devanagari', 12)
        self.english_font = ('Segoe UI', 11)
        self.header_font = ('Segoe UI', 16, 'bold')
        
    def setup_soybot(self):
        """Initialize SoyBot agent"""
        try:
            load_dotenv()
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if not groq_api_key:
                messagebox.showerror("Error", "GROQ_API_KEY not found in environment variables")
                return
            
            # Knowledge Base setup
            self.knowledge_base = PDFKnowledgeBase(
                path="Soybeanpackageofpractices.pdf",
                vector_db=LanceDb(
                    table_name="soybean_practices",
                    uri="./vectordb/soybot_db",
                    search_type=SearchType.vector,
                    embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2"),
                )
            )
            
            # Load knowledge base in background
            self.knowledge_base.load(recreate=False)
            
            # Create agent
            self.soybot_agent = Agent(
                name="Multilingual SoyBot",
                role="Expert soybean farming advisor",
                model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
                knowledge=self.knowledge_base,
                instructions=[
                    "You are an expert soybean farming advisor.",
                    "You can understand questions in Hindi, English, and Marathi.",
                    "Always respond in the SAME language as the question was asked.",
                    "Use simple, clear language suitable for farmers.",
                    "Provide practical, actionable farming advice.",
                    "Format responses with clear structure.",
                ],
                show_tool_calls=False,
                markdown=False  # Disable markdown for GUI
            )
            
            self.soybot_ready = True
            
        except Exception as e:
            messagebox.showerror("Setup Error", f"Failed to initialize SoyBot: {str(e)}")
            self.soybot_ready = False
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Header
        header_frame = tk.Frame(self.root, bg='#2e7d32', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="🌱 SoyBot - Multilingual Farming Assistant",
            font=self.header_font,
            bg='#2e7d32',
            fg='white'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            header_frame,
            text="सोयाबीन खेती सलाहकार | Languages: हिंदी, English, मराठी",
            font=self.hindi_font,
            bg='#2e7d32',
            fg='#e8f5e8'
        )
        subtitle_label.pack()
        
        # Chat area
        chat_frame = tk.Frame(self.root, bg='#f0f8ff')
        chat_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=self.hindi_font,
            bg='white',
            fg='black',
            state='disabled'
        )
        self.chat_display.pack(fill='both', expand=True, pady=(0, 10))
        
        # Configure text tags for styling
        self.chat_display.tag_configure("user", foreground="#1976d2", font=self.english_font)
        self.chat_display.tag_configure("bot", foreground="#2e7d32", font=self.hindi_font)
        self.chat_display.tag_configure("system", foreground="#666666", font=self.english_font, justify='center')
        
        # Input area
        input_frame = tk.Frame(chat_frame, bg='#f0f8ff')
        input_frame.pack(fill='x')
        
        # Question input
        self.question_var = tk.StringVar()
        self.question_entry = tk.Entry(
            input_frame,
            textvariable=self.question_var,
            font=self.hindi_font,
            width=60
        )
        self.question_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.question_entry.bind('<Return>', self.send_question)
        
        # Send button
        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_question,
            bg='#4caf50',
            fg='white',
            font=self.english_font,
            width=10
        )
        self.send_button.pack(side='right')
        
        # Sample questions frame
        samples_frame = tk.LabelFrame(
            self.root,
            text="Sample Questions / नमूना प्रश्न",
            font=self.english_font,
            bg='#f0f8ff',
            fg='#2e7d32'
        )
        samples_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Sample question buttons
        sample_questions = [
            ("सोयाबीन की बुवाई का समय?", "सोयाबीन की बुवाई का सबसे अच्छा समय क्या है?"),
            ("Best sowing time?", "What is the best time for soybean sowing?"),
            ("सोयाबीन के रोग?", "सोयाबीन में कौन से रोग लगते हैं?"),
            ("Fertilizer amount?", "How much fertilizer should be used for soybean?"),
            ("पानी की आवश्यकता?", "सोयाबीन की फसल को कितना पानी चाहिए?")
        ]
        
        for i, (button_text, question) in enumerate(sample_questions):
            btn = tk.Button(
                samples_frame,
                text=button_text,
                command=lambda q=question: self.ask_sample_question(q),
                bg='white',
                fg='#2e7d32',
                font=self.hindi_font,
                relief='ridge',
                borderwidth=1
            )
            btn.pack(side='left', padx=5, pady=5, fill='x', expand=True)
        
        # Initial welcome message
        self.add_message("🤖 SoyBot", "नमस्कार! Welcome to SoyBot!\n\nI can help you with soybean farming questions in:\n• हिंदी (Hindi)\n• English\n• मराठी (Marathi)\n\nPlease ask me anything about soybean cultivation!", "bot")
        
        # Focus on input
        self.question_entry.focus()
        
    def add_message(self, sender, message, msg_type="system"):
        """Add message to chat display"""
        self.chat_display.config(state='normal')
        
        # Add timestamp and sender
        timestamp = time.strftime("%H:%M")
        if msg_type == "user":
            self.chat_display.insert(tk.END, f"\n👤 You ({timestamp}):\n", "user")
        elif msg_type == "bot":
            self.chat_display.insert(tk.END, f"\n🤖 SoyBot ({timestamp}):\n", "bot")
        
        # Add message content
        self.chat_display.insert(tk.END, f"{message}\n", msg_type)
        self.chat_display.insert(tk.END, "-" * 60 + "\n", "system")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
    def ask_sample_question(self, question):
        """Ask a sample question"""
        self.question_var.set(question)
        self.send_question()
        
    def send_question(self, event=None):
        """Send question to SoyBot"""
        question = self.question_var.get().strip()
        if not question:
            return
            
        if not self.soybot_ready:
            messagebox.showerror("Error", "SoyBot is not ready. Please check your setup.")
            return
        
        # Clear input
        self.question_var.set("")
        
        # Add user message
        self.add_message("You", question, "user")
        
        # Disable send button
        self.send_button.config(state='disabled', text='Thinking...')
        
        # Process question in background thread
        thread = threading.Thread(target=self.process_question, args=(question,))
        thread.daemon = True
        thread.start()
        
    def process_question(self, question):
        """Process question with SoyBot"""
        try:
            # Get response from SoyBot
            response = self.soybot_agent.run(question)
            
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # Update GUI in main thread
            self.root.after(0, self.show_response, answer)
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self.root.after(0, self.show_response, error_msg)
    
    def show_response(self, response):
        """Show SoyBot response"""
        self.add_message("SoyBot", response, "bot")
        
        # Enable send button
        self.send_button.config(state='normal', text='Send')
        self.question_entry.focus()

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Try to set Hindi font support
    try:
        root.option_add('*Font', 'Noto\ Sans\ Devanagari 12')
    except:
        try:
            root.option_add('*Font', 'Arial\ Unicode\ MS 12')
        except:
            pass  # Use default font
    
    app = SoyBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()