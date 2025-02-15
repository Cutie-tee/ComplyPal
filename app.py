import streamlit as st
from datetime import datetime
import pandas as pd
import json
import re
import logging
import time
import io
import os
import tempfile
import base64
import concurrent.futures
import speech_recognition as sr
from gtts import gTTS
from fpdf import FPDF
from PIL import Image
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
from audio_recorder_streamlit import audio_recorder

# Import shared configuration
from config import *

# -----------------------
# Initialize Session State
# -----------------------
def init_session_state():
    """Initialize session state variables with default values."""
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "assessment_result" not in st.session_state:
        st.session_state.assessment_result = ""
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    if "prev_theme" not in st.session_state:
        st.session_state.prev_theme = None
    if "theme_choice" not in st.session_state:
        st.session_state.theme_choice = "Light"

# Initialize session state
init_session_state()

# -----------------------
# Streamlit Page Configuration
# -----------------------

try:
    page_icon = Image.open("superhero.jpg")
except:
    page_icon = "🤖"

st.set_page_config(
    page_title="ComplyPal - AI Compliance Assistant",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Sidebar Configuration
# -----------------------
# In the sidebar section, replace the current reset button code with this:
with st.sidebar:
    try:
        st.image("superhero.jpg", width=100, use_container_width=True)
    except:
        st.title("ComplyPal")
    
    st.title("Settings")
    
    # Theme selection
    theme_choice = st.radio(
        "🎨 Interface Theme",
        options=["Light", "Dark"],
        horizontal=True,
        help="Choose your preferred theme",
        key="theme_radio"
    )
    st.session_state.theme_choice = theme_choice
    
    st.divider()
    
    if st.button("🔄 Start New Session", use_container_width=True):
        # Store the current input key
        current_key = st.session_state.input_key + 1
        # Clear all session state except theme
        for key in list(st.session_state.keys()):
            if key != "theme_choice":
                del st.session_state[key]
        # Reinitialize session state
        st.session_state.input_key = current_key
        st.session_state.responses = {}
        st.session_state.user_name = ""
        st.session_state.chat_history = []
        st.session_state.assessment_result = ""
        st.session_state.form_submitted = False
        st.session_state.prev_theme = None
        # Use Streamlit's rerun
        st.rerun()
    
    st.divider()
    
    with st.expander("ℹ️ About ComplyPal"):
        st.write("""
        ComplyPal is your AI-powered compliance assistant. 
        It helps you navigate complex compliance requirements
        and provides personalized guidance.
        
        Built with advanced AI technology, ComplyPal offers:
        - Compliance Assessment
        - Real-time Chat Support
        - Document Generation
        - Voice Interaction
        """)

# Function to get theme-specific styles
def get_theme_styles(theme):
    if theme == "Dark":
        return {
            'bg_color': '#1E1E1E',
            'text_color': '#FFFFFF',
            'primary_color': '#4A69A5',
            'secondary_color': '#3A3A3A',
            'success_color': '#4CAF50',
            'warning_color': '#FFC107',
            'error_color': '#F44336',
            'chat_bg': '#2D2D2D',
            'user_bubble': '#4A69A5',
            'bot_bubble': '#3A3A3A',
            'shadow': '0 4px 6px rgba(0,0,0,0.3)'
        }
    else:  # Light theme
        return {
            'bg_color': '#FFFFFF',
            'text_color': '#333333',
            'primary_color': '#1976D2',
            'secondary_color': '#F5F5F5',
            'success_color': '#4CAF50',
            'warning_color': '#FFC107',
            'error_color': '#F44336',
            'chat_bg': '#F8F9FA',
            'user_bubble': '#E3F2FD',
            'bot_bubble': '#F5F5F5',
            'shadow': '0 4px 6px rgba(0,0,0,0.1)'
        }

# Generate CSS based on theme
def get_css(theme):
    styles = get_theme_styles(theme)
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    /* Base Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {styles['bg_color']};
        color: {styles['text_color']};
    }}
    
    /* Logo Styles */
    .logo-container {{
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(to right, {styles['primary_color']}22, {styles['primary_color']}11);
        border-radius: 15px;
    }}
    
    .logo-container img {{
        border-radius: 12px;
        box-shadow: {styles['shadow']};
        transition: all 0.3s ease;
    }}
    
    .logo-container img:hover {{
        transform: scale(1.02);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
    }}
    
    /* Form Styles */
    .stTextInput, .stTextArea {{
        border-radius: 8px !important;
        border: 1px solid {styles['primary_color']}33 !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput:focus, .stTextArea:focus {{
        border-color: {styles['primary_color']} !important;
        box-shadow: 0 0 0 2px {styles['primary_color']}22 !important;
    }}
    
    /* Button Styles */
    .stButton > button {{
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        background-color: {styles['primary_color']} !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }}
    
    /* Chat Styles */
    .chat-container {{
        padding: 1.5rem;
        max-height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        background-color: {styles['chat_bg']};
        border-radius: 12px;
        box-shadow: {styles['shadow']};
        margin: 1rem 0;
        scroll-behavior: smooth;
    }}
    
    .chat-bubble {{
        padding: 12px 18px;
        border-radius: 15px;
        max-width: 80%;
        font-size: 1rem;
        line-height: 1.5;
        box-shadow: {styles['shadow']};
        margin: 4px 0;
        animation: fadeIn 0.3s ease;
    }}
    
    .chat-bubble.user {{
        background-color: {styles['user_bubble']};
        color: {styles['text_color']};
        align-self: flex-end;
        border-bottom-right-radius: 5px;
    }}
    
    .chat-bubble.bot {{
        background-color: {styles['bot_bubble']};
        color: {styles['text_color']};
        align-self: flex-start;
        border-bottom-left-radius: 5px;
    }}
    
    /* Animation */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Custom Components */
    .custom-info-box {{
        padding: 1rem;
        background-color: {styles['primary_color']}11;
        border-left: 4px solid {styles['primary_color']};
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    .custom-warning-box {{
        padding: 1rem;
        background-color: {styles['warning_color']}11;
        border-left: 4px solid {styles['warning_color']};
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        background-color: {styles['bg_color']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 1rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {styles['primary_color']}11;
    }}
    
    /* Improved form visuals */
    .question-container {{
        background-color: {styles['secondary_color']};
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid {styles['primary_color']}22;
    }}
    </style>
    """

# Apply theme-based styling
st.markdown(get_css(st.session_state.theme_choice), unsafe_allow_html=True)

# -----------------------
# Helper Functions
# -----------------------
def sanitize_text(text):
    """Sanitize text for PDF generation."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\u20ac', 'EUR')
    text = text.encode('latin-1', 'replace').decode('latin-1')
    return text

def create_pdf(content, title):
    """Create a PDF with proper content formatting and error handling."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    
    if isinstance(content, list):
        content = "\n".join(content)
    
    content = sanitize_text(content)
    if not content or not content.strip():
        content = "No content available."
    
    content = re.sub(r'<[^>]+>', '', content)
    
    lines = content.split('\n')
    for line in lines:
        while len(line) > 75:
            split_point = line[:75].rfind(' ')
            if split_point == -1:
                split_point = 75
            pdf.multi_cell(0, 10, line[:split_point])
            line = line[split_point:].strip()
        pdf.multi_cell(0, 10, line)
    
    return pdf.output(dest="S")

def rerun():
    """Helper function for re-running the app."""
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.write("Please refresh the page manually to apply changes.")

def transcribe_audio(audio_bytes):
    """Transcribe audio to text."""
    if audio_bytes is not None:
        audio_file = io.BytesIO(audio_bytes)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return ""
    return ""

def text_to_speech(text, lang="en"):
    """Convert text to speech."""
    tts = gTTS(text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def get_base64_audio(file_path):
    """Convert audio file to base64."""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def format_message(sender, message, timestamp):
    """Format chat messages with improved styling."""
    if sender == "user":
        return f"""<div class="chat-bubble user">
            <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.2rem">
                {st.session_state.user_name} • {timestamp}
            </div>
            {message}
        </div>"""
    else:
        return f"""<div class="chat-bubble bot">
            <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.2rem">
                ComplyPal • {timestamp}
            </div>
            {message}
        </div>"""

# -----------------------
# Vector DB Initialization
# -----------------------
@st.cache_resource(show_spinner=True)
def initialize_vector_db():
    """Initialize or load the vector database."""
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        st.info("Loading existing vector database...")
        embedding_function = GoogleGenerativeAIEmbeddings(
            **EMBEDDING_CONFIG,
            google_api_key=GOOGLE_API_KEY
        )
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

    st.info("Processing PDF documents and building vector database...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    if not os.path.exists(PDF_DIR):
        st.error(f"PDF directory '{PDF_DIR}' not found")
        return None
        
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    documents = []

    def process_pdf(pdf):
        pdf_path = os.path.join(PDF_DIR, pdf)
        loader = PDFPlumberLoader(pdf_path)
        raw_docs = loader.load()
        docs = []
        for doc in raw_docs:
            chunks = text_splitter.split_text(doc.page_content)
            docs.extend([Document(page_content=chunk, metadata={"source": pdf}) for chunk in chunks])
        return docs

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_pdf, pdf_files))
    documents = [doc for sublist in results for doc in sublist]

    if documents:
        embedding_function = GoogleGenerativeAIEmbeddings(
            **EMBEDDING_CONFIG,
            google_api_key=GOOGLE_API_KEY
        )
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            client=client
        )
        return vectorstore
    return None

# -----------------------
# Load Questionnaire
# -----------------------
@st.cache_data
def load_questionnaire():
    """Load and validate questionnaire data."""
    try:
        df = pd.read_csv("questionnaire.csv")
        required_columns = ['question', 'type', 'options', 'help_text', 'placeholder']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception as e:
        st.error(f"Error loading questionnaire: {str(e)}")
        return pd.DataFrame(columns=['question', 'type', 'category', 'options', 'help_text', 'placeholder'])

# -----------------------
# Display Logo
# -----------------------
#st.markdown('<div class="logo-container">', unsafe_allow_html=True)
#col1, col2, col3 = st.columns([1, 2, 1])
#with col2:
 #   try:
 #       st.image("superhero.jpg", width=200, use_container_width=True)
 #   except:
  #      st.warning("Logo image not found")
#st.markdown('</div>', unsafe_allow_html=True)

# Initialize Vector DB
with st.spinner("Initializing document database..."):
    retriever = initialize_vector_db()

# Initialize LLM
llm = GoogleGenerativeAI(
    **GEMINI_CONFIG,
    google_api_key=GOOGLE_API_KEY
)

# Load questionnaire
questionnaire = load_questionnaire()

# -----------------------
# Main Application Flow
# -----------------------
if "user_name" not in st.session_state or not st.session_state.user_name:
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>👋 Welcome to ComplyPal</h1>
            <p style='font-size: 1.2rem; opacity: 0.8;'>
                Your AI-powered compliance assistant
            </p>
        </div>
    """, unsafe_allow_html=True)

# User Name Input
current_user = st.session_state.get("user_name", "")
user_name_input = st.text_input(
    "Please enter your name to begin",
    value=current_user,
    key=f"name_input_{st.session_state.input_key}",
    placeholder="Type your name here...",
    help="Your name helps us personalize your experience"
)

if user_name_input != current_user:
    st.session_state.user_name = user_name_input
    st.session_state.chat_history = []
    st.session_state.input_key += 1
    rerun()

if not st.session_state.user_name:
    st.stop()

# Main Application Tabs
st.title(f"Welcome, {st.session_state.user_name} 👋")
tab1, tab2 = st.tabs(["📋 Compliance Assessment", "💬 Chat Assistant"])

# Tab 1: Compliance Assessment
with tab1:
    st.header("Compliance Assessment")
    with st.form("questionnaire_form"):
        st.markdown("### Please complete all fields below")
        
        empty_fields = []
        
        for index, row in questionnaire.iterrows():
            question = row['question']
            
            with st.container():
                st.markdown(f"""
                <div class="question-container">
                    <h4>Q{index + 1}: {question}</h4>
                """, unsafe_allow_html=True)
                
                if row['type'] == "textarea":
                    response = st.text_area(
                        "Your answer",
                        key=f"q_{index}",
                        help=row.get('help_text', "Provide a detailed answer"),
                        placeholder=row.get('placeholder', "Enter your detailed response here...")
                    )
                elif row['type'] == "radio":
                    options = json.loads(row['options'])
                    response = st.radio(
                        "Select one option",
                        options,
                        key=f"q_{index}",
                        help=row.get('help_text', "Choose the most appropriate option")
                    )
                elif row['type'] == "multiselect":
                    options = json.loads(row['options'])
                    response = st.multiselect(
                        "Select all that apply",
                        options,
                        key=f"q_{index}",
                        help=row.get('help_text', "You can select multiple options")
                    )
                else:
                    response = st.text_input(
                        "Your answer",
                        key=f"q_{index}",
                        help=row.get('help_text', "Enter your response"),
                        placeholder=row.get('placeholder', "Type your answer here...")
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state.responses[question] = response
            if not response:
                empty_fields.append(f"Question {index + 1}")
        
        # Submit button with loading animation
        submit_container = st.container()
        with submit_container:
            submit_button = st.form_submit_button(
                "Submit Assessment",
                use_container_width=True,
                help="Click to analyze your compliance status"
            )
        
        if submit_button:
            if empty_fields:
                st.error(f"⚠️ Please complete the following questions: {', '.join(empty_fields)}")
            else:
                st.session_state["form_submitted"] = True
                st.success("✅ Assessment submitted successfully!")

    # Process Assessment
    if st.session_state.get("form_submitted"):
        if all(st.session_state.responses.values()):
            user_responses = json.dumps(st.session_state.responses, indent=2)
            with st.spinner("🔄 Analyzing compliance..."):
                retrieved_docs = retriever.similarity_search(user_responses, k=RETRIEVAL_K) if retriever else []
                context = "\n\n".join([
                    f"📝 **Source:** {doc.metadata.get('source', 'Unknown PDF')}\n{doc.page_content}"
                    for doc in retrieved_docs
                ]) if retrieved_docs else "No relevant legal references found."
                
                prompt = f"""You are an AI compliance expert. Evaluate the following system based on the user responses and legal references.

### Instructions:
1. Cite relevant laws/standards
2. Identify compliance risks
3. Provide improvement suggestions
4. Outline non-compliance penalties
5. Suggest follow-up questions
6. Determine EU AI Act risk category

Context:
{context}

User Responses:
{user_responses}

Generate a structured compliance report."""

                response = llm.invoke(prompt)
                st.session_state.assessment_result = response
                
                st.markdown("""
                    <div class="custom-info-box">
                        <h3>📋 Compliance Assessment Report</h3>
                        <p>Based on your responses, here's your comprehensive compliance assessment:</p>
                    </div>
                """, unsafe_allow_html=True)
                st.write(response)
                
                # Add download button with improved styling
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    pdf_report = create_pdf(response, "Compliance Assessment Report")
                    st.download_button(
                        "📥 Download Full Report as PDF",
                        data=pdf_report,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

# Tab 2: Chat Assistant
with tab2:
    if st.session_state.get("assessment_result"):
        st.header("💬 Chat with ComplyPal")
        
        chat_history_container = st.empty()
        
        # Chat controls
        col1, col2 = st.columns([4, 1])
        with col1:
            chat_input_mode = st.radio(
                "Input Mode:",
                ("Text", "Speech"),
                horizontal=True,
                help="Choose how you want to communicate"
            )
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                rerun()
        
        # Follow-up suggestions
        if st.session_state.chat_history:
            with st.expander("💡 Suggested Questions", expanded=False):
                if st.button("Get Suggestions", use_container_width=True):
                    with st.spinner("Generating suggestions..."):
                        chat_history_text = "\n".join([re.sub(r'<[^>]+>', '', msg) for msg in st.session_state.chat_history])
                        compliance_text = st.session_state.assessment_result
                        
                        suggestion_prompt = f"""Based on this compliance report and chat history, suggest three specific follow-up questions:

Compliance Report:
{compliance_text}

Chat History:
{chat_history_text}

Generate three clear, specific questions that would help deepen the compliance understanding."""
                        
                        suggestions = llm.invoke(suggestion_prompt)
                        st.markdown("### You might want to ask:")
                        st.markdown(suggestions)
        
        def render_chat_history():
            with chat_history_container.container():
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    st.markdown(msg, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        render_chat_history()
        
        # Chat input handling
        if chat_input_mode == "Text":
            user_input = st.chat_input(
                "Type your question here...",
                key="chat_input"
            )
        else:
            st.info("🎤 Click the button below to record your message")
            audio_bytes = audio_recorder()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                with st.spinner("🎯 Transcribing..."):
                    user_input = transcribe_audio(audio_bytes)
                    if user_input:
                        st.success(f"🗣️ Transcribed: {user_input}")
                    else:
                        st.warning("❌ Could not transcribe audio. Please try again.")
                        user_input = None
            else:
                user_input = None
        
        if user_input:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M")
            user_msg = format_message("user", user_input, timestamp)
            st.session_state.chat_history.append(user_msg)
            render_chat_history()
            
            # Generate AI response
            context = st.session_state.assessment_result
            with st.spinner("🤔 Analyzing your question..."):
                chat_prompt = f"""You are a friendly, professional compliance assistant. 
                Address the user as {st.session_state.user_name} and use the compliance report as context.

User Question: {user_input}

Compliance Context: {context}

Please provide a clear, helpful response focusing on compliance aspects when relevant. 
Be concise but thorough, and use a friendly, professional tone."""
                
                chat_response = llm.invoke(chat_prompt)
                bot_msg = format_message("bot", chat_response, datetime.now().strftime("%H:%M"))
                st.session_state.chat_history.append(bot_msg)
                render_chat_history()
            
            # Generate and play audio response
            audio_file = text_to_speech(chat_response)
            audio_base64 = get_base64_audio(audio_file)
            audio_html = f"""
            <audio autoplay controls style="display:none;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        
        # Download options
        if st.session_state.chat_history or st.session_state.assessment_result:
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.session_state.assessment_result:
                    pdf_report = create_pdf(st.session_state.assessment_result, "Compliance Report")
                    st.download_button(
                        "📥 Download Compliance Report",
                        data=pdf_report,
                        file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download your compliance assessment as PDF",
                        use_container_width=True
                    )
            
            with col2:
                if st.session_state.chat_history:
                    chat_text = []
                    for msg in st.session_state.chat_history:
                        clean_msg = re.sub(r'<[^>]+>', '', msg)
                        clean_msg = clean_msg.replace('&nbsp;', ' ').strip()
                        if clean_msg:
                            chat_text.append(clean_msg)
                    
                    chat_history_text = "\n\n".join(chat_text)
                    pdf_chat_history = create_pdf(chat_history_text, "Chat History")
                    st.download_button(
                        "📥 Download Chat History",
                        data=pdf_chat_history,
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download your conversation history as PDF",
                        use_container_width=True
                    )
    else:
        st.info("👆 Please complete the compliance assessment first to enable chat support.")