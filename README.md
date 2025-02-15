# ComplyPal - AI Compliance Assistant

ComplyPal is an intelligent compliance assistant that helps organizations navigate complex regulatory requirements through AI-powered analysis and real-time guidance.

## 🌟 Features

- **Interactive Compliance Assessment**: Complete questionnaires to evaluate your compliance status
- **AI-Powered Chat Support**: Get real-time answers to compliance-related questions
- **Voice Interaction**: Support for voice input and text-to-speech output
- **Document Analysis**: RAG-based analysis of compliance documentation
- **PDF Report Generation**: Download detailed compliance reports and chat histories
- **Customizable Interface**: Light/Dark theme support
- **Assessment Evaluation**: Thorough evaluation system for compliance responses

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - Google Generative AI (Gemini)
  - LangChain for RAG implementation
  - ChromaDB for vector storage
- **Speech Processing**:
  - Speech Recognition
  - gTTS (Google Text-to-Speech)
- **Document Processing**:
  - PDFPlumber
  - FPDF for PDF generation
- **Data Analysis**:
  - Pandas
  - NumPy
  - SciPy

## 📋 Prerequisites

- Python 3.8+
- Google API Key
- OpenAI API Key (for evaluation)
- LangSmith API Key (for evaluation)

## 🚀 Installation

### Basic Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd complypal
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For Unix/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Environment Configuration

1. Create environment file:
```bash
# For Windows
copy .env.example .env

# For Unix/MacOS
cp .env.example .env
```

2. Configure your API keys in `.env`:
```ini
# API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key

# Vector DB Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# Model Configuration
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=40
MAX_OUTPUT_TOKENS=2048

# File Paths
PDF_DIR=./pdf_docs
CHROMA_DB_PATH=./chroma_db
LOG_DIR=./logs
```

### Document Setup

1. Prepare your compliance documents:
```bash
mkdir -p pdf_docs logs
```

2. Place your PDF documents in the `pdf_docs` directory:
- Regulatory documents
- Compliance guidelines
- Standard operating procedures
- Best practice documents

3. Configure the questionnaire:
```bash
# Example questionnaire.csv structure
question,type,options,help_text,placeholder
"Describe your data processing activities",textarea,,Provide details about how you handle customer data,Enter your data processing details...
"Do you collect sensitive data?",radio,"[""Yes"",""No"",""Not Sure""]",Select if you handle sensitive personal information,
"Which security measures do you have?",multiselect,"[""Encryption"",""Access Control"",""Regular Audits"",""DPO Appointed""]",Select all security measures in place,
```

### Optional: Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Configure VS Code settings:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}

## 🎯 Usage

### Local Development
1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. For evaluation:
```bash
python evaluate.py
```

### Streamlit Cloud Deployment

1. **Prepare Your Repository**
   - Push your code to GitHub
   - Ensure `requirements.txt` is up to date:
   ```bash
   pip freeze > requirements.txt
   ```
   - Create a `.streamlit/config.toml` file:
   ```toml
   [theme]
   primaryColor = "#4A69A5"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F5F5F5"
   textColor = "#333333"
   font = "Inter"
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and main file (app.py)
   - Click "Deploy"

3. **Configure Secrets**
   - In Streamlit Cloud, go to your app settings
   - Under "Secrets", add your environment variables:
   ```yaml
   GOOGLE_API_KEY: "your_google_api_key"
   OPENAI_API_KEY: "your_openai_api_key"
   LANGSMITH_API_KEY: "your_langsmith_api_key"
   ```

4. **Advanced Settings**
   - Python version: 3.9+
   - Packages: All requirements will be installed automatically
   - Memory: Request more if needed

### Environment Variables
For local development, use `.env`:
```env
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
LANGSMITH_API_KEY=your_key
```

For Streamlit Cloud, use the secrets management system as shown above.

### Important Notes
- Ensure all API keys are kept secret
- PDF documents need to be included in the repository
- Vector DB will be rebuilt on each deployment
- Consider storage limitations on Streamlit Cloud

## 📁 Project Structure

```
complypal/
├── app.py              # Main Streamlit application
├── evaluate.py         # Evaluation script
├── config.py          # Shared configuration
├── questionnaire.csv  # Assessment questions
├── pdf_docs/         # Compliance documents
├── .env              # Environment variables
└── README.md         # Documentation
```

## ⚙️ Configuration

### Basic Configuration (config.py)

```python
# Vector DB Configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 4))

# Model Configuration
GEMINI_CONFIG = {
    'model': 'gemini-pro',
    'temperature': float(os.getenv('TEMPERATURE', 0.7)),
    'top_p': float(os.getenv('TOP_P', 0.8)),
    'top_k': int(os.getenv('TOP_K', 40)),
    'max_output_tokens': int(os.getenv('MAX_OUTPUT_TOKENS', 2048)),
}

EMBEDDING_CONFIG = {
    'model': 'models/embedding-001',
}

# File Paths
PDF_DIR = os.getenv('PDF_DIR', './pdf_docs')
CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
LOG_DIR = os.getenv('LOG_DIR', './logs')
```

### UI Configuration

The application supports theme customization through Streamlit's configuration:

```python
# config.toml
[theme]
primaryColor = "#4A69A5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#333333"
font = "Inter"
```

### Logging Configuration

```python
# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
```

### Vector Database Configuration

You can customize the vector database settings in your `.env`:

```ini
# Vector DB Fine-tuning
CHUNK_SIZE=1000        # Size of text chunks for processing
CHUNK_OVERLAP=200      # Overlap between chunks
RETRIEVAL_K=4          # Number of similar documents to retrieve

# Storage Configuration
CHROMA_DB_PATH=./chroma_db
```

### Model Configuration

Fine-tune the AI model behavior:

```ini
# Model Parameters
TEMPERATURE=0.7        # Controls randomness (0.0-1.0)
TOP_P=0.8             # Nucleus sampling parameter
TOP_K=40              # Number of tokens to consider
MAX_OUTPUT_TOKENS=2048 # Maximum response length
```

## 📊 Evaluation System

The evaluation script (`evaluate.py`) provides:
- RAG vs OpenAI response comparison
- Scoring based on:
  - Factual correctness
  - Completeness
  - Clarity
- Detailed analytics and correlation analysis
- CSV report generation

## 🔒 Security Note

- Ensure compliance documents are properly secured
- API keys should be kept confidential
- User data should be handled according to privacy regulations

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 Rights Reserved

© 2024 ComplyPal. All rights reserved.

This is a private project. No license is granted to use, copy, modify, or distribute this software without explicit permission.

Note: This project uses third-party packages, each with their own licenses. Please refer to the individual package documentation for their license terms.

## 🙋‍♂️ Support

For issues and questions:
- Open an issue in the repository
- Contact [Your Contact Information]
