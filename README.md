ComplyPal - AI Compliance Assistant

ComplyPal is an intelligent compliance assistant that helps organizations navigate complex regulatory requirements through AI-powered analysis and real-time guidance.

Features

- **Interactive Compliance Assessment**: Complete questionnaires to evaluate your compliance status
- **AI-Powered Chat Support**: Get real-time answers to compliance-related questions
- **Voice Interaction**: Support for voice input and text-to-speech output
- **Document Analysis**: RAG-based analysis of compliance documentation
- **PDF Report Generation**: Download detailed compliance reports and chat histories
- **Customizable Interface**: Light/Dark theme support
- **Assessment Evaluation**: Thorough evaluation system for compliance responses

Technical Stack

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

Prerequisites

- Python 3.8+
- Google API Key
- OpenAI API Key (for evaluation)
- LangSmith API Key (for evaluation)

Installation

Basic Setup
-I began by cloning the repository and setting up the project environment-
-Next, I created and activated a virtual environment:
-After activating the environment, I installed all the necessary dependencies:
-Environment Configuration
-I set up the environment configuration by first creating the environment file:
-Then I configured the API keys and other settings in the .env file:

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
Document Setup
For document management, I created the necessary directories for PDF documents and logs:
-Set up the questionnaire configuration with the following structure: csvCopyquestion,type,options,help_text,placeholder

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

 Important Notes
- All API keys are kept secret
- PDF documents need to be included in the repository
- Vector DB is rebuilt on each deployment

Project Structure

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

Configuration
Basic Configuration (config.py)

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

 UI Configuration

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

Vector Database Configuration

# Vector DB Fine-tuning
CHUNK_SIZE=1000        # Size of text chunks for processing
CHUNK_OVERLAP=200      # Overlap between chunks
RETRIEVAL_K=4          # Number of similar documents to retrieve

# Storage Configuration
CHROMA_DB_PATH=./chroma_db
```
Evaluation System

The evaluation script (`evaluate.py`) provides:
- RAG vs OpenAI response comparison
- Scoring based on:
  - Factual correctness
  - Completeness
  - Clarity
- Detailed analytics and correlation analysis
- CSV report generation

Security Note

- Ensure compliance documents are properly secured
- API keys should be kept confidential
- User data should be handled according to privacy regulations

Contributing

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

