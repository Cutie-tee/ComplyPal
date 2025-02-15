import json
import pandas as pd
import re
import logging
from datetime import datetime
from scipy import stats
import numpy as np
from langsmith import Client, wrappers
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Import shared configuration
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(LOG_DIR, "evaluate.log")
)

# Check for required API keys
if not all([LANGSMITH_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY]):
    print("Error: Missing API keys. Please check your .env file.")
    exit(1)

# Initialize clients
client = Client(api_key=LANGSMITH_API_KEY)
openai_client = wrappers.wrap_openai(OpenAI(api_key=OPENAI_API_KEY))

def initialize_vector_db():
    """Initialize the vector database with PDF documents."""
    
    # Use existing vector store if available
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        logging.info("Loading existing vector database...")
        embedding_function = GoogleGenerativeAIEmbeddings(
            **EMBEDDING_CONFIG,
            google_api_key=GOOGLE_API_KEY
        )
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    
    # If no existing DB, check if PDF directory exists
    if not os.path.exists(PDF_DIR):
        logging.error(f"PDF directory '{PDF_DIR}' not found")
        return None
    
    logging.info("Creating new vector database from PDFs...")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    # Load PDFs
    loader = DirectoryLoader(PDF_DIR, glob="*.pdf", loader_cls=PDFPlumberLoader)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embedding_function = GoogleGenerativeAIEmbeddings(
        **EMBEDDING_CONFIG,
        google_api_key=GOOGLE_API_KEY
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_PATH
    )
    
    return vectorstore

def generate_rag_answer(question: str, vectorstore) -> str:
    """Generate an answer using RAG."""
    if not vectorstore:
        return ""
    
    # Retrieve relevant documents
    retrieved_docs = vectorstore.similarity_search(question, k=RETRIEVAL_K)
    
    if not retrieved_docs:
        context = "No specific compliance documentation found. Using general knowledge."
    else:
        context_parts = []
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'Unknown source')
            content = doc.page_content.strip()
            context_parts.append(f"[Source: {source}]\n{content}")
        context = "\n\n".join(context_parts)
    
    prompt = f"""As a compliance expert, please answer the following question using the provided context from compliance documents. If the context doesn't contain enough information, supplement with your knowledge of compliance best practices.

Context from compliance documents:
{context}

Question:
{question}

Please provide a detailed, accurate answer that cites specific sources when possible:"""
    
    rag_llm = GoogleGenerativeAI(
        **GEMINI_CONFIG,
        google_api_key=GOOGLE_API_KEY
    )
    
    answer = rag_llm.invoke(prompt)
    return answer.strip() if answer else ""

def openai_target(question: str) -> str:
    """
    Generate a baseline answer using OpenAI for comparison.
    """
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful compliance expert."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

def evaluate_response(ai_answer: str, correct_answer: str) -> dict:
    """
    Evaluate how good the AI's answer is compared to the correct answer.
    """
    instructions = """
    You are an expert compliance evaluator tasked with assessing AI-generated responses to real compliance inquiries. This evaluation is crucial as it impacts actual business decisions and regulatory compliance.

    Context:
    - These responses are from real conversations between users and AI compliance assistants
    - The answers influence how businesses approach their regulatory obligations
    - Accuracy and factual correctness are paramount given the legal implications
    - The 'correct answer' provided is verified by compliance experts

    Please evaluate the AI answer against the expert-verified correct answer using these strict criteria:

    1. Factual Correctness (score 0-10):
       - Absolute accuracy of compliance-related statements and regulatory citations
       - Precise interpretation of laws, regulations, and compliance requirements
       - Correct citation of specific sources, standards, and regulatory frameworks
       - No misleading or outdated regulatory information
       - Accurate representation of compliance obligations and deadlines

    2. Completeness (score 0-10):
       - Comprehensive coverage of all relevant compliance requirements
       - Thorough explanation of regulatory obligations and implications
       - Inclusion of necessary context about regulatory frameworks
       - Address all key aspects of the compliance query
       - Mention of relevant cross-regulatory requirements when applicable
       - Coverage of both immediate and long-term compliance considerations

    3. Clarity (score 0-10):
       - Clear, unambiguous presentation of compliance requirements
       - Professional use of regulatory and legal terminology
       - Well-structured explanation of compliance steps and processes
       - Actionable recommendations with clear implementation guidance
       - Logical organization of complex regulatory information
       - Clear distinction between mandatory requirements and best practices

    First provide a detailed analysis of each criterion, then on the last line write only:
    SCORES: X, Y, Z
    where X = Factual score, Y = Completeness score, Z = Clarity score
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"Correct Answer: {correct_answer}\nAI Answer: {ai_answer}"}
        ]
    )
    
    evaluation = response.choices[0].message.content
    
    # Extract scores
    for line in evaluation.splitlines():
        if line.startswith("SCORES:"):
            numbers = re.findall(r'\d+', line)
            if len(numbers) == 3:
                factual = int(numbers[0])
                completeness = int(numbers[1])
                clarity = int(numbers[2])
                
                # Calculate weighted overall score
                overall = (factual * 0.4 + completeness * 0.4 + clarity * 0.2)
                
                return {
                    "factual": factual,
                    "completeness": completeness,
                    "clarity": clarity,
                    "overall": overall,
                    "explanation": evaluation
                }
    
    return {"factual": 0, "completeness": 0, "clarity": 0, "overall": 0, "explanation": evaluation}

def analyze_length_completeness_correlation(results: list) -> dict:
    """
    Analyze the correlation between answer length and completeness scores.
    Returns correlation coefficients and other metrics for both RAG and OpenAI responses.
    """
    # Extract lengths and completeness scores
    rag_lengths = [len(r["rag_answer"].split()) for r in results]
    openai_lengths = [len(r["openai_answer"].split()) for r in results]
    rag_completeness = [r["rag_completeness"] for r in results]
    openai_completeness = [r["openai_completeness"] for r in results]
    
    # Calculate correlations
    rag_correlation, rag_pvalue = stats.pearsonr(rag_lengths, rag_completeness)
    openai_correlation, openai_pvalue = stats.pearsonr(openai_lengths, openai_completeness)
    
    # Calculate average words per completeness point
    rag_words_per_point = np.mean([length/score if score > 0 else 0 
                                  for length, score in zip(rag_lengths, rag_completeness)])
    openai_words_per_point = np.mean([length/score if score > 0 else 0 
                                     for length, score in zip(openai_lengths, openai_completeness)])
    
    return {
        "rag": {
            "correlation": rag_correlation,
            "p_value": rag_pvalue,
            "avg_length": np.mean(rag_lengths),
            "words_per_point": rag_words_per_point,
            "length_range": (min(rag_lengths), max(rag_lengths))
        },
        "openai": {
            "correlation": openai_correlation,
            "p_value": openai_pvalue,
            "avg_length": np.mean(openai_lengths),
            "words_per_point": openai_words_per_point,
            "length_range": (min(openai_lengths), max(openai_lengths))
        }
    }

def save_results(results: list, timestamp: str):
    """
    Save evaluation results and analysis to CSV file.
    """
    output_filename = f"evaluation_results_{timestamp}.csv"
    
    # First save the main results
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    
    # Calculate some basic stats
    rag_scores = [r["rag_overall"] for r in results]
    average_rag = sum(rag_scores) / len(rag_scores)
    
    # Add analysis to the file
    with open(output_filename, "a") as f:
        f.write("\n\nAnalysis Results\n")
        f.write("================\n")
        
        # Overall score
        f.write(f"\nAverage RAG Score: {average_rag:.2f}/10\n")
        
        # Score distribution
        f.write("\nScore Distribution:\n")
        excellent = sum(1 for s in rag_scores if s >= 9)
        good = sum(1 for s in rag_scores if 7 <= s < 9)
        fair = sum(1 for s in rag_scores if 5 <= s < 7)
        poor = sum(1 for s in rag_scores if s < 5)
        
        f.write(f"Excellent (9-10): {excellent} answers\n")
        f.write(f"Good (7-8.9): {good} answers\n")
        f.write(f"Fair (5-6.9): {fair} answers\n")
        f.write(f"Poor (below 5): {poor} answers\n")
        
        # Different thresholds
        f.write("\nAccuracy at Different Score Thresholds:\n")
        for threshold in range(5, 10):
            passing = sum(1 for s in rag_scores if s >= threshold)
            accuracy = (passing / len(rag_scores)) * 100
            f.write(f"Score {threshold} or higher: {accuracy:.1f}%\n")
        
        # Length-Completeness Analysis
        length_completeness_metrics = analyze_length_completeness_correlation(results)
        
        f.write("\nLength-Completeness Relationship Analysis:\n")
        f.write("======================================\n")
        
        # RAG Analysis
        f.write("\nRAG Model:\n")
        f.write(f"Correlation coefficient: {length_completeness_metrics['rag']['correlation']:.3f}\n")
        f.write(f"Statistical significance (p-value): {length_completeness_metrics['rag']['p_value']:.3f}\n")
        f.write(f"Average response length: {length_completeness_metrics['rag']['avg_length']:.1f} words\n")
        f.write(f"Words per completeness point: {length_completeness_metrics['rag']['words_per_point']:.1f}\n")
        f.write(f"Response length range: {length_completeness_metrics['rag']['length_range'][0]} - {length_completeness_metrics['rag']['length_range'][1]} words\n")
        
        # OpenAI Analysis
        f.write("\nOpenAI Model:\n")
        f.write(f"Correlation coefficient: {length_completeness_metrics['openai']['correlation']:.3f}\n")
        f.write(f"Statistical significance (p-value): {length_completeness_metrics['openai']['p_value']:.3f}\n")
        f.write(f"Average response length: {length_completeness_metrics['openai']['avg_length']:.1f} words\n")
        f.write(f"Words per completeness point: {length_completeness_metrics['openai']['words_per_point']:.1f}\n")
        f.write(f"Response length range: {length_completeness_metrics['openai']['length_range'][0]} - {length_completeness_metrics['openai']['length_range'][1]} words\n")
    
    print(f"\nEvaluation complete! Results saved to: {output_filename}")

def main():
    # Create timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize vector database
    logging.info("Initializing vector database...")
    vectorstore = initialize_vector_db()
    if not vectorstore:
        logging.error("Failed to initialize vector database")
        return
    
    # Load QA pairs
    with open("compliance_qa.json", "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
    logging.info(f"Loaded {len(qa_pairs)} QA pairs")
    
    results = []
    
    # Process each question
    for idx, qa in enumerate(qa_pairs, 1):
        question = qa.get("input", "")
        correct_answer = qa.get("output", "")
        
        logging.info(f"Processing question {idx}/{len(qa_pairs)}")
        print(f"Processing question {idx}/{len(qa_pairs)}")
        
        # Generate answers
        rag_answer = generate_rag_answer(question, vectorstore)
        openai_answer = openai_target(question)
        
        # Evaluate answers
        rag_eval = evaluate_response(rag_answer, correct_answer)
        openai_eval = evaluate_response(openai_answer, correct_answer)
        
        # Store results
        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "rag_answer": rag_answer,
            "openai_answer": openai_answer,
            "rag_factual": rag_eval["factual"],
            "rag_completeness": rag_eval["completeness"],
            "rag_clarity": rag_eval["clarity"],
            "rag_overall": rag_eval["overall"],
            "openai_factual": openai_eval["factual"],
            "openai_completeness": openai_eval["completeness"],
            "openai_clarity": openai_eval["clarity"],
            "openai_overall": openai_eval["overall"]
        })
    
    # Save results and analysis
    save_results(results, timestamp)

if __name__ == "__main__":
    main()