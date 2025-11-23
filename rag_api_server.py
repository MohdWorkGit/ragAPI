# RAG System with LLaMA 3, Ollama - Modified for React PDF Viewer
# FastAPI server with CORS support and base64 PDF handling

import os
import base64
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DOC_FOLDER = "docs"
SUMMARY_FOLDER = "summaries"
VECTOR_DB_PATH = "vector_db"
# MODEL_NAME = "llama3"  # Change this to your Ollama model name
MODEL_NAME = "gemma3"  # Change this to your Ollama model name
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Global variables
qa_chain = None
current_pdf_content = {}  # Store current PDF content for context

# Pydantic models for API requests
class ChatRequest(BaseModel):
    message: str
    fileName: str
    fileContent: Optional[str] = None  # Base64 encoded PDF
    timestamp: str

class AnalyzeRequest(BaseModel):
    fileContent: str  # Base64 encoded PDF
    fileName: str
    timestamp: str

class QueryRequest(BaseModel):
    query: str

# Initialize FastAPI app
app = FastAPI(title="PDF RAG API", description="RAG API for PDF analysis with LLaMA 3")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def load_documents_from_folder(folder_path: str):
    """Load documents from the docs folder"""
    docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return docs
        
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(path, encoding='utf-8')
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(path)
            elif filename.endswith(".md"):
                loader = UnstructuredFileLoader(path)
            else:
                continue
            docs.extend(loader.load())
            logger.info(f"Loaded document: {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    return docs

def build_vector_store(docs):
    """Build FAISS vector store from documents"""
    if not docs:
        return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
        model_kwargs={'device': 'cuda'}  # Use 'cuda' if you have GPU

    )
    
    db = FAISS.from_documents(chunks, embeddings)
    return db

def create_qa_chain(db):
    """Create RetrievalQA chain"""
    if not db:
        return None
    
    llm = Ollama(
        model=MODEL_NAME,
        temperature=0.1,
        top_p=0.9
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

def decode_pdf_content(base64_content: str, filename: str):
    """Decode base64 PDF content and extract text"""
    try:
        # Decode base64
        pdf_bytes = base64.b64decode(base64_content)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        # Extract text using PyPDF2
        reader = PdfReader(temp_path)
        text_content = ""
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        # Store in global context
        current_pdf_content[filename] = {
            'content': text_content,
            'pages': len(reader.pages),
            'temp_path': temp_path
        }
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return text_content
        
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def get_contextual_response(message: str, filename: str, pdf_content: str = None):
    """Generate contextual response about the PDF"""
    try:
        llm = Ollama(model=MODEL_NAME, temperature=0.1)
        
        # Build context-aware prompt
        if pdf_content:
            prompt = f"""You are an AI assistant analyzing a PDF document named "{filename}".

Document Content:
{pdf_content[:4000]}  # Limit content to avoid token limits

User Question: {message}

Please provide a helpful and accurate response based on the document content. If the question cannot be answered from the document, say so clearly.

Response:"""
        else:
            # Fallback for when no content is provided
            prompt = f"""You are an AI assistant helping with a PDF document named "{filename}".

User Question: {message}

I don't have access to the full document content right now, but I can provide general guidance about your question. Please note that for specific information about the document, I would need access to its content.

Response:"""
        
        response = llm(prompt)
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "message": "RAG API is running",
        "model": MODEL_NAME,
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/api/chat")
async def chat_with_pdf(request: ChatRequest):
    """Chat endpoint for PDF Q&A"""
    try:
        logger.info(f"Chat request for file: {request.fileName}")
        
        pdf_content = None
        
        # If file content is provided, process it
        if request.fileContent:
            pdf_content = decode_pdf_content(request.fileContent, request.fileName)
        elif request.fileName in current_pdf_content:
            pdf_content = current_pdf_content[request.fileName]['content']
        
        # Generate response
        response = get_contextual_response(request.message, request.fileName, pdf_content)
        
        return {
            "response": response,
            "fileName": request.fileName,
            "timestamp": request.timestamp,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_pdf(request: AnalyzeRequest):
    """Analyze PDF content and extract information"""
    try:
        logger.info(f"Analyzing PDF: {request.fileName}")
        
        # Decode and process PDF
        pdf_content = decode_pdf_content(request.fileContent, request.fileName)
        
        # Generate analysis using LLM
        llm = Ollama(model=MODEL_NAME, temperature=0.1)
        
        analysis_prompt = f"""Please analyze this PDF document and provide a comprehensive summary:

Document: {request.fileName}
Content: {pdf_content[:3000]}

Please provide:
1. A brief summary of the document
2. Key topics covered
3. Main findings or conclusions
4. Document structure/organization

Analysis:"""
        
        analysis = llm(analysis_prompt)
        
        # Save summary
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)
        summary_path = os.path.join(SUMMARY_FOLDER, f"{request.fileName}.summary.txt")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Analysis of {request.fileName}\n")
            f.write(f"Generated at: {request.timestamp}\n")
            f.write("="*50 + "\n")
            f.write(analysis)
        
        return {
            "analysis": analysis,
            "fileName": request.fileName,
            "timestamp": request.timestamp,
            "summaryPath": summary_path,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_general(request: QueryRequest):
    """General Q&A endpoint using document knowledge base"""
    try:
        if not qa_chain:
            raise HTTPException(status_code=503, detail="RAG pipeline not ready. Please ensure documents are loaded.")
        
        response = qa_chain.run(request.query)
        
        return {
            "response": response,
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"General Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "qa_chain_ready": qa_chain is not None,
        "docs_folder": DOC_FOLDER,
        "current_pdfs": list(current_pdf_content.keys()),
        "timestamp": "2024-01-01T00:00:00Z"
    }

# Initialize the system
def initialize_rag_system():
    """Initialize the RAG system with documents from folder"""
    global qa_chain
    try:
        logger.info("Initializing RAG system...")
        
        # Load documents from folder
        docs = load_documents_from_folder(DOC_FOLDER)
        logger.info(f"Loaded {len(docs)} documents")
        
        if docs:
            # Build vector store
            logger.info("Building vector store...")
            db = build_vector_store(docs)
            
            # Create QA chain
            logger.info("Creating QA chain...")
            qa_chain = create_qa_chain(db)
            
            logger.info("RAG system initialized successfully!")
        else:
            logger.warning("No documents found in docs folder. QA chain will not be available.")
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_rag_system()

# Main execution
if __name__ == "__main__":
    print("🚀 Starting RAG API Server for PDF Viewer...")
    print(f"📚 Model: {MODEL_NAME}")
    print(f"🔗 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Documents Folder: {DOC_FOLDER}")
    print(f"💾 Summaries Folder: {SUMMARY_FOLDER}")
    print("="*50)
    
    # Create necessary directories
    os.makedirs(DOC_FOLDER, exist_ok=True)
    os.makedirs(SUMMARY_FOLDER, exist_ok=True)
    # Run the server
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Allow external connections
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )