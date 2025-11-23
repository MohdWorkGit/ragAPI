# RAG System with LLaMA 3, Ollama - Document-Centric with Persistent Embeddings
# FastAPI server that processes all files in /docs and creates individual embeddings

import os
import json
import pickle
import tempfile
from typing import List, Optional, Dict
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
import logging
from pathlib import Path
import hashlib
from datetime import datetime
from urllib.parse import unquote
from fastapi.responses import FileResponse
from fastapi import Header
from fastapi.responses import FileResponse, StreamingResponse
from urllib.parse import unquote
import os
import re
import mimetypes


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DOC_FOLDER = "docs"
Videos_FOLDER = "videos"
EMBEDDINGS_FOLDER = "embeddings"
METADATA_FOLDER = "metadata"
SUMMARY_FOLDER = "summaries"
MODEL_NAME = "gemma3"  # Change this to your Ollama model name
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = "BAAI/bge-m3"



# Global variables
document_stores = {}  # Dictionary to store FAISS stores for each document
document_metadata = {}  # Store document metadata

video_stores = {}  # Dictionary to store FAISS stores for each video
video_metadata = {}  # Store video metadata

embeddings_model = None

# Pydantic models for API requests
class ChatRequest(BaseModel):
    message: str
    fileName: str
    timestamp: str

class AnalyzeRequest(BaseModel):
    fileName: str
    timestamp: str

class QueryRequest(BaseModel):
    query: str
    fileName: Optional[str] = None  # If provided, search only in this document

class RefreshRequest(BaseModel):
    fileName: Optional[str] = None  # If provided, refresh only this document

# Initialize FastAPI app
app = FastAPI(title="Document-Centric RAG API", description="RAG API with persistent document embeddings")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def get_file_hash(file_path: str) -> str:
    """Generate hash for file to detect changes"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_document(file_path: str):
    """Load a single document based on its extension"""
    filename = os.path.basename(file_path)
    try:
        if filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith((".md", ".markdown")):
            loader = UnstructuredFileLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {filename}")
            return None
        
        docs = loader.load()
        logger.info(f"Successfully loaded document: {filename}")
        return docs
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None

def create_document_embeddings(file_path: str, force_refresh: bool = False):
    """Create and save embeddings for a single document"""
    filename = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)
    
    # Check if embeddings already exist and are up to date
    metadata_path = os.path.join(METADATA_FOLDER, f"{filename}.metadata.json")
    embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.faiss")
    
    if not force_refresh and os.path.exists(metadata_path) and os.path.exists(embedding_path):
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)
        
        if existing_metadata.get('file_hash') == file_hash:
            logger.info(f"Embeddings for {filename} are up to date, loading from cache")
            return load_document_embeddings(filename)
    
    # Create new embeddings
    logger.info(f"Creating embeddings for: {filename}")
    
    docs = load_document(file_path)
    if not docs:
        return False
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        logger.warning(f"No chunks created for {filename}")
        return False
    
    # Create FAISS vector store
    try:
        db = FAISS.from_documents(chunks, embeddings_model)
        
        # Save the vector store
        os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
        db.save_local(embedding_path)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'file_path': file_path,
            'file_hash': file_hash,
            'chunk_count': len(chunks),
            'created_at': datetime.now().isoformat(),
            'model_used': EMBEDDING_MODEL
        }
        
        os.makedirs(METADATA_FOLDER, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in memory
        document_stores[filename] = db
        document_metadata[filename] = metadata
        
        logger.info(f"Successfully created embeddings for {filename} with {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error creating embeddings for {filename}: {e}")
        return False

def load_document_embeddings(filename: str):
    """Load existing embeddings for a document"""
    try:
        embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.faiss")
        metadata_path = os.path.join(METADATA_FOLDER, f"{filename}.metadata.json")
        
        if not os.path.exists(embedding_path) or not os.path.exists(metadata_path):
            return False
        
        # Load vector store
        db = FAISS.load_local(embedding_path, embeddings_model, allow_dangerous_deserialization=True)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        document_stores[filename] = db
        document_metadata[filename] = metadata
        
        logger.info(f"Loaded existing embeddings for: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading embeddings for {filename}: {e}")
        return False

def scan_and_process_documents():
    """Scan docs folder and process all documents"""
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)
        logger.info(f"Created {DOC_FOLDER} directory")
        return
    
    processed_count = 0
    failed_count = 0
    
    for filename in os.listdir(DOC_FOLDER):
        file_path = os.path.join(DOC_FOLDER, filename)
        
        if os.path.isfile(file_path):
            if create_document_embeddings(file_path):
                processed_count += 1
            else:
                failed_count += 1
    
    logger.info(f"Document processing complete: {processed_count} processed, {failed_count} failed")

def get_document_qa_chain(filename: str):
    """Get QA chain for a specific document"""
    if filename not in document_stores:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found or not processed")
    
    llm = Ollama(
        model=MODEL_NAME,
        temperature=0.1,
        top_p=0.9
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=document_stores[filename].as_retriever(search_kwargs={"k": 5}),
        return_source_documents=False
    )

def get_document_analysis(filename: str):
    """Generate comprehensive analysis for a document"""
    if filename not in document_stores:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found")
    
    try:
        # Get relevant chunks for analysis
        db = document_stores[filename]
        retriever = db.as_retriever(search_kwargs={"k": 10})
        
        # Get sample content
        sample_docs = retriever.get_relevant_documents("summary main topics content")
        sample_content = "\n".join([doc.page_content for doc in sample_docs[:5]])
        
        llm = Ollama(model=MODEL_NAME, temperature=0.1)
        
        analysis_prompt = f"""Please analyze this document and provide a comprehensive summary:

Document: {filename}
Content Sample:
{sample_content[:3000]}

Please provide:
1. A brief summary of the document
2. Key topics and themes covered
3. Main findings or conclusions
4. Document structure and organization
5. Important concepts or terminology used

Analysis:"""
        
        analysis = llm(analysis_prompt)
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "message": "Document-Centric RAG API is running",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "documents_loaded": len(document_stores),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Chat endpoint for document-specific Q&A"""
    try:
        logger.info(f"Chat request for document: {request.fileName}")
        
        # Get QA chain for the specific document
        qa_chain = get_document_qa_chain(request.fileName)
        
        # Generate response
        response = qa_chain.run("جاوب بالعربية, "+request.message)
        
        return {
            "response": response,
            "fileName": request.fileName,
            "timestamp": request.timestamp,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error for {request.fileName}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_document(request: AnalyzeRequest):
    """Analyze a specific document"""
    try:
        logger.info(f"Analyzing document: {request.fileName}")
        
        analysis = get_document_analysis(request.fileName)
        
        # Save analysis
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)
        summary_path = os.path.join(SUMMARY_FOLDER, f"{request.fileName}.analysis.txt")
        
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
        logger.error(f"Analysis error for {request.fileName}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """Query specific document or all documents"""
    try:
        if request.fileName:
            # Query specific document
            qa_chain = get_document_qa_chain(request.fileName)
            response = qa_chain.run(request.query)
            source = request.fileName
        else:
            # Query all documents (combine results)
            if not document_stores:
                raise HTTPException(status_code=503, detail="No documents available")
            
            responses = []
            for filename in document_stores.keys():
                try:
                    qa_chain = get_document_qa_chain(filename)
                    doc_response = qa_chain.run(request.query)
                    responses.append(f"From {filename}:\n{doc_response}\n")
                except Exception as e:
                    logger.warning(f"Error querying {filename}: {e}")
            
            if not responses:
                raise HTTPException(status_code=500, detail="No valid responses from documents")
            
            response = "\n".join(responses)
            source = "all_documents"
        
        return {
            "response": response,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    """List all available documents with metadata"""
    documents = []
    
    for filename, metadata in document_metadata.items():
        doc_info = {
            "filename": filename,
            "chunk_count": metadata.get("chunk_count", 0),
            "created_at": metadata.get("created_at"),
            "file_path": metadata.get("file_path"),
            "available": filename in document_stores
        }
        documents.append(doc_info)
    
    return {
        "documents": documents,
        "total_count": len(documents),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/file/{filename}")
async def get_file(filename: str):
    """Serve a file from the docs folder for viewing"""
    try:
        # Decode the filename
        decoded_filename = unquote(filename)
        
        # Construct the file path
        file_path = os.path.join(DOC_FOLDER, decoded_filename)
        
        # Security check to prevent directory traversal
        file_path = os.path.abspath(file_path)
        doc_folder_abs = os.path.abspath(DOC_FOLDER)
        
        if not file_path.startswith(doc_folder_abs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {decoded_filename} not found")
        
        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if decoded_filename.lower().endswith('.pdf'):
            content_type = "application/pdf"
        elif decoded_filename.lower().endswith('.txt'):
            content_type = "text/plain"
        elif decoded_filename.lower().endswith('.docx'):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif decoded_filename.lower().endswith(('.md', '.markdown')):
            content_type = "text/markdown"
        
        # Return the file
        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=content_type,
            content_disposition_type="inline",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"  # Adjust for production
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")
    
@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve a video file from the videos folder for streaming"""
    try:
        # Decode the filename
        decoded_filename = unquote(filename)
        
        # Construct the file path
        file_path = os.path.join(Videos_FOLDER, decoded_filename)
        
        # Security check to prevent directory traversal
        file_path = os.path.abspath(file_path)
        Videos_FOLDER_abs = os.path.abspath(Videos_FOLDER)
        
        if not file_path.startswith(Videos_FOLDER_abs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Video {decoded_filename} not found")
        
        # Determine content type based on file extension
        content_type = "video/mp4"  # Default to mp4
        if decoded_filename.lower().endswith('.mp4'):
            content_type = "video/mp4"
        elif decoded_filename.lower().endswith('.webm'):
            content_type = "video/webm"
        elif decoded_filename.lower().endswith('.ogg'):
            content_type = "video/ogg"
        elif decoded_filename.lower().endswith('.ogv'):
            content_type = "video/ogg"
        elif decoded_filename.lower().endswith('.mov'):
            content_type = "video/quicktime"
        elif decoded_filename.lower().endswith('.avi'):
            content_type = "video/x-msvideo"
        elif decoded_filename.lower().endswith('.mkv'):
            content_type = "video/x-matroska"
        elif decoded_filename.lower().endswith('.wmv'):
            content_type = "video/x-ms-wmv"
        elif decoded_filename.lower().endswith('.flv'):
            content_type = "video/x-flv"
        elif decoded_filename.lower().endswith('.m4v'):
            content_type = "video/x-m4v"
        elif decoded_filename.lower().endswith('.3gp'):
            content_type = "video/3gpp"
        
        # Get file size for range requests support (needed for video seeking)
        file_size = os.path.getsize(file_path)
        
        # Return the video file with appropriate headers for streaming
        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",  # Enable partial content for video seeking
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",  # Adjust for production
                "Content-Length": str(file_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving video {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

# Also add this endpoint for downloading files with proper headers
@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download a file from the docs folder"""
    try:
        decoded_filename = unquote(filename)
        file_path = os.path.join(DOC_FOLDER, decoded_filename)
        
        # Security check
        file_path = os.path.abspath(file_path)
        doc_folder_abs = os.path.abspath(DOC_FOLDER)
        
        if not file_path.startswith(doc_folder_abs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {decoded_filename} not found")
        
        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={decoded_filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.post("/api/refresh")
async def refresh_documents(request: RefreshRequest):
    """Refresh embeddings for all documents or a specific document"""
    try:
        if request.fileName:
            # Refresh specific document
            file_path = os.path.join(DOC_FOLDER, request.fileName)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File {request.fileName} not found in {DOC_FOLDER}")
            
            success = create_document_embeddings(file_path, force_refresh=True)
            return {
                "message": f"Document {request.fileName} {'refreshed successfully' if success else 'failed to refresh'}",
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Refresh all documents
            scan_and_process_documents()
            return {
                "message": "All documents refreshed",
                "documents_processed": len(document_stores),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "documents_loaded": len(document_stores),
        "available_documents": list(document_stores.keys()),
        "docs_folder": DOC_FOLDER,
        "videos_folder": Videos_FOLDER,
        "embeddings_folder": EMBEDDINGS_FOLDER,
        "system_ready": len(document_stores) > 0,
        "timestamp": datetime.now().isoformat()
    }

class OfflineEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 local_dir="./models/embeddings", device="cuda"):
        self.model_name = model_name
        self.local_dir = Path(local_dir)
        self.device = device
        self.embeddings_model = None
        
    def download_model(self):
        """Download model for offline use (run this once while online)"""
        print(f"Downloading {self.model_name} to {self.local_dir}")
        
        # Create directory if it doesn't exist
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using sentence-transformers
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(self.model_name)
        save_path = self.local_dir / self.model_name.replace('/', '_')
        model.save(str(save_path))
        
        print(f"Model saved to {save_path}")
        return save_path
    
    def load_offline(self):
        """Load model from local directory (works offline)"""
        # Set offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Find local model path
        model_path = self.local_dir / self.model_name.replace('/', '_')
        
        if not model_path.exists():
            # Try alternative path structure
            model_path = self.local_dir / self.model_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run download_model() first while online."
            )
        
        print(f"Loading model from {model_path}")
        
        # Initialize embeddings with local path
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={'device': self.device}
        )
        
        return self.embeddings_model
    
    def get_embeddings(self):
        """Get embeddings model (download if needed, then load)"""
        try:
            # Try loading offline first
            return self.load_offline()
        except FileNotFoundError:
            # If not found, check if we're online
            try:
                import requests
                requests.get('https://huggingface.co', timeout=5)
                print("Online - downloading model...")
                self.download_model()
                return self.load_offline()
            except:
                print("Offline and model not found. Please download the model first while online.")
                raise


# Initialize the system
def initialize_system():
    """Initialize the document processing system"""
    global embeddings_model
    
    try:
        logger.info("Initializing Document-Centric RAG system...")
        
        # # Initialize embeddings model
        # embeddings_model = HuggingFaceEmbeddings(
        #     model_name=EMBEDDING_MODEL,
        #     model_kwargs={'device': 'cuda'}  # Use 'cpu' if no GPU available
        # )

         # Initialize
        offline_embeddings = OfflineEmbeddings(
            model_name=EMBEDDING_MODEL,
            local_dir="./embeddings",
            device="cuda"
        )
        
        # First time setup (while online)
        # offline_embeddings.download_model()
        
        # Use offline
        embeddings_model = offline_embeddings.get_embeddings()
        
        # Create necessary directories
        os.makedirs(DOC_FOLDER, exist_ok=True)
        os.makedirs(Videos_FOLDER, exist_ok=True)
        os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
        os.makedirs(METADATA_FOLDER, exist_ok=True)
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)
        
        # Scan and process all documents
        scan_and_process_documents()
        
        logger.info(f"System initialized successfully! Loaded {len(document_stores)} documents")
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_system()

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Document-Centric RAG API Server...")
    print(f"📚 Model: {MODEL_NAME}")
    print(f"🔗 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Documents Folder: {DOC_FOLDER}")
    print(f"💾 Embeddings Folder: {EMBEDDINGS_FOLDER}")
    print(f"📊 Metadata Folder: {METADATA_FOLDER}")
    print("="*50)
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )