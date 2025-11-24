# RAG System with Video Analysis Integration
# Combines document RAG capabilities with video analysis (audio + visual)

import os
import json
import pickle
import tempfile
import subprocess
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.schema import Document
import logging
from pathlib import Path
import hashlib
from datetime import datetime
from urllib.parse import unquote
import re
import mimetypes
import numpy as np

# Video analysis imports
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    BlipProcessor,
    BlipForConditionalGeneration
)
import whisper
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DOC_FOLDER = "docs"
Videos_FOLDER = "videos"
EMBEDDINGS_FOLDER = "embeddings"
METADATA_FOLDER = "metadata"
SUMMARY_FOLDER = "summaries"
UNIFIED_EMBEDDINGS_PATH = "embeddings/unified_index.faiss"
UNIFIED_METADATA_PATH = "metadata/unified_index.metadata.json"
MODEL_NAME = "gemma3"  # Ollama model for RAG
EMBEDDING_MODEL = "BAAI/bge-m3"

# Video Analysis Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

# Global variables - RAG
document_stores = {}
unified_store = None
document_metadata = {}
unified_metadata = {}
embeddings_model = None

# Global variables - Video Analysis
asr_model = None
summariser_model = None
tokenizer = None
blip_processor = None
blip_model = None

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
    fileName: Optional[str] = None
    search_mode: Optional[str] = "unified"
    top_k: Optional[int] = 5
    language: Optional[str] = None

class RefreshRequest(BaseModel):
    fileName: Optional[str] = None
    rebuild_unified: Optional[bool] = True

class SearchResult(BaseModel):
    content: str
    source_file: str
    score: float
    chunk_index: int

class VideoAnalysisRequest(BaseModel):
    video_filename: str
    num_frames: Optional[int] = 10
    output_language: Optional[str] = "arabic"  # "arabic" or "english"

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RAG API with Video Analysis",
    description="RAG API with unified search, video streaming, and AI-powered video analysis"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Video Analysis Functions
# ============================================================================
def extract_frames(video_path: str, num_frames: int = 10) -> List[Image.Image]:
    """Extract frames from video at regular intervals"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return []

    # Calculate frame intervals
    interval = max(1, total_frames // num_frames)
    frames = []

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= num_frames:
                break

    cap.release()
    return frames

def analyze_frames(frames: List[Image.Image]) -> List[str]:
    """Generate captions for extracted frames"""
    captions = []

    for i, frame in enumerate(frames):
        # Generate caption for each frame
        inputs = blip_processor(frame, return_tensors="pt").to(DEVICE)
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(f"Frame {i+1}: {caption}")

    return captions

def extract_key_visual_features(frames: List[Image.Image]) -> str:
    """Extract key visual information from frames"""
    if not frames:
        return "No visual information extracted."

    # Analyze a subset of representative frames
    sample_frames = frames[::max(1, len(frames)//5)]  # Sample 5 frames

    visual_descriptions = []
    for frame in sample_frames:
        inputs = blip_processor(frame, "What is happening in this image?", return_tensors="pt").to(DEVICE)
        out = blip_model.generate(**inputs, max_length=100)
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        visual_descriptions.append(description)

    return " ".join(visual_descriptions)

def process_video_file(video_path: str, num_frames: int = 10, output_language: str = "arabic") -> Dict:
    """
    Process a video file: extract audio, transcribe, analyze frames, and generate summary
    """
    try:
        # Extract audio with ffmpeg
        audio_path = video_path + ".wav"
        subprocess.run([
            "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], check=True, capture_output=True)

        # 1. AUDIO ANALYSIS
        # Transcribe with Whisper
        result = asr_model.transcribe(audio_path, task="transcribe")
        transcript = result["text"]
        detected_lang = result["language"]
        logger.info(f"Audio transcribed: {transcript[:100]}...")

        # 2. VIDEO/IMAGE ANALYSIS
        # Extract frames from video
        frames = extract_frames(video_path, num_frames=num_frames)

        # Generate frame captions
        frame_captions = analyze_frames(frames)

        # Get visual summary
        visual_summary = extract_key_visual_features(frames)

        # 3. COMBINE AUDIO AND VISUAL INFORMATION
        combined_text = f"""
        Audio Content: {transcript}

        Visual Content: {visual_summary}
        """

        # 4. GENERATE SUMMARY IN TARGET LANGUAGE
        if output_language.lower() == "arabic":
            forced_bos_token_id = tokenizer.lang_code_to_id["ar_AR"]
        else:
            forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]

        # Encode combined input
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(DEVICE)

        # Generate summary
        summary_ids = summariser_model.generate(
            inputs["input_ids"],
            max_length=200,
            min_length=60,
            length_penalty=2.0,
            num_beams=4,
            forced_bos_token_id=forced_bos_token_id
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Clean up audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return {
            "summary": summary,
            "audio_transcript": transcript,
            "visual_descriptions": frame_captions,
            "visual_summary": visual_summary,
            "detected_language": detected_lang,
            "num_frames_analyzed": len(frames),
            "device": DEVICE,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        # Clean up
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise

# ============================================================================
# RAG Utility Functions (from rag_api_server_combined.py)
# ============================================================================
class OfflineEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 local_dir="./models/embeddings", device="cuda"):
        self.model_name = model_name
        self.local_dir = Path(local_dir)
        self.device = device
        self.embeddings_model = None

    def download_model(self):
        """Download model for offline use"""
        print(f"Downloading {self.model_name} to {self.local_dir}")
        self.local_dir.mkdir(parents=True, exist_ok=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        save_path = self.local_dir / self.model_name.replace('/', '_')
        model.save(str(save_path))
        print(f"Model saved to {save_path}")
        return save_path

    def load_offline(self):
        """Load model from local directory"""
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        model_path = self.local_dir / self.model_name.replace('/', '_')
        if not model_path.exists():
            model_path = self.local_dir / self.model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading model from {model_path}")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=str(model_path),
            model_kwargs={'device': self.device}
        )
        return self.embeddings_model

    def get_embeddings(self):
        """Get embeddings model"""
        try:
            return self.load_offline()
        except FileNotFoundError:
            try:
                import requests
                requests.get('https://huggingface.co', timeout=5)
                print("Online - downloading model...")
                self.download_model()
                return self.load_offline()
            except:
                print("Offline and model not found. Please download the model first while online.")
                raise

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
        for doc in docs:
            doc.metadata['source_file'] = filename
            doc.metadata['file_path'] = file_path

        logger.info(f"Successfully loaded document: {filename}")
        return docs
    except Exception as e:
        logger.error(f"Error loading {filename}: {e}")
        return None

def create_document_embeddings(file_path: str, force_refresh: bool = False):
    """Create and save embeddings for a single document"""
    filename = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)

    metadata_path = os.path.join(METADATA_FOLDER, f"{filename}.metadata.json")
    embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.faiss")

    if not force_refresh and os.path.exists(metadata_path) and os.path.exists(embedding_path):
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)

        if existing_metadata.get('file_hash') == file_hash:
            logger.info(f"Embeddings for {filename} are up to date, loading from cache")
            return load_document_embeddings(filename)

    logger.info(f"Creating embeddings for: {filename}")

    docs = load_document(file_path)
    if not docs:
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        logger.warning(f"No chunks created for {filename}")
        return False

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
        chunk.metadata['total_chunks'] = len(chunks)

    try:
        db = FAISS.from_documents(chunks, embeddings_model)
        os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
        db.save_local(embedding_path)

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

        db = FAISS.load_local(embedding_path, embeddings_model, allow_dangerous_deserialization=True)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        document_stores[filename] = db
        document_metadata[filename] = metadata

        logger.info(f"Loaded existing embeddings for: {filename}")
        return True

    except Exception as e:
        logger.error(f"Error loading embeddings for {filename}: {e}")
        return False

def needs_unified_rebuild() -> bool:
    """Check if unified index needs to be rebuilt"""
    if not os.path.exists(UNIFIED_EMBEDDINGS_PATH) or not os.path.exists(UNIFIED_METADATA_PATH):
        logger.info("Unified index not found - rebuild needed")
        return True

    try:
        with open(UNIFIED_METADATA_PATH, 'r') as f:
            existing_unified_metadata = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading unified metadata: {e} - rebuild needed")
        return True

    if existing_unified_metadata.get('total_documents', 0) != len(document_stores):
        logger.info("Document count changed - rebuild needed")
        return True

    existing_files = set(existing_unified_metadata.get('source_files', []))
    current_files = set(document_stores.keys())

    if existing_files != current_files:
        logger.info("Document list changed - rebuild needed")
        return True

    unified_created_at = existing_unified_metadata.get('created_at', '')
    for filename, metadata in document_metadata.items():
        doc_created_at = metadata.get('created_at', '')
        if doc_created_at > unified_created_at:
            logger.info(f"Document {filename} was updated - rebuild needed")
            return True

    logger.info("Unified index is up to date")
    return False

def build_unified_index(force_rebuild: bool = False):
    """Build a unified FAISS index from all individual document stores"""
    global unified_store, unified_metadata

    if not document_stores:
        logger.warning("No documents available to build unified index")
        return False

    if not force_rebuild and not needs_unified_rebuild():
        logger.info("Using existing unified index")
        return load_unified_index()

    try:
        logger.info("Building unified index from all documents...")

        all_documents = []
        doc_mapping = {}
        current_index = 0

        for filename, store in document_stores.items():
            docs = store.docstore._dict.values()
            docs_list = list(docs)

            for i, doc in enumerate(docs_list):
                doc.metadata['unified_index'] = current_index + i
                doc.metadata['source_file'] = filename
                all_documents.append(doc)

            doc_mapping[filename] = {
                'start_index': current_index,
                'end_index': current_index + len(docs_list) - 1,
                'chunk_count': len(docs_list)
            }

            current_index += len(docs_list)

        unified_store = FAISS.from_documents(all_documents, embeddings_model)
        unified_store.save_local(UNIFIED_EMBEDDINGS_PATH)

        unified_metadata = {
            'total_documents': len(document_stores),
            'total_chunks': len(all_documents),
            'document_mapping': doc_mapping,
            'created_at': datetime.now().isoformat(),
            'source_files': list(document_stores.keys()),
            'model_used': EMBEDDING_MODEL
        }

        with open(UNIFIED_METADATA_PATH, 'w') as f:
            json.dump(unified_metadata, f, indent=2)

        logger.info(f"Unified index built with {len(all_documents)} chunks from {len(document_stores)} documents")
        return True

    except Exception as e:
        logger.error(f"Error building unified index: {e}")
        return False

def load_unified_index():
    """Load existing unified index"""
    global unified_store, unified_metadata

    try:
        if os.path.exists(UNIFIED_EMBEDDINGS_PATH) and os.path.exists(UNIFIED_METADATA_PATH):
            unified_store = FAISS.load_local(UNIFIED_EMBEDDINGS_PATH, embeddings_model, allow_dangerous_deserialization=True)

            with open(UNIFIED_METADATA_PATH, 'r') as f:
                unified_metadata = json.load(f)

            logger.info(f"Loaded unified index with {unified_metadata.get('total_chunks', 0)} chunks")
            return True
    except Exception as e:
        logger.error(f"Error loading unified index: {e}")

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

    if document_stores:
        build_unified_index()

def get_document_qa_chain(filename: str = None, use_unified: bool = False):
    """Get QA chain for a specific document or unified index"""
    if use_unified and unified_store:
        retriever = unified_store.as_retriever(search_kwargs={"k": 5})
    elif filename and filename in document_stores:
        retriever = document_stores[filename].as_retriever(search_kwargs={"k": 5})
    else:
        raise HTTPException(status_code=404, detail="Document or unified index not found")

    llm = Ollama(
        model=MODEL_NAME,
        temperature=0.1,
        top_p=0.9
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# ============================================================================
# API Endpoints - Video Analysis
# ============================================================================
@app.post("/api/video/upload_and_analyze")
async def upload_and_analyze_video(
    file: UploadFile = File(...),
    num_frames: int = 10,
    output_language: str = "arabic"
):
    """
    Upload and analyze a video file
    Returns: audio transcript, visual analysis, and combined summary
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        logger.info(f"Processing uploaded video: {file.filename}")

        # Process the video
        result = process_video_file(tmp_path, num_frames, output_language)
        result["original_filename"] = file.filename

        # Save the analysis to summaries folder
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)
        summary_path = os.path.join(SUMMARY_FOLDER, f"{file.filename}.video_analysis.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["summary_saved_to"] = summary_path

        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return result

    except Exception as e:
        # Clean up temp file on error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        logger.error(f"Error in video upload and analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/video/analyze_existing")
async def analyze_existing_video(request: VideoAnalysisRequest):
    """
    Analyze a video file that already exists in the videos folder
    """
    try:
        video_path = os.path.join(Videos_FOLDER, request.video_filename)

        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail=f"Video file {request.video_filename} not found in {Videos_FOLDER}")

        logger.info(f"Analyzing existing video: {request.video_filename}")

        # Process the video
        result = process_video_file(
            video_path,
            request.num_frames,
            request.output_language
        )
        result["video_filename"] = request.video_filename

        # Save the analysis
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)
        summary_path = os.path.join(SUMMARY_FOLDER, f"{request.video_filename}.video_analysis.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        result["summary_saved_to"] = summary_path

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# API Endpoints - RAG (from rag_api_server_combined.py)
# ============================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "message": "Enhanced RAG API with Video Analysis is running",
        "rag_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "video_analysis_available": asr_model is not None,
        "documents_loaded": len(document_stores),
        "unified_index_ready": unified_store is not None,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Chat endpoint for document-specific Q&A"""
    try:
        logger.info(f"Chat request for document: {request.fileName}")
        qa_chain = get_document_qa_chain(request.fileName)
        result = qa_chain({"query": "جاوب بالعربية, " + request.message})

        sources = []
        if 'source_documents' in result:
            for doc in result['source_documents']:
                sources.append({
                    'content': doc.page_content[:200],
                    'chunk_index': doc.metadata.get('chunk_index', -1)
                })

        return {
            "response": result['result'],
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

        if request.fileName not in document_stores:
            raise HTTPException(status_code=404, detail=f"Document {request.fileName} not found")

        db = document_stores[request.fileName]
        retriever = db.as_retriever(search_kwargs={"k": 10})
        sample_docs = retriever.get_relevant_documents("summary main topics content")
        sample_content = "\n".join([doc.page_content for doc in sample_docs[:5]])

        llm = Ollama(model=MODEL_NAME, temperature=0.1)

        analysis_prompt = f"""Please analyze this document and provide a comprehensive summary:

Document: {request.fileName}
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

@app.post("/api/ask_unified")
async def ask_unified(request: QueryRequest):
    """Ask the LLM a question using unified index across all documents"""
    try:
        if not unified_store:
            raise HTTPException(status_code=503, detail="Unified index not available. Please refresh documents.")

        logger.info(f"Unified query: {request.query}")

        qa_chain = get_document_qa_chain(use_unified=True)

        query = request.query
        if request.language and request.language.lower() == "arabic":
            query = "جاوب بالعربية, " + query

        result = qa_chain({"query": query})

        sources = []
        source_files = set()
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source_file = doc.metadata.get('source_file', 'unknown')
                source_files.add(source_file)
                sources.append({
                    'file': source_file,
                    'content_preview': doc.page_content[:200],
                    'chunk_index': doc.metadata.get('chunk_index', -1)
                })

        return {
            "response": result['result'],
            "query": request.query,
            "source": "unified_index",
            "sources_used": sources,
            "unique_source_files": list(source_files),
            "total_sources": len(sources),
            "language": request.language or "default",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Unified query error: {e}")
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
        "unified_index_ready": unified_store is not None,
        "unified_chunks": unified_metadata.get('total_chunks', 0) if unified_metadata else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/videos")
async def list_videos():
    """List all available videos"""
    if not os.path.exists(Videos_FOLDER):
        return {"videos": [], "total_count": 0}

    videos = []
    for filename in os.listdir(Videos_FOLDER):
        file_path = os.path.join(Videos_FOLDER, filename)
        if os.path.isfile(file_path):
            # Check if analysis exists
            analysis_path = os.path.join(SUMMARY_FOLDER, f"{filename}.video_analysis.json")
            has_analysis = os.path.exists(analysis_path)

            videos.append({
                "filename": filename,
                "file_path": file_path,
                "has_analysis": has_analysis,
                "analysis_path": analysis_path if has_analysis else None
            })

    return {
        "videos": videos,
        "total_count": len(videos),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/file/{filename}")
async def get_file(filename: str):
    """Serve a file from the docs folder"""
    try:
        decoded_filename = unquote(filename)
        file_path = os.path.join(DOC_FOLDER, decoded_filename)

        file_path = os.path.abspath(file_path)
        doc_folder_abs = os.path.abspath(DOC_FOLDER)

        if not file_path.startswith(doc_folder_abs):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {decoded_filename} not found")

        content_type = "application/octet-stream"
        if decoded_filename.lower().endswith('.pdf'):
            content_type = "application/pdf"
        elif decoded_filename.lower().endswith('.txt'):
            content_type = "text/plain"
        elif decoded_filename.lower().endswith('.docx'):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif decoded_filename.lower().endswith(('.md', '.markdown')):
            content_type = "text/markdown"

        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=content_type,
            content_disposition_type="inline",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
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
        decoded_filename = unquote(filename)
        file_path = os.path.join(Videos_FOLDER, decoded_filename)

        file_path = os.path.abspath(file_path)
        Videos_FOLDER_abs = os.path.abspath(Videos_FOLDER)

        if not file_path.startswith(Videos_FOLDER_abs):
            raise HTTPException(status_code=403, detail="Access denied")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Video {decoded_filename} not found")

        content_type = "video/mp4"
        ext = decoded_filename.lower().split('.')[-1]
        mime_types = {
            'mp4': 'video/mp4',
            'webm': 'video/webm',
            'ogg': 'video/ogg',
            'ogv': 'video/ogg',
            'mov': 'video/quicktime',
            'avi': 'video/x-msvideo',
            'mkv': 'video/x-matroska'
        }
        content_type = mime_types.get(ext, 'video/mp4')

        file_size = os.path.getsize(file_path)

        return FileResponse(
            path=file_path,
            filename=decoded_filename,
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*",
                "Content-Length": str(file_size)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving video {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")

@app.post("/api/refresh")
async def refresh_documents(request: RefreshRequest):
    """Refresh embeddings for all documents or a specific document"""
    try:
        if request.fileName:
            file_path = os.path.join(DOC_FOLDER, request.fileName)
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File {request.fileName} not found")

            success = create_document_embeddings(file_path, force_refresh=True)

            if success and request.rebuild_unified:
                build_unified_index(force_rebuild=True)

            return {
                "message": f"Document {request.fileName} {'refreshed successfully' if success else 'failed to refresh'}",
                "success": success,
                "unified_rebuilt": request.rebuild_unified and success,
                "timestamp": datetime.now().isoformat()
            }
        else:
            scan_and_process_documents()
            return {
                "message": "All documents refreshed",
                "documents_processed": len(document_stores),
                "unified_index_ready": unified_store is not None,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "rag_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "documents_loaded": len(document_stores),
        "available_documents": list(document_stores.keys()),
        "unified_index": {
            "ready": unified_store is not None,
            "total_chunks": unified_metadata.get('total_chunks', 0) if unified_metadata else 0,
            "source_files": unified_metadata.get('source_files', []) if unified_metadata else [],
            "created_at": unified_metadata.get('created_at') if unified_metadata else None
        },
        "video_analysis": {
            "available": asr_model is not None,
            "device": DEVICE,
            "models_loaded": {
                "whisper": asr_model is not None,
                "blip": blip_model is not None,
                "mbart": summariser_model is not None
            }
        },
        "folders": {
            "docs": DOC_FOLDER,
            "videos": Videos_FOLDER,
            "embeddings": EMBEDDINGS_FOLDER,
            "metadata": METADATA_FOLDER,
            "summaries": SUMMARY_FOLDER
        },
        "system_ready": len(document_stores) > 0 or asr_model is not None,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# System Initialization
# ============================================================================
def initialize_rag_system():
    """Initialize the RAG document processing system"""
    global embeddings_model

    try:
        logger.info("Initializing RAG system...")

        offline_embeddings = OfflineEmbeddings(
            model_name=EMBEDDING_MODEL,
            local_dir="./embeddings",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        embeddings_model = offline_embeddings.get_embeddings()

        os.makedirs(DOC_FOLDER, exist_ok=True)
        os.makedirs(Videos_FOLDER, exist_ok=True)
        os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
        os.makedirs(METADATA_FOLDER, exist_ok=True)
        os.makedirs(SUMMARY_FOLDER, exist_ok=True)

        scan_and_process_documents()

        logger.info(f"RAG system initialized! Loaded {len(document_stores)} documents")
        if unified_store:
            logger.info(f"Unified index ready with {unified_metadata.get('total_chunks', 0)} chunks")

    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise

def initialize_video_analysis_models():
    """Initialize video analysis models (Whisper, BLIP, mBART)"""
    global asr_model, summariser_model, tokenizer, blip_processor, blip_model

    try:
        logger.info("Initializing video analysis models...")

        # Load Whisper for audio transcription
        logger.info("Loading Whisper ASR model...")
        asr_model = whisper.load_model("large").to(DEVICE)
        logger.info("Whisper model loaded successfully")

        # Load mBART for summarization
        logger.info("Loading mBART summarization model...")
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        summariser_model = MBartForConditionalGeneration.from_pretrained(model_name).to(DEVICE, dtype=torch.float32)
        logger.info("mBART model loaded successfully")

        # Load BLIP for image captioning
        logger.info("Loading BLIP image captioning model...")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)
        logger.info("BLIP model loaded successfully")

        logger.info("All video analysis models initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing video analysis models: {e}")
        logger.warning("Video analysis features will not be available")
        # Don't raise - allow system to run without video analysis if models fail to load

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("="*80)
    logger.info("Starting Enhanced RAG API with Video Analysis...")
    logger.info("="*80)

    # Initialize RAG system
    initialize_rag_system()

    # Initialize video analysis models
    initialize_video_analysis_models()

    logger.info("="*80)
    logger.info("System startup complete!")
    logger.info("="*80)

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG API with Video Analysis...")
    print("="*80)
    print(f"📚 RAG Model: {MODEL_NAME}")
    print(f"🔗 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Documents Folder: {DOC_FOLDER}")
    print(f"🎥 Videos Folder: {Videos_FOLDER}")
    print(f"💾 Embeddings Folder: {EMBEDDINGS_FOLDER}")
    print(f"📊 Metadata Folder: {METADATA_FOLDER}")
    print(f"📝 Summaries Folder: {SUMMARY_FOLDER}")
    print(f"🖥️  Device: {DEVICE}")
    print("="*80)
    print("✨ Features:")
    print("  - Document RAG with unified search")
    print("  - Video upload and analysis")
    print("  - Audio transcription (Whisper)")
    print("  - Visual analysis (BLIP)")
    print("  - Multi-modal summarization (mBART)")
    print("  - Video streaming support")
    print("  - Arabic language support")
    print("="*80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload when using large models
        log_level="info"
    )
