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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains.combine_documents import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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

# Import writer manager
from writer_manager import get_writer_manager

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

def extract_video_content_as_text(video_path: str, num_frames: int = 10) -> Tuple[str, Dict]:
    """
    Extract video content as text for embeddings (audio transcript + visual descriptions)
    Returns: (combined_text, metadata_dict)
    """
    audio_path = None
    try:
        # Extract audio with ffmpeg
        audio_path = video_path + ".wav"
        subprocess.run([
            "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], check=True, capture_output=True)

        # 1. AUDIO ANALYSIS - Transcribe with Whisper
        result = asr_model.transcribe(audio_path, task="transcribe")
        transcript = result["text"]
        detected_lang = result["language"]
        logger.info(f"Audio transcribed from {os.path.basename(video_path)}: {len(transcript)} characters")

        # 2. VIDEO/IMAGE ANALYSIS - Extract frames
        frames = extract_frames(video_path, num_frames=num_frames)

        # Generate frame captions
        frame_captions = analyze_frames(frames)

        # Get visual summary
        visual_summary = extract_key_visual_features(frames)

        # 3. COMBINE AUDIO AND VISUAL INFORMATION INTO TEXT
        combined_text = f"""Video Content Analysis:

Audio Transcript:
{transcript}

Visual Analysis:
{visual_summary}

Detailed Frame Descriptions:
{chr(10).join(frame_captions)}
"""

        # Metadata for this video
        metadata = {
            "detected_language": detected_lang,
            "num_frames_analyzed": len(frames),
            "transcript_length": len(transcript),
            "has_audio": len(transcript) > 0,
            "has_visual": len(frames) > 0
        }

        # Clean up audio file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

        return combined_text, metadata

    except Exception as e:
        logger.error(f"Error extracting video content: {e}")
        # Clean up
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        raise

def process_video_file(video_path: str, num_frames: int = 10, output_language: str = "arabic") -> Dict:
    """
    Process a video file: extract audio, transcribe, analyze frames, and generate summary
    (Used for direct video analysis endpoint)
    """
    try:
        # Extract content as text
        combined_text, extraction_metadata = extract_video_content_as_text(video_path, num_frames)

        # Generate summary in target language using mBART
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

        return {
            "summary": summary,
            "audio_transcript": combined_text.split("Visual Analysis:")[0].replace("Video Content Analysis:", "").replace("Audio Transcript:", "").strip(),
            "visual_summary": extraction_metadata,
            "detected_language": extraction_metadata["detected_language"],
            "num_frames_analyzed": extraction_metadata["num_frames_analyzed"],
            "device": DEVICE,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
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

    # Extract writers from document
    try:
        # Get full text content from all chunks
        full_text = "\n".join([doc.page_content for doc in docs])
        writer_manager = get_writer_manager()
        writer_ids = writer_manager.process_document_for_writers(filename, full_text)
        logger.info(f"Extracted {len(writer_ids)} writers from {filename}")
    except Exception as e:
        logger.error(f"Error extracting writers from {filename}: {e}")
        writer_ids = []

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
            'model_used': EMBEDDING_MODEL,
            'writers_extracted': writer_ids  # Add writer IDs to metadata
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

# ============================================================================
# Video Embeddings Functions
# ============================================================================
def create_video_embeddings(file_path: str, force_refresh: bool = False, num_frames: int = 10):
    """Create and save embeddings for a video file"""
    filename = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)

    metadata_path = os.path.join(METADATA_FOLDER, f"{filename}.metadata.json")
    embedding_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.faiss")

    # Check if embeddings already exist and are up to date
    if not force_refresh and os.path.exists(metadata_path) and os.path.exists(embedding_path):
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)

        if existing_metadata.get('file_hash') == file_hash:
            logger.info(f"Embeddings for video {filename} are up to date, loading from cache")
            return load_document_embeddings(filename)  # Reuse same loading function

    # Create new embeddings
    logger.info(f"Creating embeddings for video: {filename}")

    try:
        # Extract video content as text
        video_text, video_metadata = extract_video_content_as_text(file_path, num_frames)

        # Create a Document object from the video content
        doc = Document(
            page_content=video_text,
            metadata={
                'source_file': filename,
                'file_path': file_path,
                'content_type': 'video',
                **video_metadata
            }
        )

        # Split the video content into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents([doc])

        if not chunks:
            logger.warning(f"No chunks created for video {filename}")
            return False

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)

        # Extract writers from video content (transcript + visual analysis)
        try:
            writer_manager = get_writer_manager()
            writer_ids = writer_manager.process_document_for_writers(filename, video_text)
            logger.info(f"Extracted {len(writer_ids)} writers from video {filename}")
        except Exception as e:
            logger.error(f"Error extracting writers from video {filename}: {e}")
            writer_ids = []

        # Create FAISS vector store
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
            'model_used': EMBEDDING_MODEL,
            'content_type': 'video',
            'video_metadata': video_metadata,
            'writers_extracted': writer_ids  # Add writer IDs to metadata
        }

        os.makedirs(METADATA_FOLDER, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Store in memory
        document_stores[filename] = db
        document_metadata[filename] = metadata

        logger.info(f"Successfully created embeddings for video {filename} with {len(chunks)} chunks")
        return True

    except Exception as e:
        logger.error(f"Error creating embeddings for video {filename}: {e}")
        return False

def scan_and_process_videos():
    """Scan videos folder and process all video files"""
    if not os.path.exists(Videos_FOLDER):
        os.makedirs(Videos_FOLDER)
        logger.info(f"Created {Videos_FOLDER} directory")
        return

    # Video file extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp', '.ogv', '.ogg')

    processed_count = 0
    failed_count = 0

    for filename in os.listdir(Videos_FOLDER):
        file_path = os.path.join(Videos_FOLDER, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(video_extensions):
            try:
                if create_video_embeddings(file_path):
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to process video {filename}: {e}")
                failed_count += 1

    logger.info(f"Video processing complete: {processed_count} processed, {failed_count} failed")

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

def scan_and_process_all_content():
    """Scan both docs and videos folders and process all content"""
    logger.info("Scanning and processing all content (documents and videos)...")

    # Process documents
    scan_and_process_documents()

    # Process videos
    scan_and_process_videos()

    # Build unified index with both documents and videos
    if document_stores:
        build_unified_index()
        logger.info(f"Unified index includes {len(document_stores)} items (documents + videos)")

def get_document_qa_chain(filename: str = None, use_unified: bool = False, system_message: str = ""):
    """Get QA chain for a specific document or unified index"""
    if use_unified and unified_store:
        retriever = unified_store.as_retriever(search_kwargs={"k": 5})
    elif filename and filename in document_stores:
        retriever = document_stores[filename].as_retriever(search_kwargs={"k": 5})
    else:
        raise HTTPException(status_code=404, detail="Document or unified index not found")

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.1,
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message if system_message else "You are a helpful assistant. Use the following context to answer the user's question."),
        ("system", "Context: {context}"),
        ("human", "{input}"),
    ])

    # Create the chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

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
    doc_count = sum(1 for m in document_metadata.values() if m.get('content_type') != 'video')
    video_count = sum(1 for m in document_metadata.values() if m.get('content_type') == 'video')

    return {
        "status": "OK",
        "message": "Enhanced RAG API with Video Analysis is running",
        "rag_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "video_analysis_available": asr_model is not None,
        "content_loaded": {
            "documents": doc_count,
            "videos": video_count,
            "total": len(document_stores)
        },
        "unified_index_ready": unified_store is not None,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    """Chat endpoint for document-specific Q&A"""
    try:
        logger.info(f"Chat request for document: {request.fileName}")
        qa_chain = get_document_qa_chain(request.fileName, system_message="جاوب بالعربية (Answer in Arabic)")
        result = qa_chain.invoke({"input": request.message})

        sources = []
        if 'context' in result:
            for doc in result['context']:
                sources.append({
                    'content': doc.page_content[:200],
                    'chunk_index': doc.metadata.get('chunk_index', -1)
                })

        return {
            "response": result['answer'],
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
        sample_docs = retriever.invoke("summary main topics content")
        sample_content = "\n".join([doc.page_content for doc in sample_docs[:5]])

        llm = ChatOllama(model=MODEL_NAME, temperature=0.1)

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

        analysis = llm.invoke(analysis_prompt).content

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

        system_msg = "You are a helpful assistant."
        if request.language and request.language.lower() == "arabic":
            system_msg = "جاوب بالعربية (Answer in Arabic). You are a helpful assistant."

        qa_chain = get_document_qa_chain(use_unified=True, system_message=system_msg)

        result = qa_chain.invoke({"input": request.query})

        sources = []
        source_files = set()
        if 'context' in result:
            for doc in result['context']:
                source_file = doc.metadata.get('source_file', 'unknown')
                source_files.add(source_file)
                sources.append({
                    'file': source_file,
                    'content_preview': doc.page_content[:200],
                    'chunk_index': doc.metadata.get('chunk_index', -1)
                })

        return {
            "response": result['answer'],
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
    """List all available documents and videos with metadata"""
    documents = []
    videos = []

    for filename, metadata in document_metadata.items():
        item_info = {
            "filename": filename,
            "chunk_count": metadata.get("chunk_count", 0),
            "created_at": metadata.get("created_at"),
            "file_path": metadata.get("file_path"),
            "available": filename in document_stores,
            "content_type": metadata.get("content_type", "document")
        }

        # Separate documents and videos
        if metadata.get("content_type") == "video":
            # Add video-specific metadata
            item_info["video_metadata"] = metadata.get("video_metadata", {})
            videos.append(item_info)
        else:
            documents.append(item_info)

    return {
        "documents": documents,
        "videos": videos,
        "total_documents": len(documents),
        "total_videos": len(videos),
        "total_items": len(documents) + len(videos),
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
    """Refresh embeddings for all documents/videos or a specific file"""
    try:
        if request.fileName:
            # Try to find the file in either docs or videos folder
            file_path = os.path.join(DOC_FOLDER, request.fileName)
            is_video = False

            if not os.path.exists(file_path):
                # Try videos folder
                file_path = os.path.join(Videos_FOLDER, request.fileName)
                is_video = True

                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"File {request.fileName} not found in docs or videos folder")

            # Refresh the file
            if is_video:
                success = create_video_embeddings(file_path, force_refresh=True)
            else:
                success = create_document_embeddings(file_path, force_refresh=True)

            if success and request.rebuild_unified:
                build_unified_index(force_rebuild=True)

            content_type = "video" if is_video else "document"
            return {
                "message": f"{content_type.capitalize()} {request.fileName} {'refreshed successfully' if success else 'failed to refresh'}",
                "success": success,
                "content_type": content_type,
                "unified_rebuilt": request.rebuild_unified and success,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Refresh all content (documents and videos)
            scan_and_process_all_content()

            doc_count = sum(1 for m in document_metadata.values() if m.get('content_type') != 'video')
            video_count = sum(1 for m in document_metadata.values() if m.get('content_type') == 'video')

            return {
                "message": "All content refreshed",
                "documents_processed": doc_count,
                "videos_processed": video_count,
                "total_items": len(document_stores),
                "unified_index_ready": unified_store is not None,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get detailed system status"""
    # Separate documents and videos
    doc_items = [f for f, m in document_metadata.items() if m.get('content_type') != 'video']
    video_items = [f for f, m in document_metadata.items() if m.get('content_type') == 'video']

    # Get writer statistics
    writer_manager = get_writer_manager()
    writer_stats = writer_manager.get_document_stats()

    return {
        "rag_model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "content_loaded": {
            "total_items": len(document_stores),
            "documents": len(doc_items),
            "videos": len(video_items),
            "document_list": doc_items,
            "video_list": video_items
        },
        "unified_index": {
            "ready": unified_store is not None,
            "total_chunks": unified_metadata.get('total_chunks', 0) if unified_metadata else 0,
            "source_files": unified_metadata.get('source_files', []) if unified_metadata else [],
            "created_at": unified_metadata.get('created_at') if unified_metadata else None,
            "includes_videos": len(video_items) > 0
        },
        "video_analysis": {
            "available": asr_model is not None,
            "device": DEVICE,
            "models_loaded": {
                "whisper": asr_model is not None,
                "blip": blip_model is not None,
                "mbart": summariser_model is not None
            },
            "videos_in_rag": len(video_items)
        },
        "writers": writer_stats,
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
# Writer Management API Endpoints
# ============================================================================

class WriterSearchRequest(BaseModel):
    query: str

class WriterAliasRequest(BaseModel):
    writer_id: str
    alias: str

class WriterMergeRequest(BaseModel):
    writer_id1: str
    writer_id2: str
    keep_id: Optional[str] = None

@app.get("/api/writers")
async def list_writers():
    """List all writers in the database"""
    try:
        writer_manager = get_writer_manager()
        writers = writer_manager.list_all_writers()

        return {
            "writers": writers,
            "total_count": len(writers),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing writers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/writers/stats")
async def get_writers_stats():
    """Get statistics about writers"""
    try:
        writer_manager = get_writer_manager()
        stats = writer_manager.get_document_stats()

        return {
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting writer stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/writers/{writer_id}")
async def get_writer_details(writer_id: str):
    """Get detailed information about a specific writer"""
    try:
        writer_manager = get_writer_manager()
        writer = writer_manager.get_writer(writer_id)

        if not writer:
            raise HTTPException(status_code=404, detail=f"Writer {writer_id} not found")

        return {
            "writer": writer,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting writer {writer_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/writers/search")
async def search_writers(request: WriterSearchRequest):
    """Search for writers by name or information"""
    try:
        writer_manager = get_writer_manager()
        results = writer_manager.search_writers(request.query)

        return {
            "query": request.query,
            "results": results,
            "total_results": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error searching writers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/writers/by-name/{name}")
async def get_writer_by_name(name: str):
    """Get writer by name (with fuzzy matching)"""
    try:
        writer_manager = get_writer_manager()
        writer = writer_manager.get_writer_by_name(name)

        if not writer:
            raise HTTPException(status_code=404, detail=f"Writer '{name}' not found")

        return {
            "writer": writer,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting writer by name {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{filename}/writers")
async def get_document_writers(filename: str):
    """Get all writers mentioned in a specific document or video"""
    try:
        decoded_filename = unquote(filename)
        writer_manager = get_writer_manager()
        writers = writer_manager.get_writers_by_document(decoded_filename)

        return {
            "document": decoded_filename,
            "writers": writers,
            "total_writers": len(writers),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting writers for document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/writers/add-alias")
async def add_writer_alias(request: WriterAliasRequest):
    """Add an alias to a writer"""
    try:
        writer_manager = get_writer_manager()
        writer_manager.add_alias(request.writer_id, request.alias)

        return {
            "message": f"Alias '{request.alias}' added to writer {request.writer_id}",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adding alias: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/writers/merge")
async def merge_writers(request: WriterMergeRequest):
    """Merge two writer entries (for handling duplicates)"""
    try:
        writer_manager = get_writer_manager()
        success = writer_manager.merge_writers(
            request.writer_id1,
            request.writer_id2,
            request.keep_id
        )

        if success:
            return {
                "message": f"Writers merged successfully",
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to merge writers")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging writers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# System Initialization
# ============================================================================
def initialize_rag_system():
    """Initialize the RAG document and video processing system"""
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

        # Scan and process all content (documents + videos)
        scan_and_process_all_content()

        # Count documents vs videos
        doc_count = sum(1 for m in document_metadata.values() if m.get('content_type') != 'video')
        video_count = sum(1 for m in document_metadata.values() if m.get('content_type') == 'video')

        logger.info(f"RAG system initialized! Loaded {doc_count} documents and {video_count} videos")
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

    # Initialize video analysis models
    initialize_video_analysis_models()

    # Initialize RAG system
    initialize_rag_system()

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
    print("  - Video RAG: Chat with video content (audio + visual)")
    print("  - Videos automatically processed for embeddings")
    print("  - Unified index includes both documents and videos")
    print("  - Audio transcription (Whisper)")
    print("  - Visual frame analysis (BLIP)")
    print("  - Multi-modal summarization (mBART)")
    print("  - Video streaming and file serving")
    print("  - Arabic language support")
    print("="*80)
    print("📖 Usage:")
    print("  1. Add documents to 'docs' folder")
    print("  2. Add videos to 'videos' folder")
    print("  3. Server automatically creates embeddings on startup")
    print("  4. Chat with documents/videos using /api/chat endpoint")
    print("  5. Use /api/ask_unified for cross-content queries")
    print("="*80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload when using large models
        log_level="info"
    )
