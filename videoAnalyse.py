from fastapi import FastAPI, File, UploadFile
import uvicorn
import tempfile
import subprocess
import torch
from transformers import (
    MBartForConditionalGeneration, 
    MBart50Tokenizer,
    BlipProcessor, 
    BlipForConditionalGeneration
)
import whisper
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Whisper (GPU)
asr_model = whisper.load_model("large").to(DEVICE)

# Load mBART (GPU)
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
summariser_model = MBartForConditionalGeneration.from_pretrained(model_name).to(DEVICE, dtype=torch.float32)

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(DEVICE)

def extract_frames(video_path, num_frames=10):
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

def analyze_frames(frames):
    """Generate captions for extracted frames"""
    captions = []
    
    for i, frame in enumerate(frames):
        # Generate caption for each frame
        inputs = blip_processor(frame, return_tensors="pt").to(DEVICE)
        out = blip_model.generate(**inputs, max_length=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        captions.append(f"Frame {i+1}: {caption}")
    
    return captions

def extract_key_visual_features(frames):
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

@app.post("/summarise")
async def summarise(file: UploadFile = File(...)):
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Extract audio with ffmpeg
        audio_path = tmp_path + ".wav"
        subprocess.run([
            "ffmpeg", "-i", tmp_path, "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], check=True)
        
        # 1. AUDIO ANALYSIS
        # Transcribe with Whisper
        result = asr_model.transcribe(audio_path, task="transcribe")
        transcript = result["text"]
        detected_lang = result["language"]
        print("Transcript: " + transcript)
        
        # 2. VIDEO/IMAGE ANALYSIS
        # Extract frames from video
        frames = extract_frames(tmp_path, num_frames=10)
        
        # Generate frame captions
        frame_captions = analyze_frames(frames)
        
        # Get visual summary
        visual_summary = extract_key_visual_features(frames)
        
        # 3. COMBINE AUDIO AND VISUAL INFORMATION
        # Create a combined context
        combined_text = f"""
        Audio Content: {transcript}
        
        Visual Content: {visual_summary}
        """
        
        # 4. GENERATE ARABIC SUMMARY
        # Set target language to Arabic
        forced_bos_token_id = tokenizer.lang_code_to_id["ar_AR"]
        
        # Encode combined input
        inputs = tokenizer(
            combined_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(DEVICE)
        
        # Generate summary (to Arabic)
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
            "audio_transcript": transcript,
            "visual_descriptions": frame_captions,
            "visual_summary": visual_summary,
            "detected_language": detected_lang,
            "num_frames_analyzed": len(frames),
            "device": DEVICE
        }
    
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)