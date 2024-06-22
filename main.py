from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch
import whisper
import os
from tempfile import NamedTemporaryFile

# Initialize FastAPI app
app = FastAPI()

# Mount static files and set up templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Check device availability and load models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)
summarization_model = pipeline("summarization")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/whisper")
async def handle_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="File not provided")
    
    try:
        with NamedTemporaryFile(delete=False) as temp:
            temp.write(file.file.read())
            temp_path = temp.name
        
        # Transcribe the audio
        result = whisper_model.transcribe(temp_path)
        transcription_text = result["text"]
        
        # Summarize the transcription
        summary = summarization_model(transcription_text, max_length=150, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text']
        
        # Extract timestamps
        timestamps = extract_timestamps(result["segments"])
        
        response_data = {
            "filename": file.filename,
            "transcription": transcription_text,
            "summary": summary_text,
            "timestamps": timestamps
        }
        
        return JSONResponse(content=response_data)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_timestamps(segments):
    timestamps = []
    for segment in segments:
        timestamps.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text']
        })
    return timestamps
