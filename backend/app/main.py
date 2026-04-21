from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import tempfile
from pathlib import Path

# Import your models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.fusion_model import fusion_detector
from app.utils.visualization import create_confidence_chart, create_comparison_chart, create_radar_chart
from app.chatbot.feedback import chatbot

# Initialize FastAPI app
app = FastAPI(title="Fake News Detection API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory for uploads
TEMP_DIR = tempfile.mkdtemp()
os.makedirs(TEMP_DIR, exist_ok=True)

# Get the absolute path to frontend directory
FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"

# Request/Response models
class TextRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str
    analysis_context: Optional[dict] = None

# API Endpoints

@app.get("/")
async def root():
    """Serve the main HTML page"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        return {"message": "Fake News Detection API is running", "status": "active", "error": "Frontend not found"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "frontend_path": str(FRONTEND_DIR)}

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    """Predict if a text news article is fake or real"""
    try:
        result = fusion_detector.predict(text=request.text)
        
        # Generate visualizations
        viz_chart = create_confidence_chart(result['fake_score'], result['confidence'])
        
        if len(result['visualization_data']['components']) > 1:
            comp_chart = create_comparison_chart(result['visualization_data']['components'])
        else:
            comp_chart = viz_chart
        
        return {
            **result,
            'visualization_chart': viz_chart,
            'comparison_chart': comp_chart
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict if an image is real or fake"""
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = fusion_detector.predict(image_path=file_path)
        
        # Generate visualizations
        viz_chart = create_confidence_chart(result['fake_score'], result['confidence'])
        
        if len(result['visualization_data']['components']) > 1:
            comp_chart = create_comparison_chart(result['visualization_data']['components'])
        else:
            comp_chart = viz_chart
        
        # Clean up
        os.remove(file_path)
        
        return {
            **result,
            'visualization_chart': viz_chart,
            'comparison_chart': comp_chart
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/both")
async def predict_both(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """Predict using both text and image"""
    try:
        # Save image temporarily
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = fusion_detector.predict(text=text, image_path=file_path)
        
        # Generate multiple visualizations
        viz_chart = create_confidence_chart(result['fake_score'], result['confidence'])
        comp_chart = create_comparison_chart(result['visualization_data']['components'])
        
        # Create radar chart if multiple components
        if len(result['visualization_data']['components']) >= 3:
            radar_chart = create_radar_chart(result['visualization_data']['components'])
        else:
            radar_chart = comp_chart
        
        # Clean up
        os.remove(file_path)
        
        return {
            **result,
            'visualization_chart': viz_chart,
            'comparison_chart': comp_chart,
            'radar_chart': radar_chart
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot")
async def chat_with_bot(request: ChatRequest):
    """Get chatbot feedback about the analysis"""
    try:
        response = chatbot.get_response(request.message, request.analysis_context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")