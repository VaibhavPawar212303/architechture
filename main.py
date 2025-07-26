import os
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_ID = os.getenv("MODEL_ID", "microsoft/Phi-3-vision-128k-instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500"))
PORT = int(os.getenv("PORT", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# FastAPI app initialization
app = FastAPI(
    title="Phi-3 Vision API",
    description="API for image analysis using Microsoft's Phi-3 Vision model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None
model_load_time = None

# Pydantic models
class VisionRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to analyze the image")
    image_url: HttpUrl = Field(..., description="URL of the image to analyze")
    max_tokens: Optional[int] = Field(None, ge=1, le=1000, description="Maximum tokens to generate")

class VisionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    generated_text: str = Field(..., description="The generated analysis text")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: dict = Field(..., description="Model information")

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    timestamp: str
    device: str
    model_loaded: bool
    model_id: str
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Global variable to track startup time
startup_time = time.time()

def load_model():
    """Load the Phi-3 Vision model and processor"""
    global model, processor, model_load_time
    
    start_time = time.time()
    logger.info(f"Loading model {MODEL_ID} on device: {DEVICE}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map=DEVICE, 
            trust_remote_code=True, 
            torch_dtype=DTYPE,
            attn_implementation="eager"  # Disable FlashAttention2 for CPU/compatibility
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        model_load_time = time.time() - start_time
        logger.info(f"Model and processor loaded successfully in {model_load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Phi-3 Vision API...")
    load_model()
    logger.info("Startup complete!")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/", response_model=dict)
def root():
    """Root endpoint with API information"""
    return {
        "message": "Phi-3 Vision API is running",
        "version": "1.0.0",
        "model_id": MODEL_ID,
        "device": DEVICE,
        "endpoints": {
            "health": "/health",
            "analyze": "/analyse-image",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        device=DEVICE,
        model_loaded=model is not None,
        model_id=MODEL_ID,
        uptime_seconds=round(uptime, 2)
    )

@app.post("/analyse-image", response_model=VisionResponse)
def analyse_image(request: VisionRequest):
    """Analyze an image using the Phi-3 Vision model"""
    
    if model is None or processor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check the server logs."
        )
    
    start_time = time.time()
    
    try:
        # Download and process the image
        logger.info(f"Fetching image from: {request.image_url}")
        response = requests.get(str(request.image_url), timeout=30)
        response.raise_for_status()
        
        # Validate image
        try:
            image = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Prepare the messages for the model
        messages = [{"role": "user", "content": f"<|image_1|>\n{request.prompt}"}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = processor(prompt, [image], return_tensors="pt").to(DEVICE)
        
        # Generate response
        max_tokens = request.max_tokens or MAX_NEW_TOKENS
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                eos_token_id=processor.tokenizer.eos_token_id, 
                max_new_tokens=max_tokens, 
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode the response
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response_text = processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        processing_time = time.time() - start_time
        
        logger.info(f"Successfully processed image in {processing_time:.2f} seconds")
        
        return VisionResponse(
            generated_text=response_text,
            processing_time=round(processing_time, 2),
            model_info={
                "model_id": MODEL_ID,
                "device": DEVICE,
                "max_tokens_used": max_tokens
            }
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Could not fetch image: {str(e)}"
        )
    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA out of memory error")
        raise HTTPException(
            status_code=507, 
            detail="Insufficient GPU memory. Try with a smaller image or reduce max_tokens."
        )
    except Exception as e:
        logger.error(f"Error during image analysis: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error during image analysis: {str(e)}"
        )

@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "cuda_available": torch.cuda.is_available(),
        "model_load_time": model_load_time,
        "max_tokens": MAX_NEW_TOKENS
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        log_level="info"
    )