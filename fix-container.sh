#!/bin/bash

echo "🔧 Fixing NumPy compatibility issue..."

# Stop the current container
echo "🛑 Stopping current container..."
docker stop phi3-api 2>/dev/null || true
docker rm phi3-api 2>/dev/null || true

# Remove the old image
echo "🗑️ Removing old image..."
docker rmi phi3-fastapi 2>/dev/null || true

# Update requirements.txt to fix NumPy compatibility
echo "📝 Updating requirements.txt with NumPy fix..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.36.0
torch==2.1.0
tokenizers==0.15.0
accelerate==0.24.1
pydantic==2.5.0
python-multipart==0.0.6
requests==2.31.0
numpy<2.0.0
EOF

# Update main.py to fix flash-attention warning
echo "📝 Updating main.py with attention implementation fix..."
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Phi-3 API", description="FastAPI server with Microsoft Phi-3 model")

# Global variables for model and tokenizer
model = None
tokenizer = None

class ChatRequest(BaseModel):
    message: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class ChatResponse(BaseModel):
    response: str
    status: str

@app.on_event("startup")
async def load_model():
    """Load the Phi-3 model and tokenizer on startup"""
    global model, tokenizer
    
    try:
        logger.info("Loading Phi-3 model...")
        
        # Use Phi-3-mini-4k-instruct model (smaller, faster)
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager"  # Use eager attention to avoid flash-attention warnings
        )
        
        # Move to CPU if CUDA is not available
        if not torch.cuda.is_available():
            model = model.to('cpu')
            
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Phi-3 FastAPI Server is running!"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate response using Phi-3 model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format the message for Phi-3
        formatted_prompt = f"<|user|>\n{request.message}<|end|>\n<|assistant|>\n"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return ChatResponse(response=response, status="success")
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    global model, tokenizer
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "device": str(next(model.parameters()).device),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "🔨 Rebuilding Docker image with fixes..."
docker build -t phi3-fastapi .

if [ $? -eq 0 ]; then
    echo "✅ Docker image rebuilt successfully"
    
    echo "🚀 Starting fixed container..."
    docker run -d --name phi3-api -p 8000:8000 -v $(pwd)/models:/app/models phi3-fastapi
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully!"
        echo "📋 Monitoring startup (this will take a few minutes)..."
        echo "   View logs: docker logs -f phi3-api"
        echo "   Check status: curl http://localhost:8000/health"
        echo ""
        echo "⏳ The model is loading... Check logs to monitor progress."
    else
        echo "❌ Failed to start container"
        exit 1
    fi
else
    echo "❌ Failed to rebuild Docker image"
    exit 1
fi