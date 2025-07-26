#!/bin/bash

echo "🔍 Diagnosing container issue..."

# Check container status
echo "📊 Container status:"
docker ps -a | grep phi3-api

# Get last logs to see what happened
echo -e "\n📋 Last 20 lines of logs:"
docker logs --tail 20 phi3-api

# Check if it's an OOM (Out of Memory) issue
echo -e "\n🧠 Checking for memory issues..."
docker logs phi3-api 2>&1 | grep -i -E "(killed|memory|oom|out of memory)" || echo "No obvious memory errors found"

# Stop and remove existing container
echo -e "\n🛑 Cleaning up existing container..."
docker stop phi3-api 2>/dev/null || true
docker rm phi3-api 2>/dev/null || true

# Check available system memory
echo -e "\n💾 System memory info:"
if command -v free >/dev/null; then
    free -h
elif command -v vm_stat >/dev/null; then
    # macOS
    echo "macOS system - checking memory..."
    vm_stat | head -5
else
    echo "Cannot determine system memory"
fi

# Option 1: Try with memory optimized version
echo -e "\n🔧 Applying memory-optimized fix..."

# Replace main.py with memory-optimized version
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
    max_length: Optional[int] = 256  # Reduced default
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
        logger.info("Loading Phi-3 model with memory optimizations...")
        
        # Use Phi-3-mini-4k-instruct model
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load model with aggressive memory optimizations
        logger.info("Loading model with memory optimizations...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 even on CPU to save memory
            device_map=None,  # Load on CPU to avoid GPU memory issues
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            use_cache=False  # Disable KV cache to save memory
        )
        
        # Explicitly move to CPU and optimize
        model = model.to('cpu')
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully with memory optimizations!")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Phi-3 FastAPI Server is running!", "memory_optimized": True}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "memory_optimized": True,
        "device": str(next(model.parameters()).device) if model else "unknown"
    }

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
        inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response with memory optimizations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=min(request.max_length, 512),  # Cap max length
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache
                num_beams=1  # Use greedy decoding to save memory
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        # Clean up memory
        del inputs, outputs
        
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
        "dtype": str(next(model.parameters()).dtype),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "memory_optimized": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Rebuild with memory optimizations
echo -e "\n🔨 Rebuilding with memory optimizations..."
docker build -t phi3-fastapi .

if [ $? -eq 0 ]; then
    echo "✅ Docker image rebuilt successfully"
    
    echo -e "\n🚀 Starting container with increased memory limits..."
    # Start with explicit memory limits (adjust based on your system)
    docker run -d \
        --name phi3-api \
        -p 8000:8000 \
        -v $(pwd)/models:/app/models \
        --memory=6g \
        --memory-swap=8g \
        --oom-kill-disable=false \
        phi3-fastapi
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started with memory optimizations!"
        echo -e "\n📋 Monitoring startup..."
        echo "   View logs: docker logs -f phi3-api"
        echo "   Check health: curl http://localhost:8000/health"
        echo ""
        echo "⏳ Wait 2-5 minutes for model loading with optimizations..."
        
        # Brief monitoring
        sleep 5
        if docker ps | grep -q phi3-api; then
            echo "✅ Container is running"
            echo "📊 Container stats:"
            docker stats phi3-api --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
        else
            echo "⚠️ Container may have issues. Check: docker logs phi3-api"
        fi
    else
        echo "❌ Failed to start container"
        exit 1
    fi
else
    echo "❌ Failed to rebuild Docker image"
    exit 1
fi