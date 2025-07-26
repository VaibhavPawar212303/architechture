# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import requests
from io import BytesIO

app = FastAPI(title="Phi-3 Vision API")

MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on device: {DEVICE}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=DEVICE, trust_remote_code=True, torch_dtype=DTYPE
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Model and processor loaded successfully.")

class VisionRequest(BaseModel):
    prompt: str
    image_url: HttpUrl
class VisionResponse(BaseModel):
    generated_text: str

@app.post("/analyse-image", response_model=VisionResponse)
def analyse_image(request: VisionRequest):
    try:
        response = requests.get(str(request.image_url))
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")

    messages = [{"role": "user", "content": f"<|image_1|>\n{request.prompt}"}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(DEVICE)

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=500, do_sample=False)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return VisionResponse(generated_text=response_text)

@app.get("/")
def root():
    return {"message": "Phi-3 Vision API is running."}