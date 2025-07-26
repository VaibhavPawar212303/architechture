# Stage 1: Builder (download the model)
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir huggingface-hub>=0.17.1

# Avoid storing secrets in layers
ENV HUGGINGFACE_HUB_ADD_TOKEN_AS_GIT_CREDENTIAL=false
ARG HF_AUTH_TOKEN

# Login to Hugging Face and download the model
RUN huggingface-cli login --token $HF_AUTH_TOKEN
RUN huggingface-cli download microsoft/Phi-3.5-vision-instruct-onnx \
    --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* \
    --local-dir .

# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app

# Install OS and Python deps
RUN apt-get update && apt-get install -y \
    build-essential python3-dev git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install required Python libraries
RUN pip install --no-cache-dir git+https://github.com/microsoft/onnxruntime-genai.git
RUN pip install --no-cache-dir gradio

# Copy the model files and app code
COPY --from=builder /app .
COPY app.py /app/app.py

EXPOSE 4050
CMD ["python", "app.py"]
