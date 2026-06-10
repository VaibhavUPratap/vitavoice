FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (libsndfile1 is critical for librosa/soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download the Wav2Vec 2.0 model weights to bake them into the image
# This ensures fast container starts and offline reliability
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2Model; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"

# Copy backend app and ml code
COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY datasets/ /app/datasets/

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
