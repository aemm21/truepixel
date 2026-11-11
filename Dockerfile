FROM python:3.11-slim

WORKDIR /app

# Instalamos PyTorch CPU (wheel ligera) y las dependencias del proyecto
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TRUEPIXEL_MODEL_PATH=/app/truepixel_v1.pth \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
