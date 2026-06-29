# Base image python ki
FROM python:3.10-slim

# System dependencies jo OpenCV ke liye chahiye (cloud par crash hone se bachane ke liye)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Working directory set karo
WORKDIR /app

# Sabse pehle requirements copy aur install karo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Poora project copy karo
COPY . .

# Streamlit ke liye port expose karo
EXPOSE 8501

# App chalane ki command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]