# Use Python 3.12 to match your project
FROM python:3.12-slim

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system packages (glibc locales, build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Copy only dependency files first (better layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# If you use TensorFlow on CPU under Python 3.12, pin 2.16+ in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code and models
COPY . /app

# Streamlit needs to run on the port HF provides ($PORT) and bind 0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose default local port (HF Spaces will set $PORT env)
EXPOSE 7860

# Start Streamlit (use the PORT env if present, fall back to 7860 locally)
CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
