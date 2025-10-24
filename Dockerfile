# ---------- Base ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# System deps that TF/PIL/Streamlit commonly need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# ---------- Workdir ----------
WORKDIR /app
COPY . /app/
# Copy dependency manifests first (better layer caching)
 

# Install uv and then resolve deps into system environment
RUN pip install --upgrade pip && pip install --no-cache-dir uv && \
    uv pip install --system -r /app/requirements.txt

EXPOSE 7860

# Optional healthcheck
HEALTHCHECK CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Run Streamlit on the port provided by Spaces
CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]
