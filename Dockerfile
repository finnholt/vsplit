FROM python:3.12-slim

# System deps: ffmpeg (required), git (uv installs from VCS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: lightweight Python package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install Python deps first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Then bring in the rest of the source
COPY . .

# Streamlit defaults
EXPOSE 8501
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.maxUploadSize=2048"]
