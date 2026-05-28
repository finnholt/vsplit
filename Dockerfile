FROM python:3.12-slim

# Swap Debian apt source to Aliyun mirror (much faster from China)
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources

# System deps: ffmpeg (required), git (uv installs from VCS)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyPI mirror for pip + uv (Aliyun)
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

# uv: lightweight Python package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install third-party deps first (better layer caching).
# --no-install-project skips building vsplit itself so the cache layer
# stays valid as long as pyproject.toml/uv.lock don't change.
COPY pyproject.toml uv.lock README.md ./

# Rewrite locked package sources from pypi.org to Aliyun mirror at build
# time. This keeps the committed lock file portable (still works abroad)
# while drastically speeding up builds from mainland China.
RUN sed -i 's|https://pypi.org/simple|https://mirrors.aliyun.com/pypi/simple|g; s|https://files.pythonhosted.org|https://mirrors.aliyun.com/pypi|g' uv.lock || true

RUN uv sync --frozen --no-dev --no-install-project

# Then bring in the rest of the source and install the project itself.
COPY . .
RUN uv sync --frozen --no-dev

# Streamlit defaults
EXPOSE 8501
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.maxUploadSize=2048"]
