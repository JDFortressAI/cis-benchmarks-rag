# Use a slim Python 3.11 base image for minimal footprint
FROM python:3.11-slim

# Set working directory.
WORKDIR /app

# Install uv 
RUN pip install --no-cache-dir uv

# Copy pyproject.toml and uv.lock first for caching dependency layers.
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv (frozen for reproducibility; no dev groups).
RUN uv sync --frozen --no-cache --no-install-project

# Copy the rest of the project code (assume .gitignore excludes .venv, uv.lock, etc.).
COPY . .

RUN chmod -R a+r /app/images

# Create user before downloading models so cache has correct permissions
RUN useradd -m appuser

# Pre-download the reranker model and tokenizer as appuser to bake them into the image.
USER appuser
RUN /app/.venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('BAAI/bge-reranker-large'); \
    AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')"

# Expose Streamlit's default port.
EXPOSE 8501

# Run the Streamlit app (adjust if main file is not app.py; add --server.port=8501 for explicit port).
CMD ["/app/.venv/bin/streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]