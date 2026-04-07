# ── RouterEnv-v1 Production Dockerfile ─────────────────────────────────────
# Standard OpenEnv environment deployment for Hugging Face Spaces.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ✅ Install system dependencies and uv for high-speed package installs
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ✅ HF Spaces: run as user 1000 (required for openenv tag)
RUN useradd -m -u 1000 user
ENV PATH="/root/.local/bin:/home/user/.local/bin:${PATH}"

# ✅ Set working directory
WORKDIR /app

# ✅ Copy requirements and install
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# ✅ Copy source code with correct ownership
COPY --chown=user:user . .

USER user

# ✅ HF Spaces expects port 7860 (not 8000)
EXPOSE 7860

# ✅ Hardware metadata
LABEL "ai.openenv.cpu"="2"
LABEL "ai.openenv.memory"="8GB"

# ✅ Start server on 0.0.0.0:7860
CMD ["python", "-m", "router_env.server", "--host", "0.0.0.0", "--port", "7860"]
