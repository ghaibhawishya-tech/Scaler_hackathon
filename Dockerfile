FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN useradd -m -u 1000 user
ENV PATH="/root/.local/bin:/home/user/.local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
# ✅ FIX: added --no-cache to ensure fresh install every time
RUN uv pip install --system --no-cache -r requirements.txt

COPY --chown=user:user . .
USER user

EXPOSE 7860

LABEL "ai.openenv.cpu"="2"
LABEL "ai.openenv.memory"="8GB"

CMD ["python", "-m", "router_env.server", "--host", "0.0.0.0", "--port", "7860"]