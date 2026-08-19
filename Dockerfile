FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen --no-dev

COPY . .

RUN mkdir -p data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "voiceraghh.server:app", "--host", "0.0.0.0", "--port", "8000"]
