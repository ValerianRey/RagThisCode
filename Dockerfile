# syntax=docker/dockerfile:1.7

# Use an official Python base image compatible with requires-python  3.13
FROM python:3.13-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only project files needed for dependency resolution first for better caching
COPY pyproject.toml ./

# Install project dependencies into the system interpreter using uv
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the app
COPY . .

# Ensure data directory exists (mounted volume can override)
RUN mkdir -p /app/data/chroma_langchain_db

EXPOSE 9000

# Default command: run the server with uv's Python
CMD ["uv", "run", "python", "server.py"]
