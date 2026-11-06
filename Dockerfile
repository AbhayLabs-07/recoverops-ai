# RecoverOps AI - Phase 1 Core Engine Dockerfile
# Base: Official minimal Python 3.11 Debian image
FROM python:3.11-slim

# Prevent generation of .pyc files and enable unbuffered console output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy minimal configuration, source code, test suite, and documentation
COPY requirements.txt .
COPY .python-version .
COPY README.md .
COPY app/ app/
COPY tests/ tests/
COPY docs/ docs/

# Validate suite during container build to ensure artifact integrity
RUN python -m unittest discover -s tests -v

# Default entrypoint executes the 3 demonstration pipelines
CMD ["python", "-m", "app.main"]

