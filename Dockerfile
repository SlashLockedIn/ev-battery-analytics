FROM python:3.11-slim

# Prevent Python from writing .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but safe for many libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose port for local
EXPOSE 5000

# Run with gunicorn (production)
# IMPORTANT: app:app -> app.py has "app = Flask(...)"
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT:-5000} webapp.app:app"]