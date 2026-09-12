FROM osgeo/gdal:ubuntu-small-3.6.3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install dependencies
COPY pyproject.toml /app/
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install .

COPY . /app/

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
