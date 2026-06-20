# Use Python 3.10 as the base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (for faster rebuilds)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy all project files into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]