FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY frontend/ .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]