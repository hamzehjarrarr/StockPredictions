# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY src/ ./src/
COPY app.py .
COPY app_streamlit.py .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Start FastAPI and Streamlit together
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app_streamlit.py --server.port 8501"]