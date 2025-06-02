# Get python image (latest version)
FROM python:latest

# Set working directory to /app
WORKDIR /app

# Copy only the requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY . .

# Expose port 8000
EXPOSE 8000

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "apis:app", "--host", "0.0.0.0", "--port", "8000"]
