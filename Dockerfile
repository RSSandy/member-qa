FROM python:3.11-slim

WORKDIR /app

# Install dependencies early (faster Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port Render will use
ENV PORT=10000
EXPOSE 10000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
