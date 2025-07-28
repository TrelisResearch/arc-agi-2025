FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY sandbox_requirements.txt .
RUN pip install --no-cache-dir -r sandbox_requirements.txt

# Copy the server code and required executor modules
COPY sandbox_server.py .
COPY subprocess_executor.py .

# Expose the port
EXPOSE 7934

# Run the server
CMD ["python", "sandbox_server.py"]
